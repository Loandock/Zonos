import asyncio
import json
import os
import time
import torch
import torchaudio
import websockets
from io import BytesIO
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import torch._dynamo as dynamo
import threading
import functools
import inspect
import warnings
import re

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda.graphs")

# Path to your custom voice sample
CUSTOM_VOICE_PATH = "sesame_fine_tune.wav"
HARDCODED_SPEECH = "Heyy! So... sure lemme walk you through the mortgage process. Basically, we start by looking at your finances, like your credit and income, and then we figure out which loan works best for you. We run these numbers through the underwriter's software to kinda se if they'll let you take it out, you know? It might sound like a lot at first, but I'm here to help you every step of the way. So, if that sounds good, just lemme know, and we can get started right away!"

# GH200-specific optimizations
torch.set_float32_matmul_precision('high')  # Use TF32 for matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(True)

# Grace Hopper specific memory optimization
torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory

# Optimize CUDA kernel launches
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_COMPILE_PARALLEL"] = "1"

# Pre-compile model with TorchDynamo for GH200
dynamo.config.cache_size_limit = 512  # Increase cache size for compiled graphs

print("Loading model...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
model.requires_grad_(False).eval()

# Pre-load the speaker embedding
print(f"Loading custom voice from {CUSTOM_VOICE_PATH}...")
custom_wav, custom_sr = torchaudio.load(CUSTOM_VOICE_PATH)

# Use full audio file for better voice cloning
print(f"Using full audio file: {custom_wav.shape[1]/custom_sr:.1f}s")
custom_speaker_embedding = model.make_speaker_embedding(custom_wav, custom_sr)
print("Custom voice loaded successfully!")

# Warmup the model for faster first inference
print("Performing GH200 warmup...")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    warmup_text = "This is a warmup text to prime the model for faster inference."
    cond_dict = make_cond_dict(
        text=warmup_text,
        speaker=custom_speaker_embedding,
        language="en-us",
        fmax=22050
    )
    conditioning = model.prepare_conditioning(cond_dict)
    
    # Warm up the streaming path
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,
        chunk_schedule=[32, 64, 128],
        chunk_overlap=8,
    )
    
    # Process a few chunks
    for i, _ in enumerate(stream_generator):
        if i >= 3:  # Just process a few chunks
            break
    
    # Force synchronization to complete warmup
    torch.cuda.synchronize()
print("GH200 warmup complete")

# Pre-allocate WAV buffer pool
buffer_pool = [BytesIO() for _ in range(10)]
buffer_idx = 0

def get_next_buffer():
    global buffer_idx
    buffer = buffer_pool[buffer_idx]
    buffer.seek(0)
    buffer.truncate(0)
    buffer_idx = (buffer_idx + 1) % len(buffer_pool)
    return buffer

def numpy_to_wav(audio_tensor, sampling_rate=44100):
    """Convert tensor to WAV bytes using buffer pool"""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    buffer = get_next_buffer()
    torchaudio.save(buffer, audio_tensor.cpu(), sampling_rate, format="wav")
    buffer.seek(0)
    return buffer.read()

class StreamingTTSSession:
    # Class-level concurrency control
    MAX_CONCURRENT_GENERATIONS = 3
    generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    
    def __init__(self, websocket, language="en-us"):
        self.websocket = websocket
        self.language = language
        self.speaker_embedding = custom_speaker_embedding
        self.text_buffer = ""
        self.is_generating = False
        self.generation_task = None
        self.sentence_end_markers = ['.', '!', '?', '\n']  # Simplified to focus on sentences
        
    async def add_text(self, text):
        """Add text to the buffer and trigger generation if not already running"""
        self.text_buffer += text
        
        # Only start generation if we have enough complete text and not already generating
        if not self.is_generating and self.has_complete_sentence():
            self.is_generating = True
            self.generation_task = asyncio.create_task(self.generate_from_buffer())
    
    def has_complete_sentence(self):
        """Check if we have a complete sentence or substantial text"""
        if not self.text_buffer:
            return False
            
        # Check for sentence ending markers
        for marker in self.sentence_end_markers:
            if marker in self.text_buffer:
                return True
                
        # For longer chunks without sentence markers, consider them complete
        if len(self.text_buffer.split()) >= 10:  # Longer chunks for GH200
            return True
                
        return False
    
    def get_complete_text_chunk(self):
        """Get text up to a complete sentence, preferring natural breaks"""
        if not self.text_buffer:
            return ""
        
        # Try to find complete sentences first
        for marker in self.sentence_end_markers:
            marker_index = self.text_buffer.find(marker)
            if marker_index != -1:
                # Skip if it's a decimal point in a number (e.g., "45.5%")
                if marker == '.':
                    # Check if it's between digits or followed by % or other numeric indicators
                    if marker_index > 0 and marker_index < len(self.text_buffer) - 1:
                        prev_char = self.text_buffer[marker_index-1]
                        next_char = self.text_buffer[marker_index+1]
                        
                        # Handle decimal numbers (e.g., "45.5")
                        if prev_char.isdigit() and (next_char.isdigit() or next_char == '%'):
                            continue
                        
                        # Handle abbreviations (e.g., "Dr.", "Inc.", "etc.")
                        common_abbrevs = ["Mr", "Mrs", "Dr", "Prof", "Inc", "Co", "Ltd", "etc", "vs"]
                        for abbrev in common_abbrevs:
                            if marker_index >= len(abbrev) and self.text_buffer[marker_index-len(abbrev):marker_index] == abbrev:
                                # Ensure it's really the end of the abbreviation, not within a word
                                if marker_index-len(abbrev) == 0 or not self.text_buffer[marker_index-len(abbrev)-1].isalpha():
                                    continue
                
                # Include the marker and any following space
                end_index = marker_index + 1
                
                # Look ahead for closing quotes, parentheses, and other ending punctuation
                while end_index < len(self.text_buffer) and self.text_buffer[end_index] in ['"', "'", ')', ']', '}']:
                    end_index += 1
                    
                # Include any following space or newline
                if end_index < len(self.text_buffer) and self.text_buffer[end_index] in [' ', '\n']:
                    end_index += 1
                    
                return self.text_buffer[:end_index]
        
        # If no sentence markers, check for longer chunks of text (GH200 can handle longer chunks)
        if len(self.text_buffer) > 300:  # Process larger chunks on GH200
            # Find a good break point like a comma
            for marker in [',', ';', ':', ' - ']:
                marker_index = self.text_buffer.rfind(marker, 0, 300)
                if marker_index != -1:
                    end_index = marker_index + 1
                    if end_index < len(self.text_buffer) and self.text_buffer[end_index] == ' ':
                        end_index += 1
                    return self.text_buffer[:end_index]
            
            # If no good break point, just take a large chunk
            words = self.text_buffer[:300].split()
            if len(words) > 1:
                return " ".join(words[:-1]) + " "  # Return up to the last word to avoid cutting words
            return self.text_buffer[:300]
        
        # For shorter text, return if it ends with space (likely a complete phrase)
        if self.text_buffer.endswith(" ") and len(self.text_buffer.split()) > 3:
            return self.text_buffer
        
        # Otherwise, wait for more text
        return ""

    def detect_content_type(self, text):
        """Detect the type of content to adjust TTS parameters accordingly"""
        text = text.lower().strip()
        content_type = {
            'is_greeting': False,
            'is_question': False,
            'is_exclamation': False,
            'is_list': False,
            'contains_numbers': False,
            'speaking_rate_modifier': 0  # Adjustment to default speaking rate
        }
        
        # Detect greetings
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'welcome']
        if any(text.startswith(pattern) for pattern in greeting_patterns):
            content_type['is_greeting'] = True
            content_type['speaking_rate_modifier'] = -1  # Slightly slower
        
        # Detect questions
        if '?' in text:
            content_type['is_question'] = True
            content_type['speaking_rate_modifier'] = 1  # Slightly faster
        
        # Detect exclamations
        if '!' in text:
            content_type['is_exclamation'] = True
            content_type['speaking_rate_modifier'] = 2  # Faster for excitement
        
        # Detect numeric content
        if any(char.isdigit() for char in text):
            content_type['contains_numbers'] = True
            
        return content_type
    
    async def generate_from_buffer(self):
        """Generate audio from the current text buffer"""
        try:
            # Limit concurrent generations
            async with StreamingTTSSession.generation_semaphore:
                # Get text up to a natural boundary
                complete_text = self.get_complete_text_chunk()
                if not complete_text:
                    self.is_generating = False
                    return
                    
                # Remove the processed text from the buffer
                self.text_buffer = self.text_buffer[len(complete_text):]
                
                print(f"Processing chunk: '{complete_text}'")
                print(f"Remaining buffer: '{self.text_buffer}'")
                
                # Set a random seed for reproducibility
                torch.manual_seed(421)
                
                # Detect content type and adjust parameters
                content_type = self.detect_content_type(complete_text)
                
                # Adjust speaking rate based on content type
                speaking_rate = 13  # Base rate
                speaking_rate += content_type['speaking_rate_modifier']
                
                # Create conditioning dictionary with optimized parameters for voice cloning
                print(f"Creating conditioning for text: '{complete_text}'")
                cond_dict = make_cond_dict(
                    text=complete_text, 
                    speaker=self.speaker_embedding, 
                    language=self.language,
                    fmax=22050,  # Optimal for voice cloning
                    speaking_rate=speaking_rate  # Adjusted based on content type
                )
                
                # Stream generation parameters
                print("Starting streaming generation...")
                start_time = time.time()
                
                # Send initial metadata
                await self.websocket.send(json.dumps({
                    "type": "metadata",
                    "sampling_rate": model.autoencoder.sampling_rate,
                    "text": complete_text
                }))
                
                # Generate and stream audio chunks
                chunk_counter = 0
                
                # GH200-optimized generation with BFloat16
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        # Prepare conditioning - do this outside the stream to avoid overhead
                        conditioning = model.prepare_conditioning(cond_dict)
                        
                        # Create a GH200-optimized chunk schedule for maximum efficiency
                        stream_generator = model.stream(
                            prefix_conditioning=conditioning,
                            audio_prefix_codes=None,
                            chunk_schedule=[64, 64, 64, 64],  # Smaller initial chunks for faster startup
                            chunk_overlap=32,  # Overlap smaller than smallest chunk
                        )
                        
                        # Buffer for batching audio chunks
                        audio_buffer = []
                        buffer_size_limit = 2  # Smaller batches for more frequent updates
                        buffer_time_limit = 0.01  # Shorter time limit for more responsive delivery
                        last_send_time = time.time()
                        
                        for audio_chunk in stream_generator:
                            if audio_chunk is None:
                                continue
                                
                            # Keep tensor on GPU as long as possible
                            audio_tensor = audio_chunk
                            
                            chunk_counter += 1
                            current_time = time.time() - start_time
                            generated_time = audio_tensor.shape[0] / model.autoencoder.sampling_rate
                            
                            print(f"Generated chunk {chunk_counter}: time {current_time*1000:.0f}ms | generated {generated_time*1000:.0f}ms of audio")
                            
                            # Convert to WAV bytes (only now moving to CPU)
                            with torch.cuda.stream(torch.cuda.Stream(priority=-1)):  # Non-blocking high-priority conversion
                                audio_bytes = numpy_to_wav(audio_tensor, model.autoencoder.sampling_rate)
                                
                            # Add to buffer instead of sending immediately
                            audio_buffer.append(audio_bytes)
                            
                            # Send batched chunks if buffer is full or time limit reached
                            if len(audio_buffer) >= buffer_size_limit or (time.time() - last_send_time) > buffer_time_limit:
                                # Concatenate all chunks into a single message
                                if audio_buffer:
                                    combined_message = b''.join(audio_buffer)
                                    await self.websocket.send(combined_message)
                                    print(f"Sent batch of {len(audio_buffer)} chunks as single message ({len(combined_message)/1024:.1f} KB)")
                                    audio_buffer = []
                                    last_send_time = time.time()
                        
                        # Send any remaining chunks in the buffer
                        if audio_buffer:
                            combined_message = b''.join(audio_buffer)
                            await self.websocket.send(combined_message)
                            print(f"Sent final batch of {len(audio_buffer)} chunks as single message ({len(combined_message)/1024:.1f} KB)")
                
                # Send end of stream marker
                await self.websocket.send(json.dumps({"type": "end"}))
                total_time = time.time() - start_time
                print(f"Streaming completed in {total_time*1000:.0f}ms")
                
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.websocket.send(json.dumps({"type": "error", "message": str(e)}))
            except:
                pass
        finally:
            self.is_generating = False
            # If more text was added during generation, start a new generation
            if self.has_complete_sentence():
                self.generation_task = asyncio.create_task(self.generate_from_buffer())

    async def generate_hardcoded_speech(self):
        """Split hardcoded speech into sentences for more natural delivery"""
        # Split the hardcoded speech into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', HARDCODED_SPEECH)
        for sentence in sentences:
            if sentence.strip():
                print(f"Adding sentence: '{sentence}'")
                await self.add_text(sentence + " ")
                # Wait for this sentence to process before adding the next
                while self.is_generating:
                    await asyncio.sleep(0.1)
        
        print("All sentences queued")

async def handle_websocket(websocket):
    """Handle incoming websocket connections"""
    try:
        # Create a streaming session for this connection
        session = StreamingTTSSession(websocket, "en-us")
        
        # Auto-start with hardcoded text, split into natural sentences
        print("Auto-starting with hardcoded text (no client input needed)")
        await session.generate_hardcoded_speech()
        
        # Keep the server running after generating the hardcoded text
        await asyncio.Future()
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass

async def main():
    # Start websocket server
    host = "0.0.0.0"  # Listen on all interfaces
    port = 8765
    
    print(f"Starting GH200-optimized websocket server on {host}:{port}")
    async with websockets.serve(
        handle_websocket, 
        host, 
        port,
        max_size=10_000_000,  # 10MB max message size
        max_queue=32,         # Larger queue for GH200
        ping_timeout=300,     # Longer timeout
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Print CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # GH200 optimization: run garbage collection before starting
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    asyncio.run(main())