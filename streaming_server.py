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

# Path to your custom voice sample
CUSTOM_VOICE_PATH = "sesame_fine_tune.wav"

# === H100 OPTIMIZATIONS ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

def numpy_to_wav(audio_array, sampling_rate=44100):
    """Convert numpy array to WAV bytes"""
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
    buffer = BytesIO()
    torchaudio.save(buffer, audio_tensor, sampling_rate, format="wav")
    buffer.seek(0)
    return buffer.read()

class StreamingTTSSession:
    def __init__(self, websocket, language="en-us"):
        self.websocket = websocket
        self.language = language
        self.speaker_embedding = custom_speaker_embedding
        self.text_buffer = ""
        self.is_generating = False
        self.generation_task = None
        self.min_words_to_generate = 2  # Wait for at least 2 words before generating
        self.max_words_per_chunk = 5  # Process at most 5 words at once for faster responses
        self.sentence_end_markers = ['.', '!', '?', ';', '\n']  # End chunk at these markers
        
    async def add_text(self, text):
        """Add text to the buffer and trigger generation if not already running"""
        self.text_buffer += text
        
        # Only start generation if we have enough complete words and not already generating
        if not self.is_generating and self.has_enough_complete_words():
            self.is_generating = True
            self.generation_task = asyncio.create_task(self.generate_from_buffer())
    
    def has_enough_complete_words(self):
        """Check if we have enough complete words to start generation"""
        if not self.text_buffer:
            return False
            
        # Count complete words (separated by spaces)
        words = self.text_buffer.split()
        return len(words) >= self.min_words_to_generate
    
    def get_complete_text_chunk(self):
        """Get text up to a natural boundary with improved rules for speech synthesis"""
        if not self.text_buffer:
            return ""
        
        # First priority: Check for sentence end markers with improved handling
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
                
                # Handle question and exclamation marks inside quotations or parentheses
                if marker in ['!', '?']:
                    # Check if we're inside a quotation or parenthesis
                    text_before_marker = self.text_buffer[:marker_index]
                    if (text_before_marker.count('(') > text_before_marker.count(')') or
                        text_before_marker.count('"') % 2 == 1 or
                        text_before_marker.count("'") % 2 == 1):
                        
                        # Check if there's a closing bracket or quote right after
                        if marker_index + 1 < len(self.text_buffer) and self.text_buffer[marker_index+1] in [')', '"', "'"]:
                            # Let's continue to find the end of this parenthetical or quoted expression
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
        
        # Check for other natural pause markers like commas, colons, semicolons
        # but only if we already have enough words to meet minimum length
        pause_markers = [',', ':', ';']
        words = self.text_buffer.split()
        if len(words) >= self.min_words_to_generate:
            for marker in pause_markers:
                marker_index = self.text_buffer.find(marker)
                if marker_index != -1:
                    # Make sure we're not in the middle of a number (e.g. "10,000")
                    if marker == ',' and marker_index > 0 and marker_index < len(self.text_buffer) - 1:
                        if self.text_buffer[marker_index-1].isdigit() and self.text_buffer[marker_index+1].isdigit():
                            continue
                    
                    # Include the marker and any following space
                    end_index = marker_index + 1
                    if end_index < len(self.text_buffer) and self.text_buffer[end_index] == ' ':
                        end_index += 1
                    return self.text_buffer[:end_index]
        
        # Handle parenthetical expressions and quotes (complete them if possible)
        opening_chars = {'(': ')', '"': '"', "'": "'", '[': ']', '{': '}'}
        for open_char, close_char in opening_chars.items():
            open_index = self.text_buffer.find(open_char)
            if open_index != -1:
                # Look for corresponding closing character
                close_index = self.text_buffer.find(close_char, open_index + 1)
                if close_index != -1:
                    # If we have a complete expression and it's followed by a space, use it
                    if close_index + 1 < len(self.text_buffer) and self.text_buffer[close_index+1] == ' ':
                        return self.text_buffer[:close_index+2]  # Include the closing char and space
        
        # Second priority: If we have enough words, take a chunk but try to end at a natural point
        if len(words) >= self.max_words_per_chunk:
            # Determine how many words to include (up to max_words_per_chunk)
            chunk_size = min(len(words), self.max_words_per_chunk)
            
            # Try to find a good break point in the chunk (prepositions, conjunctions, etc.)
            good_break_words = ['and', 'or', 'but', 'yet', 'for', 'nor', 'so', 'at', 'by', 'in', 'of', 'on', 'to', 'with']
            
            # Start from max_words_per_chunk-1 and work backwards to find a good break point
            for i in range(chunk_size-1, max(chunk_size-3, 0), -1):  # Look back up to 3 words
                if words[i].lower() in good_break_words:
                    chunk_size = i + 1
                    break
            
            # Take exactly chunk_size words
            chunk_text = " ".join(words[:chunk_size])
            
            # Find the actual position in the original text to include any trailing space
            full_text_pos = len(chunk_text)
            if full_text_pos < len(self.text_buffer) and self.text_buffer[full_text_pos] == ' ':
                chunk_text += ' '
                full_text_pos += 1
            
            return self.text_buffer[:full_text_pos]
        
        # Third priority: If we have at least min_words and the buffer ends with a space
        if len(words) >= self.min_words_to_generate and self.text_buffer.endswith(" "):
            return self.text_buffer
        
        # Otherwise, find the last space and take everything before it
        # but only if we have at least min_words
        last_space_index = self.text_buffer.rfind(" ")
        if last_space_index != -1:  # Space found
            text = self.text_buffer[:last_space_index+1]
            words = text.split()
            if len(words) >= self.min_words_to_generate:
                return text
        
        # Not enough complete words yet
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
        
        # Detect lists (numbered or bulleted)
        list_indicators = [') ', '. ', '- ', '* ']
        if any(indicator in text for indicator in list_indicators):
            for i in range(1, 10):  # Check for numbered lists
                if f"{i}. " in text or f"{i}) " in text:
                    content_type['is_list'] = True
                    break
        
        # Detect numeric content
        if any(char.isdigit() for char in text):
            content_type['contains_numbers'] = True
            
        return content_type
    
    async def generate_from_buffer(self):
        """Generate audio from the current text buffer"""
        try:
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
            conditioning = model.prepare_conditioning(cond_dict)
            
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
            
            # H100-optimized chunk schedule - larger chunks for better throughput
            with torch.amp.autocast('cuda', dtype=torch.float16):  # Use FP16 for H100
                stream_generator = model.stream(
                    prefix_conditioning=conditioning,
                    audio_prefix_codes=None,
                    chunk_schedule=[32, 64, 64, 64, 64, 64, 64, 64],  # H100-optimized larger chunks
                    chunk_overlap=4,  # Increased overlap for smoother transitions
                )
                
                for audio_chunk in stream_generator:
                    if audio_chunk is None:
                        continue
                        
                    # Convert to numpy and then to WAV
                    audio_np = audio_chunk.cpu().numpy().squeeze()
                    audio_bytes = numpy_to_wav(audio_np, model.autoencoder.sampling_rate)
                    
                    chunk_counter += 1
                    current_time = time.time() - start_time
                    generated_time = audio_np.shape[0] / model.autoencoder.sampling_rate
                    
                    print(f"Sending chunk {chunk_counter}: time {current_time*1000:.0f}ms | generated {generated_time*1000:.0f}ms of audio")
                    
                    # Send audio chunk
                    await self.websocket.send(audio_bytes)
                    
                    # Small delay to allow other tasks to run
                    await asyncio.sleep(0.001)
            
            # Send end of stream marker
            await self.websocket.send(json.dumps({"type": "end"}))
            print("Streaming completed")
            
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
            if self.has_enough_complete_words():
                self.generation_task = asyncio.create_task(self.generate_from_buffer())

async def handle_websocket(websocket):
    """Handle incoming websocket connections"""
    try:
        # Create a streaming session for this connection
        session = None
        
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "start_session":
                # Initialize a new streaming session
                language = data.get("language", "en-us")
                session = StreamingTTSSession(websocket, language)
                await websocket.send(json.dumps({"type": "session_started"}))
                
            elif data["type"] == "add_text" and session:
                # Add text to the current session
                text = data.get("text", "")
                if text:
                    await session.add_text(text)
                    
            elif data["type"] == "end_session" and session:
                # End the current session
                if session.generation_task and not session.generation_task.done():
                    session.generation_task.cancel()
                session = None
                await websocket.send(json.dumps({"type": "session_ended"}))
                
            elif data["type"] == "generate":
                # One-off generation for compatibility
                text = data.get("text", "")
                language = data.get("language", "en-us")
                if text:
                    # Create a temporary session and generate the complete text
                    temp_session = StreamingTTSSession(websocket, language)
                    await temp_session.add_text(text)
                    while temp_session.text_buffer:
                        await temp_session.generate_from_buffer()
                
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
    
    print(f"Starting H100-optimized websocket server on {host}:{port}")
    async with websockets.serve(handle_websocket, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Print CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    asyncio.run(main())