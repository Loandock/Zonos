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
# No 25s limitation - use the full audio file
print(f"Using full audio file: {custom_wav.shape[1]/custom_sr:.1f}s")

with torch.amp.autocast('cuda', dtype=torch.float16):
    custom_speaker_embedding = model.make_speaker_embedding(custom_wav, custom_sr)
print("Custom voice loaded successfully!")

# === CRITICAL: PRE-COMPILE CUDA KERNELS ===
print("Pre-compiling CUDA kernels (this will take ~30-60 seconds)...")
warmup_start = time.time()

# Warmup text for regular generation
dummy_text = "This is a warmup pass to optimize CUDA performance."
dummy_cond = make_cond_dict(
    text=dummy_text,
    speaker=custom_speaker_embedding,
    language="en-us"
)

# Run a complete generation to compile all kernels
with torch.amp.autocast('cuda', dtype=torch.float16):
    with torch.no_grad():
        conditioning = model.prepare_conditioning(dummy_cond)
        
        # Generate a complete sentence to warm up all kernels
        print("Warming up regular generation...")
        audio_codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            max_new_tokens=500,
            cfg_scale=2.0,
            sampling_params=dict(min_p=0.1),
            progress_bar=False
        )
        
        # Also warm up streaming generation
        print("Warming up streaming generation...")
        stream_generator = model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            chunk_schedule=[17, *range(9, 20)],
            chunk_overlap=1,
        )
        
        # Get first chunk to compile streaming kernels
        first_chunk = next(stream_generator)
        
        # Get second chunk to ensure all streaming is warmed up
        try:
            second_chunk = next(stream_generator)
        except StopIteration:
            pass

print(f"CUDA kernel pre-compilation completed in {time.time() - warmup_start:.1f}s")
print("H100 is now ready for high-performance inference!")

def numpy_to_mp3(audio_np, sample_rate):
    """Convert numpy array to MP3 bytes"""
    import io
    import soundfile as sf
    
    # Create a BytesIO object
    mp3_io = io.BytesIO()
    
    # Write the audio data to the BytesIO object as WAV
    with io.BytesIO() as wav_io:
        sf.write(wav_io, audio_np, sample_rate, format='WAV')
        wav_io.seek(0)
        
        # Convert WAV to MP3 using pydub
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(wav_io)
        audio_segment.export(mp3_io, format="mp3", bitrate="128k")
    
    # Get the MP3 bytes
    mp3_io.seek(0)
    return mp3_io.read()

async def generate_audio_stream(websocket, text, language):
    """Generate and stream audio for the given text"""
    # Create conditioning
    print(f"Creating conditioning for text: '{text}'")
    cond_dict = make_cond_dict(
        text=text,
        speaker=custom_speaker_embedding,
        language=language
    )
    
    # Send initial metadata
    await websocket.send(json.dumps({
        "type": "metadata",
        "text": text
    }))
    
    # Generate and stream audio
    print("Starting streaming generation...")
    start_time = time.time()
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            conditioning = model.prepare_conditioning(cond_dict)
            
            # Use streaming generation
            chunk_counter = 0
            for chunk in model.stream(
                prefix_conditioning=conditioning,
                audio_prefix_codes=None,
                chunk_schedule=[17, *range(9, 100)],
                chunk_overlap=1,
            ):
                # Convert to numpy and then to MP3
                chunk = chunk.cpu().numpy().squeeze()
                audio_bytes = numpy_to_mp3(chunk, model.autoencoder.sampling_rate)
                
                chunk_counter += 1
                current_time = time.time() - start_time
                generated_time = chunk.shape[0] / model.autoencoder.sampling_rate
                
                print(f"Sending chunk {chunk_counter}: time {current_time*1000:.0f}ms | generated {generated_time*1000:.0f}ms of audio")
                
                # Send audio chunk
                await websocket.send(audio_bytes)
    
    # Send end of stream marker
    await websocket.send(json.dumps({"type": "end"}))
    print(f"Streaming completed in {(time.time() - start_time)*1000:.0f}ms")

async def handle_websocket(websocket):
    """Handle incoming websocket connections"""
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "generate":
                text = data.get("text", "Hello, this is a test of the streaming TTS system.")
                language = data.get("language", "en-us")
                
                # We ignore any speaker_path from the client and always use our custom voice
                await generate_audio_stream(websocket, text, language)
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
