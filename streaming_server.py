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
CUSTOM_VOICE_PATH = "/Users/arnavjha/Downloads/sesame_fine_tune.wav"

# Load the model once at startup
print("Loading model...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
model.requires_grad_(False).eval()

# Pre-load the speaker embedding to save time during generation
print(f"Loading custom voice from {CUSTOM_VOICE_PATH}...")
custom_wav, custom_sr = torchaudio.load(CUSTOM_VOICE_PATH)
custom_speaker_embedding = model.make_speaker_embedding(custom_wav, custom_sr)
print("Custom voice loaded successfully!")

def numpy_to_mp3(audio_array, sampling_rate=44100):
    """Convert numpy array to MP3 bytes"""
    # Convert to float32 if not already
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
    
    # Create a BytesIO buffer
    buffer = BytesIO()
    
    # Save as MP3
    torchaudio.save(buffer, audio_tensor, sampling_rate, format="mp3")
    
    # Get the bytes
    buffer.seek(0)
    return buffer.read()

async def generate_audio_stream(websocket, text, language="en-us"):
    """Generate audio in chunks and stream to the websocket"""
    
    # Always use our pre-loaded custom voice
    speaker_embedding = custom_speaker_embedding
    
    # Set a random seed for reproducibility
    torch.manual_seed(421)
    
    # Create conditioning dictionary with optimized parameters for voice cloning
    print(f"Creating conditioning for text: '{text}'")
    cond_dict = make_cond_dict(
        text=text, 
        speaker=speaker_embedding, 
        language=language,
        fmax=22050,  # Optimal for voice cloning
        speaking_rate=13  # Adjust based on your sample's natural pace
    )
    conditioning = model.prepare_conditioning(cond_dict)
    
    # Stream generation parameters
    print("Starting streaming generation...")
    start_time = time.time()
    
    # Send initial metadata
    await websocket.send(json.dumps({
        "type": "metadata",
        "sampling_rate": model.autoencoder.sampling_rate,
        "text": text
    }))
    
    # Generate and stream audio chunks
    chunk_counter = 0
    
    # Use optimal chunking for production environment
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,  # no audio prefix in this test
        chunk_schedule=[17, *range(9, 100)],  # optimal schedule for RTX4090/H100
        chunk_overlap=1,  # tokens to overlap between chunks (affects crossfade)
    )
    
    for audio_chunk in stream_generator:
        if audio_chunk is None:
            continue
            
        # Convert to numpy and then to MP3
        audio_np = audio_chunk.cpu().numpy().squeeze()
        audio_bytes = numpy_to_mp3(audio_np, model.autoencoder.sampling_rate)
        
        chunk_counter += 1
        current_time = time.time() - start_time
        generated_time = audio_np.shape[0] / model.autoencoder.sampling_rate
        
        print(f"Sending chunk {chunk_counter}: time {current_time*1000:.0f}ms | generated {generated_time*1000:.0f}ms of audio")
        
        # Send audio chunk
        await websocket.send(audio_bytes)
        
        # Small delay to allow other tasks to run
        await asyncio.sleep(0.001)
    
    # Send end of stream marker
    await websocket.send(json.dumps({"type": "end"}))
    print("Streaming completed")

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
    
    print(f"Starting websocket server on {host}:{port}")
    async with websockets.serve(handle_websocket, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 