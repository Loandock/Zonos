# test_tts_standalone.py
import torch
import torchaudio
import time

from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

def test_tts():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    model.requires_grad_(False).eval()
    
    # Load a reference speaker
    print("Loading reference audio...")
    try:
        wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    except Exception as e:
        print(f"Error loading reference audio: {e}")
        # Create a simple sine wave as a fallback
        sr = 44100
        wav = torch.sin(2 * torch.pi * 440 * torch.arange(sr * 3) / sr).unsqueeze(0)
        print("Created a synthetic audio fallback")
    
    # Create speaker embedding
    speaker = model.make_speaker_embedding(wav, sr)
    
    # Test text
    text = "This is a test of the Zonos text-to-speech system."
    print(f"Generating speech for: {text}")
    
    # Create conditioning dictionary
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    
    # Prepare conditioning
    with torch.no_grad():
        conditioning = model.prepare_conditioning(cond_dict)
    
    # Stream generation
    print("Starting streaming generation...")
    start_time = time.time()
    
    # Use a safe chunk schedule
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,
        chunk_schedule=[3, 5, 8, 12, 15, 20, 25, 30],
        chunk_overlap=1,
    )
    
    # Collect audio chunks
    audio_chunks = []
    for audio_chunk in stream_generator:
        audio_chunks.append(audio_chunk.cpu())
        print(f"Generated chunk of shape {audio_chunk.shape}")
    
    # Concatenate all chunks
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=1)
        print(f"Full audio shape: {full_audio.shape}")
        
        # Save the audio
        output_file = "tts_test_output.wav"
        torchaudio.save(output_file, full_audio, 44100)
        print(f"Saved audio to {output_file}")
    else:
        print("No audio chunks were generated")
    
    elapsed_time = time.time() - start_time
    print(f"Speech generation completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    test_tts()
