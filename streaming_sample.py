import time

import torch
import torchaudio

from zonos.conditioning import make_cond_dict
from zonos.model import Zonos


def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    model.requires_grad_(False).eval()

    # Load a reference speaker audio to generate a speaker embedding.
    print("Loading reference audio...")
    wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(421)

    # Define the text prompt.
    text = "Hello, world! This is a test of streaming generation from Zonos with explicit chunk scheduling for faster start and fewer cuts later."

    # Create the conditioning dictionary (using text, speaker embedding, language, etc.).
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)

    # --- STREAMING GENERATION ---
    print("Starting streaming generation...")

    # Define chunk schedule: start with small chunks for faster initial output,
    # then gradually increase to larger chunks for fewer cuts
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,  # no audio prefix in this test
        chunk_schedule=[17, *range(9, 100)],  # optimal schedule for RTX4090
        chunk_overlap=1,  # tokens to overlap between chunks (affects crossfade)
    )

    # Accumulate audio chunks as they are generated.
    audio_chunks = []
    t0 = time.time()

    for i, audio_chunk in enumerate(stream_generator):
        elapsed = int((time.time() - t0) * 1000)
        print(
            f"Received chunk {i + 1}: time {elapsed}ms | generated up to {audio_chunk.shape[1] / 44.1 + elapsed:.0f}ms"
        )
        audio_chunks.append(audio_chunk.cpu())

    if len(audio_chunks) == 0:
        print("No audio chunks were generated.")
        return

    # Concatenate all audio chunks along the time axis.
    full_audio = torch.cat(audio_chunks, dim=-1)
    out_sr = model.autoencoder.sampling_rate

    # Save the full audio as a WAV file.
    torchaudio.save("stream_sample.wav", full_audio, out_sr)
    print(f"Saved streaming audio to 'stream_sample.wav' (sampling rate: {out_sr} Hz).")


if __name__ == "__main__":
    main()