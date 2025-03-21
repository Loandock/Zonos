import asyncio
import websockets
import json
import wave
import os

async def receive_audio():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to TTS server")
        
        # Initialize output directory
        os.makedirs("received_audio", exist_ok=True)
        chunk_counter = 0
        
        # Receive streaming audio
        while True:
            message = await websocket.recv()
            
            if isinstance(message, str):
                # Handle text messages (metadata, etc.)
                try:
                    data = json.loads(message)
                    print(f"Received metadata: {data}")
                    if data.get("type") == "end":
                        print("End of stream received")
                        break
                except:
                    print(f"Received text: {message}")
            else:
                # Handle binary audio data
                chunk_counter += 1
                filename = f"received_audio/chunk_{chunk_counter}.wav"
                with open(filename, "wb") as f:
                    f.write(message)
                print(f"Saved audio chunk {chunk_counter} to {filename}")
        
        print("Audio reception complete")

asyncio.run(receive_audio()) 