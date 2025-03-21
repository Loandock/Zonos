import asyncio
import websockets
import json
import os
import time
from datetime import datetime

async def receive_audio():
    uri = "ws://localhost:8765"
    
    # Timing variables
    session_start_time = time.time()
    chunk_times = []
    last_chunk_time = None
    
    # Console log with timestamp
    def log(message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        elapsed = time.time() - session_start_time
        print(f"[{timestamp}] [{elapsed:.3f}s] {message}")
    
    log("Connecting to TTS server...")
    
    async with websockets.connect(uri) as websocket:
        log("Connected to TTS server")
        
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
                    log(f"Received metadata: {data}")
                    if data.get("type") == "end":
                        # Calculate statistics
                        total_time = time.time() - session_start_time
                        
                        if chunk_counter > 0:
                            avg_time_between_chunks = sum([chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]) / (len(chunk_times) - 1) if len(chunk_times) > 1 else 0
                            time_to_first_chunk = chunk_times[0] - session_start_time
                            
                            log("=========== GENERATION COMPLETE ===========")
                            log(f"Total chunks: {chunk_counter}")
                            log(f"Total time: {total_time:.3f}s")
                            log(f"Time to first chunk: {time_to_first_chunk:.3f}s")
                            log(f"Avg time between chunks: {avg_time_between_chunks:.3f}s")
                            log("===========================================")
                        
                        log("End of stream received")
                        break
                except:
                    log(f"Received text: {message}")
            else:
                # Handle binary audio data
                current_time = time.time()
                chunk_times.append(current_time)
                
                if last_chunk_time:
                    time_since_last = current_time - last_chunk_time
                else:
                    time_since_last = current_time - session_start_time
                
                chunk_counter += 1
                last_chunk_time = current_time
                
                filename = f"received_audio/chunk_{chunk_counter}.wav"
                with open(filename, "wb") as f:
                    f.write(message)
                
                log(f"Chunk #{chunk_counter:3d} | {len(message)/1024:.1f} KB | +{time_since_last:.3f}s")
        
        log("Audio reception complete")

if __name__ == "__main__":
    asyncio.run(receive_audio())