#!/usr/bin/env python3
"""
Voxtral WebSocket Server using Mistral Cloud API
Port: 5000
Uses Mistral's cloud-based Voxtral transcription service
"""

import asyncio
import json
import numpy as np
import websockets
from datetime import datetime
import sys
import os
import tempfile
import soundfile as sf

print("Initializing Voxtral API proxy server...")

# Check for Mistral API key
api_key = os.environ.get("MISTRAL_API_KEY", None)
if not api_key:
    print("⚠️  Warning: MISTRAL_API_KEY not set")
    print("   Set it with: export MISTRAL_API_KEY=your_api_key")
    print("   Get your key from: https://console.mistral.ai/")
    print("   For now, server will run but transcription will fail")
else:
    print(f"✅ Using Mistral API key: {api_key[:8]}...")

try:
    from mistralai import Mistral
    print("✅ Mistral AI library loaded")
    device_info = "Mistral Cloud API"
except ImportError:
    print("❌ Error: mistralai library not installed")
    print("   Install with: pip install mistralai")
    sys.exit(1)

# Store audio buffers per client
audio_buffers = {}

async def transcribe_with_mistral(audio_path, language="de"):
    """Transcribe audio using Mistral's Voxtral API"""
    try:
        client = Mistral(api_key=api_key)

        print(f"[Mistral API] Uploading audio file...")
        with open(audio_path, "rb") as f:
            transcription_response = client.audio.transcriptions.complete(
                model="voxtral-mini-latest",
                file={
                    "content": f,
                    "file_name": os.path.basename(audio_path),
                },
                language=language if language != "german" else "de"
            )

        return transcription_response.text
    except Exception as e:
        print(f"[Mistral API] Error: {e}")
        raise e

async def handle_client(websocket):
    client_id = id(websocket)
    audio_buffers[client_id] = []

    print(f"Client {client_id} connected")

    # Send ready message
    await websocket.send(json.dumps({
        "type": "ready",
        "message": f"Voxtral ready (Mistral Cloud API)",
        "clientId": str(client_id),
        "model": "voxtral-mini-latest",
        "device": device_info
    }))

    try:
        async for message in websocket:
            # Handle binary audio data
            if isinstance(message, bytes):
                audio_buffers[client_id].append(message)
                buffer_size = sum(len(b) for b in audio_buffers[client_id])
                duration = buffer_size / (16000 * 2)  # 16kHz, 16-bit

                await websocket.send(json.dumps({
                    "type": "chunk_received",
                    "chunkSize": len(message),
                    "totalBuffered": buffer_size,
                    "durationSeconds": f"{duration:.2f}"
                }))

            # Handle control messages
            elif isinstance(message, str):
                data = json.loads(message)

                if data.get("type") == "configure":
                    await websocket.send(json.dumps({
                        "type": "configured",
                        "config": data
                    }))

                elif data.get("type") == "transcribe" or data.get("type") == "transcribe_stream":
                    if not audio_buffers[client_id]:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No audio data received"
                        }))
                        continue

                    # Concatenate all audio chunks
                    audio_data = b''.join(audio_buffers[client_id])

                    await websocket.send(json.dumps({
                        "type": "processing",
                        "message": "Transcribing with Voxtral Cloud API...",
                        "audioSize": len(audio_data)
                    }))

                    try:
                        start_time = datetime.now()

                        # Convert bytes to numpy array (16-bit PCM to float32)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_duration = len(audio_data) / (16000 * 2)

                        print(f"[Voxtral] Transcribing {audio_duration:.1f}s of audio")

                        # Limit buffer to last 30 seconds for streaming to prevent hang
                        if data.get("type") == "transcribe_stream":
                            max_samples = 30 * 16000  # 30 seconds
                            if len(audio_np) > max_samples:
                                print(f"[Voxtral] Buffer too large ({audio_duration:.1f}s), keeping last 30s")
                                audio_np = audio_np[-max_samples:]
                                audio_duration = 30.0

                        # Save to temporary WAV file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp_path = tmp.name
                            sf.write(tmp_path, audio_np, 16000)

                        # Transcribe with Mistral API
                        text = await transcribe_with_mistral(tmp_path, language="de")

                        # Clean up temp file
                        os.unlink(tmp_path)

                        processing_time = (datetime.now() - start_time).total_seconds()
                        rtfx = audio_duration / processing_time if processing_time > 0 else 0

                        print(f"[Voxtral] Result: {text[:100]}...")
                        print(f"[Voxtral] Speed: {rtfx:.2f}x real-time")

                        # Send appropriate response type
                        response_type = "partial_transcription" if data.get("type") == "transcribe_stream" else "transcription"

                        response = {
                            "type": response_type,
                            "text": text,
                            "chunks": [],
                            "model": "voxtral-mini-latest",
                            "device": device_info,
                            "isPartial": data.get("type") == "transcribe_stream"
                        }

                        # Add performance metrics for full transcription
                        if response_type == "transcription":
                            response["performance"] = {
                                "processingTime": f"{processing_time:.2f}s",
                                "audioDuration": f"{audio_duration:.2f}s",
                                "rtfx": f"{rtfx:.2f}x"
                            }

                        await websocket.send(json.dumps(response))

                        # Clear or trim buffer based on mode
                        if data.get("type") == "transcribe":
                            audio_buffers[client_id] = []
                        else:
                            # For streaming, keep buffer but limit size
                            max_buffer_bytes = 30 * 16000 * 2  # 30 seconds of 16-bit PCM
                            current_buffer_size = sum(len(b) for b in audio_buffers[client_id])
                            if current_buffer_size > max_buffer_bytes:
                                # Recombine and trim
                                all_audio = b''.join(audio_buffers[client_id])
                                audio_buffers[client_id] = [all_audio[-max_buffer_bytes:]]
                                print(f"[Voxtral] Trimmed buffer to last 30s")

                    except Exception as e:
                        print(f"Transcription error: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))

                elif data.get("type") == "clear":
                    audio_buffers[client_id] = []
                    await websocket.send(json.dumps({
                        "type": "cleared",
                        "message": "Audio buffer cleared"
                    }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected")
    finally:
        if client_id in audio_buffers:
            del audio_buffers[client_id]

async def main():
    print("\n" + "="*60)
    print("🎙️  Voxtral WebSocket Server (Mistral Cloud API)")
    print("="*60)
    print(f"URL: ws://localhost:5000/transcribe")
    print(f"Model: voxtral-mini-latest")
    print(f"Device: {device_info}")
    print(f"Cost: ~$0.001 per minute of audio")
    print("="*60 + "\n")

    async with websockets.serve(handle_client, "0.0.0.0", 5000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
