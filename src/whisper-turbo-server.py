#!/usr/bin/env python3
"""
GPU-Accelerated Whisper Large v3 Turbo WebSocket Server using faster-whisper
Runs on NVIDIA GPU for high-speed transcription with state-of-the-art accuracy
Model: openai/whisper-large-v3-turbo
"""

import asyncio
import json
import numpy as np
import websockets
from faster_whisper import WhisperModel
import io
import struct
from datetime import datetime
import spacy
import tempfile
import soundfile as sf

# Initialize model
print("Loading Whisper Large v3 Turbo model (faster-whisper)...")
print("This is a large model and may take a moment to download and load...")

# Force CPU mode to avoid GPU memory contention with other models
# On a 4GB GPU, running 3 models simultaneously causes severe performance issues
print("Using CPU mode to prevent GPU memory contention...")
print("(Allows Whisper Small and Parakeet to use GPU without competition)")

model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8",  # INT8 quantization for good CPU performance
    num_workers=4  # Multi-threaded CPU inference
)
device_info = "CPU (INT8, 4 threads)"
print(f"✅ Whisper Large v3 Turbo loaded successfully on CPU")

print(f"   Device: {device_info}")
print(f"   Model: large-v3-turbo (state-of-the-art multilingual)")

# Load spaCy German model for better sentence detection
print("Loading spaCy German model for sentence detection...")
nlp = spacy.load("de_core_news_sm")
print("✅ spaCy German model loaded successfully")

# Store audio buffers and processing positions per client
audio_buffers = {}
processed_positions = {}  # Track what audio has been transcribed for streaming

async def handle_client(websocket):
    client_id = id(websocket)
    audio_buffers[client_id] = []
    processed_positions[client_id] = 0

    print(f"Client {client_id} connected")

    # Send ready message
    await websocket.send(json.dumps({
        "type": "ready",
        "message": f"Whisper Large v3 Turbo ready (faster-whisper on {device_info})",
        "clientId": str(client_id),
        "model": "large-v3-turbo",
        "device": device_info
    }))

    try:
        async for message in websocket:
            # Handle binary audio data
            if isinstance(message, bytes):
                audio_buffers[client_id].append(message)
                buffer_size = sum(len(b) for b in audio_buffers[client_id])
                duration = buffer_size / (16000 * 2)  # 16kHz, 16-bit

                print(f"Client {client_id}: received {len(message)} bytes (total: {duration:.1f}s)")

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
                    # Configuration handled (language, etc.)
                    await websocket.send(json.dumps({
                        "type": "configured",
                        "config": data
                    }))

                elif data.get("type") == "transcribe":
                    # Full transcription
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
                        "message": "Transcribing with Whisper Large v3 Turbo...",
                        "audioSize": len(audio_data)
                    }))

                    try:
                        start_time = datetime.now()

                        # Convert bytes to numpy array (16-bit PCM to float32)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe with faster-whisper
                        segments, info = model.transcribe(
                            audio_np,
                            language="de",  # German
                            beam_size=5,
                            vad_filter=True,  # Voice activity detection
                            word_timestamps=True
                        )

                        # Collect all segments
                        full_text = ""
                        chunks = []
                        for segment in segments:
                            full_text += segment.text
                            chunks.append({
                                "timestamp": [segment.start, segment.end],
                                "text": segment.text
                            })

                        processing_time = (datetime.now() - start_time).total_seconds()
                        audio_duration = len(audio_data) / (16000 * 2)
                        rtfx = audio_duration / processing_time if processing_time > 0 else 0

                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": full_text,
                            "chunks": chunks,
                            "model": "large-v3-turbo",
                            "device": device_info,
                            "language": info.language,
                            "language_probability": info.language_probability,
                            "performance": {
                                "processingTime": f"{processing_time:.2f}s",
                                "audioDuration": f"{audio_duration:.2f}s",
                                "rtfx": f"{rtfx:.2f}x"
                            }
                        }))

                        # Clear buffer
                        audio_buffers[client_id] = []

                    except Exception as e:
                        print(f"Transcription error: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))

                elif data.get("type") == "transcribe_stream":
                    # Streaming transcription with delta approach
                    if not audio_buffers[client_id]:
                        continue

                    # Get all accumulated audio
                    all_audio_data = b''.join(audio_buffers[client_id])
                    total_bytes = len(all_audio_data)

                    # Get position of last processed audio
                    last_processed = processed_positions[client_id]

                    # Calculate NEW audio that hasn't been transcribed
                    new_audio_bytes = total_bytes - last_processed

                    # Only process if we have at least 3 seconds of NEW audio
                    min_new_bytes = 16000 * 2 * 3  # 3 seconds minimum
                    if new_audio_bytes < min_new_bytes:
                        continue

                    try:
                        # Extract NEW audio with 2s overlap for context
                        overlap_bytes = 16000 * 2 * 2  # 2 seconds overlap
                        start_pos = max(0, last_processed - overlap_bytes)
                        new_audio_data = all_audio_data[start_pos:]

                        # Convert to numpy
                        audio_np = np.frombuffer(new_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_duration = len(new_audio_data) / (16000 * 2)

                        overlap_duration = (last_processed - start_pos) / (16000 * 2) if last_processed > start_pos else 0
                        new_duration = audio_duration - overlap_duration

                        print(f"[Whisper Turbo Stream] Processing {new_duration:.1f}s NEW audio (with {overlap_duration:.1f}s overlap)")

                        # Quick transcription for streaming
                        start_time = datetime.now()
                        segments, info = model.transcribe(
                            audio_np,
                            language="de",
                            beam_size=5,
                            vad_filter=True,
                            word_timestamps=False
                        )

                        # Get text
                        text = ""
                        for segment in segments:
                            text += segment.text

                        processing_time = (datetime.now() - start_time).total_seconds()
                        print(f"[Whisper Turbo Stream] Processed in {processing_time:.2f}s")

                        # Use spaCy for better sentence detection
                        sentences = []
                        if text.strip():
                            doc = nlp(text.strip())
                            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                            print(f"[spaCy] Detected {len(sentences)} sentence(s)")
                            print(f"[Whisper Turbo Stream] Text: {text[:100]}...")

                        # Only send if we got text
                        if text.strip():
                            await websocket.send(json.dumps({
                                "type": "partial_transcription",
                                "text": text,
                                "sentences": sentences,
                                "isPartial": True
                            }))

                        # Update processed position to current total
                        processed_positions[client_id] = total_bytes

                        # Keep only last 60 seconds in memory to prevent unbounded growth
                        max_buffer_bytes = 60 * 16000 * 2  # 60 seconds
                        if total_bytes > max_buffer_bytes:
                            # Keep last 60s and adjust processed position
                            all_audio = b''.join(audio_buffers[client_id])
                            audio_buffers[client_id] = [all_audio[-max_buffer_bytes:]]

                            # Adjust processed position
                            bytes_removed = total_bytes - max_buffer_bytes
                            processed_positions[client_id] = max(0, processed_positions[client_id] - bytes_removed)
                            print(f"[Whisper Turbo Stream] Trimmed buffer to last 60s")

                    except Exception as e:
                        print(f"Stream transcription error: {e}")
                        import traceback
                        traceback.print_exc()

                elif data.get("type") == "clear":
                    audio_buffers[client_id] = []
                    processed_positions[client_id] = 0
                    await websocket.send(json.dumps({
                        "type": "cleared",
                        "message": "Audio buffer cleared"
                    }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected")
    finally:
        if client_id in audio_buffers:
            del audio_buffers[client_id]
        if client_id in processed_positions:
            del processed_positions[client_id]

async def main():
    print("Starting Whisper Large v3 Turbo WebSocket server on ws://localhost:5003/transcribe")
    print(f"Device: {device_info}")
    print("Model: large-v3-turbo (openai/whisper-large-v3-turbo)")

    async with websockets.serve(handle_client, "0.0.0.0", 5003):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
