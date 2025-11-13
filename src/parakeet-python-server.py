#!/usr/bin/env python3
"""
Parakeet TDT WebSocket Server using NVIDIA NeMo
Port: 3001
"""

import asyncio
import json
import numpy as np
import websockets
from datetime import datetime
import sys
import os

print("Loading Parakeet TDT model...")
print("This may take a few minutes on first run...")

try:
    import nemo.collections.asr as nemo_asr
    import torch

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Parakeet TDT model from NVIDIA (v3 supports multilingual including German)
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3"
    )

    # Move to device and optimize
    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    # Disable gradients for faster inference
    torch.set_grad_enabled(False)
    model.eval()

    print(f"✅ Parakeet TDT model loaded successfully on {device}", flush=True)
    print("🚀 Chunked processing enabled for files > 60 seconds", flush=True)
    device_info = f"{device.upper()}"

except ImportError as e:
    print(f"❌ Error: NVIDIA NeMo not installed")
    print("\nTo install NeMo:")
    print("  pip install nemo_toolkit[asr]")
    print("Or:")
    print("  pip install 'nemo_toolkit[all]'")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading Parakeet model: {e}")
    print("\nMake sure you have installed:")
    print("  pip install nemo_toolkit[asr]")
    sys.exit(1)

# Store audio buffers per client
audio_buffers = {}

def chunk_audio_with_overlap(audio_array, sample_rate=16000, chunk_seconds=30, overlap_seconds=3):
    """
    Split audio into overlapping chunks for efficient processing.

    Args:
        audio_array: Numpy array of audio samples
        sample_rate: Sample rate in Hz (default 16000)
        chunk_seconds: Length of each chunk in seconds (default 30)
        overlap_seconds: Overlap between chunks in seconds (default 3)

    Returns:
        List of audio chunks with metadata
    """
    chunk_samples = chunk_seconds * sample_rate
    overlap_samples = overlap_seconds * sample_rate
    stride = chunk_samples - overlap_samples

    chunks = []
    for start_idx in range(0, len(audio_array), stride):
        end_idx = min(start_idx + chunk_samples, len(audio_array))
        chunk = audio_array[start_idx:end_idx]

        # Store metadata for intelligent merging
        chunks.append({
            'audio': chunk,
            'start_time': start_idx / sample_rate,
            'end_time': end_idx / sample_rate,
            'is_first': start_idx == 0,
            'is_last': end_idx >= len(audio_array)
        })

        if end_idx >= len(audio_array):
            break

    return chunks

def merge_transcriptions(chunk_results, overlap_seconds=3):
    """
    Intelligently merge transcriptions from overlapping chunks.
    Removes duplicate text in overlapping regions.
    """
    if not chunk_results:
        return ""

    if len(chunk_results) == 1:
        return chunk_results[0]['text']

    merged_text = chunk_results[0]['text']

    for i in range(1, len(chunk_results)):
        current_text = chunk_results[i]['text']

        # Simple word-based merging: take last ~10 words from previous,
        # first ~10 words from current, find overlap
        prev_words = merged_text.split()
        curr_words = current_text.split()

        # Find overlap by checking last N words of previous against first M words of current
        overlap_found = False
        for overlap_len in range(min(15, len(prev_words), len(curr_words)), 0, -1):
            if prev_words[-overlap_len:] == curr_words[:overlap_len]:
                # Found overlap, merge by taking current from after overlap
                merged_text = merged_text + " " + " ".join(curr_words[overlap_len:])
                overlap_found = True
                break

        if not overlap_found:
            # No overlap found, just concatenate with space
            merged_text = merged_text + " " + current_text

    return merged_text.strip()

async def handle_client(websocket):
    client_id = id(websocket)
    audio_buffers[client_id] = []

    print(f"Client {client_id} connected")

    # Send ready message
    await websocket.send(json.dumps({
        "type": "ready",
        "message": f"Parakeet TDT ready (nvidia/parakeet-tdt-0.6b-v2 on {device_info})",
        "clientId": str(client_id),
        "model": "parakeet-tdt-0.6b-v2",
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

                elif data.get("type") == "transcribe":
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
                        "message": "Transcribing with Parakeet TDT...",
                        "audioSize": len(audio_data)
                    }))

                    try:
                        start_time = datetime.now()

                        # Convert bytes to numpy array (16-bit PCM to float32)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_duration = len(audio_data) / (16000 * 2)

                        # Save to temporary WAV file (NeMo expects file paths)
                        import tempfile
                        import soundfile as sf

                        # Use chunked processing for long audio files (> 60 seconds)
                        CHUNK_THRESHOLD = 60  # seconds

                        if audio_duration > CHUNK_THRESHOLD:
                            print(f"[Parakeet] Long audio detected ({audio_duration:.1f}s), using chunked processing")

                            # Split into chunks
                            chunks = chunk_audio_with_overlap(
                                audio_np,
                                sample_rate=16000,
                                chunk_seconds=30,
                                overlap_seconds=3
                            )

                            chunk_results = []

                            # Process each chunk
                            for i, chunk_data in enumerate(chunks):
                                chunk_audio = chunk_data['audio']
                                chunk_duration = len(chunk_audio) / 16000

                                # Save chunk to temp file
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                    tmp_path = tmp.name
                                    sf.write(tmp_path, chunk_audio, 16000)

                                # Transcribe chunk
                                chunk_start = datetime.now()
                                transcription = model.transcribe([tmp_path])
                                chunk_time = (datetime.now() - chunk_start).total_seconds()

                                chunk_text = transcription[0].text if transcription and len(transcription) > 0 else ""

                                chunk_results.append({
                                    'text': chunk_text,
                                    'start_time': chunk_data['start_time'],
                                    'end_time': chunk_data['end_time']
                                })

                                os.unlink(tmp_path)

                                print(f"[Parakeet] Chunk {i+1}/{len(chunks)}: {chunk_duration:.1f}s → {chunk_time:.2f}s ({chunk_duration/chunk_time:.1f}x RT) - {chunk_text[:50]}...")

                                # Send progress update
                                progress_pct = ((i + 1) / len(chunks)) * 100
                                await websocket.send(json.dumps({
                                    "type": "processing",
                                    "message": f"Processing chunk {i+1}/{len(chunks)} ({progress_pct:.0f}%)",
                                    "progress": progress_pct
                                }))

                            # Merge all transcriptions
                            text = merge_transcriptions(chunk_results, overlap_seconds=3)
                            print(f"[Parakeet] Merged {len(chunks)} chunks into final transcription")

                        else:
                            # Original code for short audio (< 60 seconds)
                            print(f"[Parakeet] Short audio ({audio_duration:.1f}s), using standard processing")

                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                tmp_path = tmp.name
                                sf.write(tmp_path, audio_np, 16000)

                            transcription = model.transcribe([tmp_path])
                            text = transcription[0].text if transcription and len(transcription) > 0 else ""
                            os.unlink(tmp_path)

                        processing_time = (datetime.now() - start_time).total_seconds()
                        rtfx = audio_duration / processing_time if processing_time > 0 else 0

                        print(f"[Parakeet] Total: {audio_duration:.1f}s → {processing_time:.2f}s ({rtfx:.1f}x RT)")

                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": text,
                            "chunks": [],  # Parakeet TDT doesn't return word timestamps by default
                            "model": "parakeet-tdt-0.6b-v3",
                            "device": device_info,
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
                    # Streaming transcription
                    if not audio_buffers[client_id]:
                        continue

                    audio_data = b''.join(audio_buffers[client_id])

                    # Only process if we have at least 1 second
                    if len(audio_data) < 16000 * 2:
                        continue

                    try:
                        # Convert to numpy
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_duration = len(audio_data) / (16000 * 2)

                        print(f"[Parakeet Stream] Transcribing {audio_duration:.1f}s of accumulated audio")

                        # Limit buffer to last 30 seconds to prevent hang
                        # Keep more context than other models for better accuracy
                        max_samples = 30 * 16000  # 30 seconds
                        if len(audio_np) > max_samples:
                            print(f"[Parakeet Stream] Buffer too large ({audio_duration:.1f}s), keeping last 30s")
                            audio_np = audio_np[-max_samples:]
                            audio_duration = 30.0

                        # Save to temp file
                        import tempfile
                        import soundfile as sf

                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp_path = tmp.name
                            sf.write(tmp_path, audio_np, 16000)

                        # Quick transcription with full context
                        start_time = datetime.now()
                        transcription = model.transcribe([tmp_path])
                        text = transcription[0].text if transcription and len(transcription) > 0 else ""
                        processing_time = (datetime.now() - start_time).total_seconds()

                        os.unlink(tmp_path)

                        print(f"[Parakeet Stream] Processed in {processing_time:.2f}s")

                        # Only send if we got text
                        if text.strip():
                            await websocket.send(json.dumps({
                                "type": "partial_transcription",
                                "text": text,
                                "isPartial": True
                            }))
                            print(f"[Parakeet Stream] Sent: {text}")

                        # Keep buffer but limit size to prevent unbounded growth
                        # Trim to last 30 seconds worth of audio
                        max_buffer_bytes = 30 * 16000 * 2  # 30 seconds of 16-bit PCM
                        current_buffer_size = sum(len(b) for b in audio_buffers[client_id])
                        if current_buffer_size > max_buffer_bytes:
                            # Recombine and trim
                            all_audio = b''.join(audio_buffers[client_id])
                            audio_buffers[client_id] = [all_audio[-max_buffer_bytes:]]
                            print(f"[Parakeet Stream] Trimmed buffer to last 30s")

                    except Exception as e:
                        print(f"Stream transcription error: {e}")
                        import traceback
                        traceback.print_exc()

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
    print("\n" + "="*60, flush=True)
    print("🚀 Parakeet TDT WebSocket Server", flush=True)
    print("="*60, flush=True)
    print(f"URL: ws://localhost:5002/transcribe", flush=True)
    print(f"Model: nvidia/parakeet-tdt-0.6b-v3 (Multilingual)", flush=True)
    print(f"Device: {device_info}", flush=True)
    print(f"Optimization: Chunked processing for long audio (>60s)", flush=True)
    print("="*60 + "\n", flush=True)

    async with websockets.serve(handle_client, "0.0.0.0", 5002):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
