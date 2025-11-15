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

# Store audio buffers and windowing state per client
audio_buffers = {}
window_positions = {}  # Track window positions for each client
full_transcripts = {}  # Store accumulated full transcript per client
last_window_starts = {}  # Track last window start position for each client

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

def merge_with_overlap(existing_text, new_text, max_overlap_words=50):
    """
    Merge new transcription with existing, removing overlapping words.
    Uses word-level overlap detection to handle 2-second audio overlap.
    """
    if not existing_text:
        return new_text
    if not new_text:
        return existing_text

    existing_words = existing_text.strip().split()
    new_words = new_text.strip().split()

    # Search for overlap from end of existing to start of new
    # Check up to 50 words for overlap (covers ~2-3 seconds of speech)
    best_overlap = 0
    search_limit = min(max_overlap_words, len(existing_words), len(new_words))

    for overlap_len in range(search_limit, 0, -1):
        existing_suffix = existing_words[-overlap_len:]
        new_prefix = new_words[:overlap_len]

        # Case-insensitive comparison
        if [w.lower() for w in existing_suffix] == [w.lower() for w in new_prefix]:
            best_overlap = overlap_len
            break

    if best_overlap > 0:
        # Merge: existing + (new without overlapping prefix)
        unique_new_words = new_words[best_overlap:]
        merged = existing_text + " " + " ".join(unique_new_words)
        print(f"[Merge] Found {best_overlap}-word overlap, appending {len(unique_new_words)} new words")
        return merged.strip()
    else:
        # No overlap found, concatenate
        print(f"[Merge] No overlap found, concatenating")
        return (existing_text + " " + new_text).strip()

def extract_last_sentence(text):
    """Extract the last sentence from text (simple sentence boundary detection)."""
    if not text.strip():
        return ""

    # Split by sentence-ending punctuation
    import re
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        # Add punctuation back to last sentence
        last_sent = sentences[-1]
        # Find the punctuation that ended this sentence
        if text.strip().endswith('.'):
            last_sent += '.'
        elif text.strip().endswith('!'):
            last_sent += '!'
        elif text.strip().endswith('?'):
            last_sent += '?'
        return last_sent
    return text.strip()

async def handle_client(websocket):
    client_id = id(websocket)
    audio_buffers[client_id] = []
    window_positions[client_id] = 0  # Track current window end position
    full_transcripts[client_id] = ""  # Accumulated full transcript
    last_window_starts[client_id] = None  # Track last window start position

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
                    # Streaming transcription with growing windows
                    if not audio_buffers[client_id]:
                        continue

                    # Get all accumulated audio
                    audio_data = b''.join(audio_buffers[client_id])
                    total_bytes = len(audio_data)
                    total_duration = total_bytes / (16000 * 2)

                    # Only process if we have at least 1 second
                    if total_duration < 1.0:
                        continue

                    # Windowing parameters
                    WINDOW_SIZE = 30.0  # Max window size
                    OVERLAP = 2.0  # Overlap when starting new window

                    # Determine current window start based on which window we're in
                    current_window_num = int(window_positions[client_id] / (WINDOW_SIZE - OVERLAP))
                    window_start_time = max(0, current_window_num * (WINDOW_SIZE - OVERLAP) - OVERLAP)

                    # Window grows from start to current total duration (up to 30s max per window)
                    window_end_time = total_duration
                    window_duration = window_end_time - window_start_time

                    # If window exceeds 30s, start a new window
                    if window_duration > WINDOW_SIZE:
                        # Start new window at (current_window_start + 28s)
                        window_start_time = window_start_time + (WINDOW_SIZE - OVERLAP)
                        window_end_time = total_duration
                        window_duration = window_end_time - window_start_time
                        print(f"[Parakeet Stream] Starting new window at {window_start_time:.1f}s")

                    # Skip if no new audio since last processing
                    if window_end_time <= window_positions[client_id]:
                        continue

                    try:
                        # Extract window from buffer
                        start_sample = int(window_start_time * 16000)
                        end_sample = int(window_end_time * 16000)

                        # Convert to numpy and extract window
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        window_audio = audio_np[start_sample:end_sample]
                        window_duration = len(window_audio) / 16000

                        print(f"[Parakeet Stream] Processing growing window {window_start_time:.1f}s-{window_end_time:.1f}s ({window_duration:.1f}s, total: {total_duration:.1f}s)")

                        # Save to temp file
                        import tempfile
                        import soundfile as sf

                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp_path = tmp.name
                            sf.write(tmp_path, window_audio, 16000)

                        # Transcribe window
                        start_time = datetime.now()
                        transcription = model.transcribe([tmp_path])
                        window_text = transcription[0].text if transcription and len(transcription) > 0 else ""
                        processing_time = (datetime.now() - start_time).total_seconds()

                        os.unlink(tmp_path)

                        print(f"[Parakeet Stream] Window transcribed in {processing_time:.2f}s: {window_text[:80]}...")

                        # Growing window (same start) vs new window (different start)
                        if window_start_time == last_window_starts[client_id]:
                            # Growing window - REPLACE transcript (re-transcribing same audio)
                            full_transcripts[client_id] = window_text
                            print(f"[Parakeet Stream] Growing window - replacing transcript")
                        else:
                            # New window - MERGE with overlap detection
                            if full_transcripts[client_id]:
                                full_transcripts[client_id] = merge_with_overlap(
                                    full_transcripts[client_id],
                                    window_text,
                                    max_overlap_words=100  # Larger window for 2s overlap
                                )
                                print(f"[Parakeet Stream] New window - merging with overlap detection")
                            else:
                                # First window ever
                                full_transcripts[client_id] = window_text
                                print(f"[Parakeet Stream] First window - initializing transcript")
                            # Update last window start
                            last_window_starts[client_id] = window_start_time

                        # Extract last sentence for chunked display
                        last_sentence = extract_last_sentence(full_transcripts[client_id])

                        print(f"[Parakeet Stream] Full transcript length: {len(full_transcripts[client_id])} chars")
                        print(f"[Parakeet Stream] Last sentence: {last_sentence[:60]}...")

                        # Update window position to end of processed audio
                        window_positions[client_id] = total_duration

                        # Send response with both full transcript and last sentence
                        if window_text.strip():
                            await websocket.send(json.dumps({
                                "type": "partial_transcription",
                                "text": window_text,  # Current window text (for compatibility)
                                "fullTranscript": full_transcripts[client_id],  # Merged full transcript
                                "lastSentence": last_sentence,  # Last sentence only
                                "isPartial": True,
                                "windowInfo": {
                                    "start": window_start_time,
                                    "end": window_end_time,
                                    "duration": window_duration
                                }
                            }))

                        # Clean old audio buffer to save memory
                        # Keep last 90 seconds for potential re-windowing
                        max_buffer_bytes = 90 * 16000 * 2
                        if total_bytes > max_buffer_bytes:
                            keep_from_byte = total_bytes - max_buffer_bytes
                            audio_buffers[client_id] = [audio_data[keep_from_byte:]]
                            # Adjust window position
                            window_positions[client_id] = min(window_positions[client_id], 90.0)
                            print(f"[Parakeet Stream] Trimmed buffer to last 90s")

                    except Exception as e:
                        print(f"Stream transcription error: {e}")
                        import traceback
                        traceback.print_exc()

                elif data.get("type") == "clear":
                    audio_buffers[client_id] = []
                    window_positions[client_id] = 0
                    full_transcripts[client_id] = ""
                    last_window_starts[client_id] = None
                    await websocket.send(json.dumps({
                        "type": "cleared",
                        "message": "Audio buffer cleared"
                    }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected")
    finally:
        if client_id in audio_buffers:
            del audio_buffers[client_id]
        if client_id in window_positions:
            del window_positions[client_id]
        if client_id in full_transcripts:
            del full_transcripts[client_id]
        if client_id in last_window_starts:
            del last_window_starts[client_id]

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
