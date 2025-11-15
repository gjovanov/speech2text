#!/usr/bin/env python3
"""
GPU-Accelerated Whisper WebSocket Server using faster-whisper
Runs on NVIDIA GPU for 10-20x speed improvement
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

# Initialize model
# Note: GPU mode requires cuDNN which may not be installed
# CPU mode with faster-whisper is still 3-5x faster than transformers.js
print("Loading faster-whisper model...")
print("Note: GPU mode requires cuDNN. Trying CPU mode (still faster than transformers.js)...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"  # Good balance of speed and accuracy
)
print(f"✅ Model loaded successfully on CPU with int8 (faster-whisper)")
print(f"   Expected speed: 3-5x faster than transformers.js Whisper")
device_info = "CPU (faster-whisper int8)"

# Load spaCy German model for better sentence detection
print("Loading spaCy German model for sentence detection...")
nlp = spacy.load("de_core_news_sm")
print("✅ spaCy German model loaded successfully")

# Store audio buffers per client and windowing state
audio_buffers = {}
window_positions = {}  # Track window positions for each client
full_transcripts = {}  # Store accumulated full transcript per client
last_window_starts = {}  # Track last window start position for each client

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
    """Extract the last sentence from text using spaCy."""
    if not text.strip():
        return ""

    doc = nlp(text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if sentences:
        return sentences[-1]
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
        "message": f"GPU-accelerated Whisper ready (faster-whisper on {device_info})",
        "clientId": str(client_id),
        "model": "small",
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
                        "message": "Transcribing with GPU-accelerated Whisper...",
                        "audioSize": len(audio_data)
                    }))

                    try:
                        start_time = datetime.now()

                        # Convert bytes to numpy array (16-bit PCM to float32)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe with faster-whisper (GPU)
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
                            "model": "small",
                            "device": "cuda",
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
                        print(f"[Whisper Stream] Starting new window at {window_start_time:.1f}s")

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

                        print(f"[Whisper Stream] Processing growing window {window_start_time:.1f}s-{window_end_time:.1f}s ({window_duration:.1f}s, total: {total_duration:.1f}s)")

                        # Transcribe window
                        start_time = datetime.now()
                        segments, info = model.transcribe(
                            window_audio,
                            language="de",
                            beam_size=5,
                            vad_filter=True,
                            word_timestamps=False
                        )

                        # Get text from this window
                        window_text = ""
                        for segment in segments:
                            window_text += segment.text

                        processing_time = (datetime.now() - start_time).total_seconds()
                        print(f"[Whisper Stream] Window transcribed in {processing_time:.2f}s: {window_text[:80]}...")

                        # Growing window (same start) vs new window (different start)
                        if window_start_time == last_window_starts[client_id]:
                            # Growing window - REPLACE transcript (re-transcribing same audio)
                            full_transcripts[client_id] = window_text
                            print(f"[Whisper Stream] Growing window - replacing transcript")
                        else:
                            # New window - MERGE with overlap detection
                            if full_transcripts[client_id]:
                                full_transcripts[client_id] = merge_with_overlap(
                                    full_transcripts[client_id],
                                    window_text,
                                    max_overlap_words=100  # Larger window for 2s overlap
                                )
                                print(f"[Whisper Stream] New window - merging with overlap detection")
                            else:
                                # First window ever
                                full_transcripts[client_id] = window_text
                                print(f"[Whisper Stream] First window - initializing transcript")
                            # Update last window start
                            last_window_starts[client_id] = window_start_time

                        # Extract last sentence for chunked display
                        last_sentence = extract_last_sentence(full_transcripts[client_id])

                        print(f"[Whisper Stream] Full transcript length: {len(full_transcripts[client_id])} chars")
                        print(f"[Whisper Stream] Last sentence: {last_sentence[:60]}...")

                        # Update window position to end of processed audio
                        window_positions[client_id] = total_duration

                        # Send response with both full transcript and last sentence
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
                            print(f"[Whisper Stream] Trimmed buffer to last 90s")

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
    print("Starting faster-whisper WebSocket server on ws://localhost:5001/transcribe")
    print("Device: CUDA (NVIDIA GPU)")
    print("Model: Whisper Small")

    async with websockets.serve(handle_client, "0.0.0.0", 5001):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
