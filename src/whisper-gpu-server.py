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
import torch

# Initialize model - try GPU first, fall back to CPU
print("Loading faster-whisper model...")

# Check for GPU availability
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"  # Use float16 for GPU
    )
    device_info = f"GPU (faster-whisper float16, small) - {torch.cuda.get_device_name(0)}"
    print(f"✅ Model loaded successfully on GPU with float16")
else:
    print("No GPU detected, using CPU")
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"  # Good balance of speed and accuracy on CPU
    )
    device_info = "CPU (faster-whisper int8, small)"
    print(f"✅ Model loaded successfully on CPU with int8")

# Load spaCy German model for better sentence detection
print("Loading spaCy German model for sentence detection...")
nlp = spacy.load("de_core_news_sm")
print("✅ spaCy German model loaded successfully")

# Store audio buffers per client and windowing state
audio_buffers = {}
window_positions = {}  # Track window positions for each client
full_transcripts = {}  # Store accumulated full transcript per client
last_window_starts = {}  # Track last window start position for each client
completed_transcripts = {}  # Store text from completed windows (before current window)

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
    """
    Extract the last sentence(s) from text using spaCy.
    Applies heuristic: if last sentence has ≤3 words, include previous sentences.
    - Last sentence ≤3 words + previous >3 words = return last 2 sentences
    - Last sentence ≤3 words + previous ≤3 words = return last 3 sentences (max)
    - Last sentence >3 words = return only last sentence
    """
    if not text.strip():
        return ""

    doc = nlp(text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return text.strip()

    # First, handle date/abbreviation fragments (e.g., "19. November")
    # Merge if the "previous sentence" is very short and looks like a fragment
    if len(sentences) >= 2:
        last_sent = sentences[-1]
        prev_sent = sentences[-2]

        if len(prev_sent) < 10:  # Short fragment
            is_likely_fragment = (
                prev_sent[-1] == '.' and
                (prev_sent[:-1].replace('.', '').isdigit() or  # Number like "19."
                 len(prev_sent.split()) == 1)  # Single word abbreviation
            )

            if is_likely_fragment:
                # Merge the fragment with the following sentence
                sentences[-2] = prev_sent + " " + last_sent
                sentences.pop()  # Remove the last sentence (now merged)

    # Now apply the word count heuristic
    last_sentence = sentences[-1]
    last_word_count = len(last_sentence.split())

    if last_word_count <= 3 and len(sentences) > 1:
        # Last sentence has ≤3 words, check previous
        prev_sentence = sentences[-2]
        prev_word_count = len(prev_sentence.split())

        if prev_word_count <= 3 and len(sentences) > 2:
            # Both last and previous have ≤3 words, return last 3 sentences
            return " ".join(sentences[-3:])
        else:
            # Previous has >3 words, return last 2 sentences
            return " ".join(sentences[-2:])

    # Last sentence has >3 words, return only it
    return last_sentence

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
    window_positions[client_id] = 0  # Track current window end position
    full_transcripts[client_id] = ""  # Accumulated full transcript
    last_window_starts[client_id] = None  # Track last window start position
    completed_transcripts[client_id] = ""  # Text from completed windows

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
                    # Full transcription (VoD mode)
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
                        audio_duration = len(audio_data) / (16000 * 2)

                        # Use chunked processing for long audio files (> 60 seconds)
                        CHUNK_THRESHOLD = 60  # seconds

                        if audio_duration > CHUNK_THRESHOLD:
                            print(f"[Whisper] Long audio detected ({audio_duration:.1f}s), using chunked processing")

                            # Split into chunks
                            chunks_data = chunk_audio_with_overlap(
                                audio_np,
                                sample_rate=16000,
                                chunk_seconds=30,
                                overlap_seconds=3
                            )

                            chunk_results = []
                            all_segments = []

                            # Process each chunk
                            for i, chunk_data in enumerate(chunks_data):
                                chunk_audio = chunk_data['audio']
                                chunk_duration = len(chunk_audio) / 16000

                                # Transcribe chunk
                                chunk_start = datetime.now()
                                segments, info = model.transcribe(
                                    chunk_audio,
                                    language="de",
                                    beam_size=5,
                                    vad_filter=True,
                                    word_timestamps=True
                                )

                                # Collect chunk text and segments with word timestamps
                                chunk_text = ""
                                for segment in segments:
                                    chunk_text += segment.text

                                    # Extract word-level timestamps
                                    words_data = []
                                    if hasattr(segment, 'words') and segment.words:
                                        for word in segment.words:
                                            words_data.append({
                                                "word": word.word,
                                                "start": word.start + chunk_data['start_time'],
                                                "end": word.end + chunk_data['start_time']
                                            })

                                    # Adjust timestamps relative to full audio
                                    all_segments.append({
                                        "timestamp": [segment.start + chunk_data['start_time'], segment.end + chunk_data['start_time']],
                                        "text": segment.text,
                                        "words": words_data
                                    })

                                chunk_time = (datetime.now() - chunk_start).total_seconds()

                                chunk_results.append({
                                    'text': chunk_text,
                                    'start_time': chunk_data['start_time'],
                                    'end_time': chunk_data['end_time']
                                })

                                print(f"[Whisper] Chunk {i+1}/{len(chunks_data)}: {chunk_duration:.1f}s → {chunk_time:.2f}s ({chunk_duration/chunk_time:.1f}x RT) - {chunk_text[:50]}...")

                                # Send progress update
                                progress_pct = ((i + 1) / len(chunks_data)) * 100
                                await websocket.send(json.dumps({
                                    "type": "processing",
                                    "message": f"Processing chunk {i+1}/{len(chunks_data)} ({progress_pct:.0f}%)",
                                    "progress": progress_pct
                                }))

                            # Merge all transcriptions
                            full_text = merge_transcriptions(chunk_results, overlap_seconds=3)
                            print(f"[Whisper] Merged {len(chunks_data)} chunks into final transcription")
                            print(f"[Whisper] Collected {len(all_segments)} segments with timestamps")

                        else:
                            # Original code for short audio (< 60 seconds)
                            print(f"[Whisper] Short audio ({audio_duration:.1f}s), using standard processing")

                            # Transcribe with faster-whisper
                            segments, info = model.transcribe(
                                audio_np,
                                language="de",  # German
                                beam_size=5,
                                vad_filter=True,  # Voice activity detection
                                word_timestamps=True
                            )

                            # Collect all segments with word timestamps
                            full_text = ""
                            all_segments = []
                            for segment in segments:
                                full_text += segment.text

                                # Extract word-level timestamps
                                words_data = []
                                if hasattr(segment, 'words') and segment.words:
                                    for word in segment.words:
                                        words_data.append({
                                            "word": word.word,
                                            "start": word.start,
                                            "end": word.end
                                        })

                                all_segments.append({
                                    "timestamp": [segment.start, segment.end],
                                    "text": segment.text,
                                    "words": words_data
                                })

                            print(f"[Whisper] Short audio: Collected {len(all_segments)} segments with timestamps")

                        processing_time = (datetime.now() - start_time).total_seconds()
                        rtfx = audio_duration / processing_time if processing_time > 0 else 0

                        print(f"[Whisper] Sending response with {len(all_segments)} segments")
                        print(f"[Whisper] Full text length: {len(full_text)} characters")
                        print(f"[Whisper] First 100 chars: {full_text[:100]}")

                        response = {
                            "type": "transcription",
                            "text": full_text,
                            "segments": all_segments,
                            "model": "large-v3",
                            "device": device_info,
                            "performance": {
                                "processingTime": f"{processing_time:.2f}s",
                                "audioDuration": f"{audio_duration:.2f}s",
                                "rtfx": f"{rtfx:.2f}x"
                            }
                        }

                        await websocket.send(json.dumps(response))
                        print(f"[Whisper] Response sent successfully")

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
                            word_timestamps=True
                        )

                        # Get text from this window
                        window_text = ""
                        for segment in segments:
                            window_text += segment.text

                        processing_time = (datetime.now() - start_time).total_seconds()
                        print(f"[Whisper Stream] Window transcribed in {processing_time:.2f}s: {window_text[:80]}...")

                        # Growing window (same start) vs new window (different start)
                        if window_start_time == last_window_starts[client_id]:
                            # Growing window - keep completed portion, replace current window portion
                            full_transcripts[client_id] = completed_transcripts[client_id] + window_text
                            print(f"[Whisper Stream] Growing window - keeping completed ({len(completed_transcripts[client_id])} chars) + new window")
                        else:
                            # New window - merge previous full transcript into completed, start new window
                            if full_transcripts[client_id]:
                                # Merge the previous window's text into completed transcripts
                                completed_transcripts[client_id] = merge_with_overlap(
                                    completed_transcripts[client_id],
                                    full_transcripts[client_id],
                                    max_overlap_words=100  # Larger window for 2s overlap
                                )
                                print(f"[Whisper Stream] New window - merged previous window into completed")

                            # Now start the new window
                            full_transcripts[client_id] = completed_transcripts[client_id] + window_text
                            print(f"[Whisper Stream] New window - completed ({len(completed_transcripts[client_id])} chars) + new window")

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
                    completed_transcripts[client_id] = ""
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
        if client_id in completed_transcripts:
            del completed_transcripts[client_id]

async def main():
    print("Starting faster-whisper WebSocket server on ws://localhost:5001/transcribe")
    print("Device: CUDA (NVIDIA GPU)")
    print("Model: Whisper large-v3")

    async with websockets.serve(handle_client, "0.0.0.0", 5001):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
