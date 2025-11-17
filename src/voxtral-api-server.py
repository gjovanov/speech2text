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

# Store audio buffers and windowing state per client
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
    best_overlap = 0
    search_limit = min(max_overlap_words, len(existing_words), len(new_words))

    for overlap_len in range(search_limit, 0, -1):
        existing_suffix = existing_words[-overlap_len:]
        new_prefix = new_words[:overlap_len]

        if [w.lower() for w in existing_suffix] == [w.lower() for w in new_prefix]:
            best_overlap = overlap_len
            break

    if best_overlap > 0:
        unique_new_words = new_words[best_overlap:]
        merged = existing_text + " " + " ".join(unique_new_words)
        print(f"[Merge] Found {best_overlap}-word overlap, appending {len(unique_new_words)} new words")
        return merged.strip()
    else:
        print(f"[Merge] No overlap found, concatenating")
        return (existing_text + " " + new_text).strip()

def extract_last_sentence(text):
    """
    Extract the last sentence(s) from text using regex-based sentence splitting.
    Applies heuristic: if last sentence has ≤3 words, include previous sentences.
    - Last sentence ≤3 words + previous >3 words = return last 2 sentences
    - Last sentence ≤3 words + previous ≤3 words = return last 3 sentences (max)
    - Last sentence >3 words = return only last sentence
    """
    if not text.strip():
        return ""

    import re
    # Match sentence endings: period/!/? followed by space and capital letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text.strip())
    parts = [s.strip() for s in parts if s.strip()]

    if not parts:
        return text.strip()

    # First, handle date/abbreviation fragments (e.g., "19. November")
    if len(parts) >= 2:
        last_part = parts[-1]
        prev_part = parts[-2]

        if len(prev_part) < 10 and prev_part.endswith('.'):
            prev_without_period = prev_part[:-1].replace('.', '')
            if prev_without_period.isdigit() or len(prev_part.split()) == 1:
                # Merge the fragment with the following sentence
                parts[-2] = prev_part + " " + last_part
                parts.pop()  # Remove the last part (now merged)

    # Now apply the word count heuristic
    last_sentence = parts[-1]
    last_word_count = len(last_sentence.split())

    if last_word_count <= 3 and len(parts) > 1:
        # Last sentence has ≤3 words, check previous
        prev_sentence = parts[-2]
        prev_word_count = len(prev_sentence.split())

        if prev_word_count <= 3 and len(parts) > 2:
            # Both last and previous have ≤3 words, return last 3 sentences
            result = " ".join(parts[-3:])
        else:
            # Previous has >3 words, return last 2 sentences
            result = " ".join(parts[-2:])
    else:
        # Last sentence has >3 words, return only it
        result = last_sentence

    # Make sure result has proper punctuation
    if result and result[-1] not in '.!?':
        if text.strip().endswith('.'):
            result += '.'
        elif text.strip().endswith('!'):
            result += '!'
        elif text.strip().endswith('?'):
            result += '?'

    return result

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

async def transcribe_with_mistral(audio_path, language="de", return_timestamps=False):
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

        # Voxtral API doesn't support response_format parameter
        # Just return the text - timestamps not available from this API
        return transcription_response.text
    except Exception as e:
        print(f"[Mistral API] Error: {e}")
        raise e

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

                elif data.get("type") == "transcribe":
                    # Full transcription (VoD mode)
                    if not audio_buffers[client_id]:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No audio data received"
                        }))
                        continue

                    audio_data = b''.join(audio_buffers[client_id])

                    await websocket.send(json.dumps({
                        "type": "processing",
                        "message": "Transcribing with Voxtral Cloud API...",
                        "audioSize": len(audio_data)
                    }))

                    try:
                        start_time = datetime.now()

                        # Convert bytes to numpy array
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_duration = len(audio_data) / (16000 * 2)

                        # Use chunked processing for long audio files (> 60 seconds)
                        CHUNK_THRESHOLD = 60  # seconds

                        if audio_duration > CHUNK_THRESHOLD:
                            print(f"[Voxtral] Long audio detected ({audio_duration:.1f}s), using chunked processing")

                            # Split into chunks
                            chunks_data = chunk_audio_with_overlap(
                                audio_np,
                                sample_rate=16000,
                                chunk_seconds=30,
                                overlap_seconds=3
                            )

                            chunk_results = []

                            # Process each chunk
                            for i, chunk_data in enumerate(chunks_data):
                                chunk_audio = chunk_data['audio']
                                chunk_duration = len(chunk_audio) / 16000

                                # Save chunk to temp file
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                    tmp_path = tmp.name
                                    sf.write(tmp_path, chunk_audio, 16000)

                                # Transcribe chunk
                                chunk_start = datetime.now()
                                chunk_text = await transcribe_with_mistral(tmp_path, language="de")
                                chunk_time = (datetime.now() - chunk_start).total_seconds()

                                os.unlink(tmp_path)

                                chunk_results.append({
                                    'text': chunk_text,
                                    'start_time': chunk_data['start_time'],
                                    'end_time': chunk_data['end_time']
                                })

                                print(f"[Voxtral] Chunk {i+1}/{len(chunks_data)}: {chunk_duration:.1f}s → {chunk_time:.2f}s ({chunk_duration/chunk_time:.1f}x RT) - {chunk_text[:50]}...")

                                # Send progress update
                                progress_pct = ((i + 1) / len(chunks_data)) * 100
                                await websocket.send(json.dumps({
                                    "type": "processing",
                                    "message": f"Processing chunk {i+1}/{len(chunks_data)} ({progress_pct:.0f}%)",
                                    "progress": progress_pct
                                }))

                            # Merge all transcriptions
                            text = merge_transcriptions(chunk_results, overlap_seconds=3)
                            print(f"[Voxtral] Merged {len(chunks_data)} chunks into final transcription")

                        else:
                            # Original code for short audio (< 60 seconds)
                            print(f"[Voxtral] Short audio ({audio_duration:.1f}s), using standard processing")

                            # Save to temporary WAV file
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                tmp_path = tmp.name
                                sf.write(tmp_path, audio_np, 16000)

                            # Transcribe with Mistral API
                            text = await transcribe_with_mistral(tmp_path, language="de")

                            os.unlink(tmp_path)

                        processing_time = (datetime.now() - start_time).total_seconds()
                        rtfx = audio_duration / processing_time if processing_time > 0 else 0

                        print(f"[Voxtral] Result: {text[:100]}...")

                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": text,
                            "chunks": [],
                            "model": "voxtral-mini-latest",
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
                        print(f"[Voxtral Stream] Starting new window at {window_start_time:.1f}s")

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

                        print(f"[Voxtral Stream] Processing growing window {window_start_time:.1f}s-{window_end_time:.1f}s ({window_duration:.1f}s, total: {total_duration:.1f}s)")

                        # Save to temporary WAV file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp_path = tmp.name
                            sf.write(tmp_path, window_audio, 16000)

                        # Transcribe window
                        start_time = datetime.now()
                        window_text = await transcribe_with_mistral(tmp_path, language="de")
                        processing_time = (datetime.now() - start_time).total_seconds()

                        os.unlink(tmp_path)

                        print(f"[Voxtral Stream] Window transcribed in {processing_time:.2f}s: {window_text[:80]}...")

                        # Growing window (same start) vs new window (different start)
                        if window_start_time == last_window_starts[client_id]:
                            # Growing window - keep completed portion, replace current window portion
                            full_transcripts[client_id] = completed_transcripts[client_id] + window_text
                            print(f"[Voxtral Stream] Growing window - keeping completed ({len(completed_transcripts[client_id])} chars) + new window")
                        else:
                            # New window - merge previous full transcript into completed, start new window
                            if full_transcripts[client_id]:
                                # Merge the previous window's text into completed transcripts
                                completed_transcripts[client_id] = merge_with_overlap(
                                    completed_transcripts[client_id],
                                    full_transcripts[client_id],
                                    max_overlap_words=100  # Larger window for 2s overlap
                                )
                                print(f"[Voxtral Stream] New window - merged previous window into completed")

                            # Now start the new window
                            full_transcripts[client_id] = completed_transcripts[client_id] + window_text
                            print(f"[Voxtral Stream] New window - completed ({len(completed_transcripts[client_id])} chars) + new window")

                            # Update last window start
                            last_window_starts[client_id] = window_start_time

                        # Extract last sentence
                        last_sentence = extract_last_sentence(full_transcripts[client_id])

                        print(f"[Voxtral Stream] Full transcript length: {len(full_transcripts[client_id])} chars")
                        print(f"[Voxtral Stream] Last sentence: {last_sentence[:60]}...")

                        # Update window position to end of processed audio
                        window_positions[client_id] = total_duration

                        # Send response
                        if window_text.strip():
                            await websocket.send(json.dumps({
                                "type": "partial_transcription",
                                "text": window_text,
                                "fullTranscript": full_transcripts[client_id],
                                "lastSentence": last_sentence,
                                "isPartial": True,
                                "windowInfo": {
                                    "start": window_start_time,
                                    "end": window_end_time,
                                    "duration": window_duration
                                }
                            }))

                        # Clean old audio buffer
                        max_buffer_bytes = 90 * 16000 * 2
                        if total_bytes > max_buffer_bytes:
                            keep_from_byte = total_bytes - max_buffer_bytes
                            audio_buffers[client_id] = [audio_data[keep_from_byte:]]
                            window_positions[client_id] = min(window_positions[client_id], 90.0)
                            print(f"[Voxtral Stream] Trimmed buffer to last 90s")

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
