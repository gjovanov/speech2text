#!/usr/bin/env python3
"""
File Upload Server with MP3 to WAV conversion
Port: 9000
Handles file uploads and converts MP3 to WAV using FFmpeg
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
import cgi
import subprocess
import tempfile
from pathlib import Path

UPLOAD_DIR = Path("/app/media")
UPLOAD_DIR.mkdir(exist_ok=True)

class UploadHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle file upload"""
        if self.path != '/upload':
            self.send_error(404)
            return

        try:
            # Parse multipart form data
            content_type = self.headers['Content-Type']
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Expected multipart/form-data")
                return

            # Parse the form
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )

            if 'file' not in form:
                self.send_error(400, "No file provided")
                return

            file_item = form['file']
            if not file_item.filename:
                self.send_error(400, "No filename")
                return

            filename = os.path.basename(file_item.filename)
            file_data = file_item.file.read()

            print(f"Received file: {filename} ({len(file_data)} bytes)")

            # Determine file type
            file_ext = Path(filename).suffix.lower()

            if file_ext == '.mp3':
                # Convert MP3 to WAV
                output_filename = Path(filename).stem + '.wav'
                output_path = UPLOAD_DIR / output_filename

                # Save MP3 to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    tmp_mp3.write(file_data)
                    tmp_mp3_path = tmp_mp3.name

                try:
                    # Convert using FFmpeg
                    print(f"Converting {filename} to WAV...")
                    subprocess.run([
                        'ffmpeg', '-i', tmp_mp3_path,
                        '-ar', '16000',  # 16kHz sample rate
                        '-ac', '1',      # Mono
                        '-y',            # Overwrite
                        str(output_path)
                    ], check=True, capture_output=True)

                    print(f"✓ Converted to {output_filename}")

                finally:
                    # Clean up temp file
                    os.unlink(tmp_mp3_path)

            elif file_ext == '.wav':
                # Save WAV directly (optionally re-encode to ensure 16kHz mono)
                output_filename = filename
                output_path = UPLOAD_DIR / output_filename

                # Save to temporary file first
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                    tmp_wav.write(file_data)
                    tmp_wav_path = tmp_wav.name

                try:
                    # Re-encode to ensure correct format (16kHz, mono, 16-bit)
                    print(f"Processing {filename}...")
                    subprocess.run([
                        'ffmpeg', '-i', tmp_wav_path,
                        '-ar', '16000',  # 16kHz sample rate
                        '-ac', '1',      # Mono
                        '-y',            # Overwrite
                        str(output_path)
                    ], check=True, capture_output=True)

                    print(f"✓ Saved {output_filename}")

                finally:
                    # Clean up temp file
                    os.unlink(tmp_wav_path)

            else:
                self.send_error(400, f"Unsupported file type: {file_ext}. Only WAV and MP3 allowed.")
                return

            # Success response
            response = {
                'success': True,
                'filename': output_filename,
                'path': f'/media/{output_filename}',
                'message': f'File uploaded successfully'
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
            print(f"✗ {error_msg}")
            self.send_error(500, error_msg)
        except Exception as e:
            print(f"✗ Upload error: {str(e)}")
            self.send_error(500, str(e))

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'healthy')
        else:
            self.send_error(404)

if __name__ == '__main__':
    print("="*60)
    print("File Upload Server")
    print("="*60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"URL: http://0.0.0.0:9000/upload")
    print(f"Max file size: 2GB")
    print(f"Supported formats: WAV, MP3 (auto-converted to 16kHz mono WAV)")
    print("="*60 + "\n")

    server = HTTPServer(('0.0.0.0', 9000), UploadHandler)
    server.serve_forever()
