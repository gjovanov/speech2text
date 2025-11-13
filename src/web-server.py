#!/usr/bin/env python3
"""
Web Server with Dynamic File List API
Port: 8000
Serves static files and provides /api/files endpoint for media directory listing
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import json
from pathlib import Path
from urllib.parse import unquote

MEDIA_DIR = Path("/app/media")

class WebServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/app", **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        # Parse the path
        path = unquote(self.path.split('?')[0])

        # API endpoint for file list
        if path == '/api/files':
            self.send_file_list()
            return

        # Redirect root to realtime-transcription.html
        if path == '/' or path == '':
            self.send_response(302)
            self.send_header('Location', '/realtime-transcription.html')
            self.end_headers()
            return

        # Let SimpleHTTPRequestHandler handle static files
        super().do_GET()

    def send_file_list(self):
        """Send JSON list of WAV files in media directory"""
        try:
            # Get all WAV files in media directory
            wav_files = []
            if MEDIA_DIR.exists():
                for file in sorted(MEDIA_DIR.glob('*.wav')):
                    wav_files.append({
                        'filename': file.name,
                        'path': f'/media/{file.name}',
                        'size': file.stat().st_size
                    })

            response = {
                'success': True,
                'files': wav_files
            }

            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        """Custom logging format"""
        print(f"{self.address_string()} - {format % args}")

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), WebServerHandler)
    print("Web server started on port 8000")
    server.serve_forever()
