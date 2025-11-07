#!/usr/bin/env python3
"""
Generate PNG diagrams from Mermaid files using Kroki API
"""

import requests
import base64
import zlib
from pathlib import Path

def generate_diagram(mmd_file, output_file):
    """Generate PNG from Mermaid file using Kroki API"""

    # Read mermaid content
    with open(mmd_file, 'r') as f:
        mermaid_code = f.read()

    # Compress and encode
    compressed = zlib.compress(mermaid_code.encode('utf-8'), level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')

    # Generate diagram using Kroki API
    url = f"https://kroki.io/mermaid/png/{encoded}"

    print(f"Generating {output_file.name}...")

    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"✓ Generated {output_file.name} ({len(response.content)} bytes)")
        return True
    else:
        print(f"✗ Failed to generate {output_file.name}: HTTP {response.status_code}")
        return False

def main():
    assets_dir = Path(__file__).parent

    diagrams = [
        ('system-architecture.mmd', 'system-architecture.png'),
        ('data-flow.mmd', 'data-flow.png'),
        ('docker-deployment.mmd', 'docker-deployment.png'),
        ('realtime-streaming-flow.mmd', 'realtime-streaming-flow.png'),
        ('comparison-mode-flow.mmd', 'comparison-mode-flow.png'),
        ('integration-workflow.mmd', 'integration-workflow.png'),
    ]

    success_count = 0
    for mmd_file, png_file in diagrams:
        mmd_path = assets_dir / mmd_file
        png_path = assets_dir / png_file

        if not mmd_path.exists():
            print(f"✗ Source file not found: {mmd_file}")
            continue

        if generate_diagram(mmd_path, png_path):
            success_count += 1

    print(f"\n✓ Successfully generated {success_count}/{len(diagrams)} diagrams")

if __name__ == '__main__':
    main()
