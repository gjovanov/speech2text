#!/usr/bin/env python3
"""Generate favicon files - Simple and expressive design"""
from PIL import Image, ImageDraw, ImageFont
import io

def create_favicon():
    """Create a simple, bold favicon with waveform and text"""
    sizes = [16, 32, 48, 64, 128, 256]
    images = []

    for size in sizes:
        # Create image with transparency
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Scale factor
        s = size / 100.0

        # Bold gradient background circle
        draw.ellipse(
            [(0, 0), (size, size)],
            fill=(102, 126, 234)  # Purple-blue
        )

        # Add subtle inner glow
        draw.ellipse(
            [(5 * s, 5 * s), (95 * s, 95 * s)],
            fill=(118, 75, 162)  # Darker purple
        )

        if size >= 32:
            # Draw bold waveform bars (audio visualization)
            bar_width = max(2, int(4 * s))
            spacing = max(3, int(6 * s))
            start_x = int(20 * s)

            # 5 bars with varying heights (waveform pattern)
            heights = [0.6, 0.85, 0.5, 0.75, 0.4]  # Normalized heights

            for i, height in enumerate(heights):
                x = start_x + i * (bar_width + spacing)
                bar_height = int(40 * s * height)
                y_top = int(50 * s - bar_height / 2)
                y_bottom = int(50 * s + bar_height / 2)

                # Draw bold white bars
                draw.rectangle(
                    [(x, y_top), (x + bar_width, y_bottom)],
                    fill=(255, 255, 255, 255)
                )
        else:
            # For very small sizes (16x16), just show 3 simple bars
            bar_width = max(1, int(2 * s))
            spacing = max(2, int(3 * s))
            start_x = int(30 * s)

            heights = [0.5, 0.8, 0.5]

            for i, height in enumerate(heights):
                x = start_x + i * (bar_width + spacing)
                bar_height = int(30 * s * height)
                y_top = int(50 * s - bar_height / 2)
                y_bottom = int(50 * s + bar_height / 2)

                draw.rectangle(
                    [(x, y_top), (x + bar_width, y_bottom)],
                    fill=(255, 255, 255, 255)
                )

        # Add "AI" text for larger sizes
        if size >= 48:
            try:
                # Try to use a bold system font
                font_size = int(24 * s)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                text = "AI"

                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position text in bottom right
                text_x = int(62 * s)
                text_y = int(58 * s)

                # Draw text with shadow for depth
                draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 100), font=font)
                draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
            except:
                # Fallback: draw simple shapes for "AI"
                # Draw "A"
                draw.polygon(
                    [(60 * s, 75 * s), (65 * s, 60 * s), (70 * s, 75 * s)],
                    fill=(255, 255, 255, 255)
                )
                # Draw "I"
                draw.rectangle(
                    [(75 * s, 60 * s), (78 * s, 75 * s)],
                    fill=(255, 255, 255, 255)
                )

        images.append(img)

    return images

if __name__ == '__main__':
    print("Generating favicon files...")

    images = create_favicon()

    # Save PNG files
    for i, size in enumerate([16, 32, 48, 64, 128, 256]):
        filename = f'src/html/favicon-{size}x{size}.png'
        images[i].save(filename, 'PNG')
        print(f"✓ Created {filename}")

    # Save main favicon.png (32x32)
    images[1].save('src/html/favicon.png', 'PNG')
    print("✓ Created src/html/favicon.png")

    # Save ICO file with multiple sizes
    images[0].save(
        'src/html/favicon.ico',
        format='ICO',
        sizes=[(16, 16), (32, 32), (48, 48)]
    )
    print("✓ Created src/html/favicon.ico")

    # Save apple-touch-icon
    images[5].save('src/html/apple-touch-icon.png', 'PNG')
    print("✓ Created src/html/apple-touch-icon.png")

    print("\n✅ All favicon files generated successfully!")
