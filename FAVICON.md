# Favicon Documentation

## Design Concept

The favicon for this Real-time Transcription project features a **simple, bold, and expressive design**:

- **🎵 Waveform Bars**: 5 bold vertical bars representing audio visualization/equalizer
- **🤖 "AI" Text**: Clearly indicates AI-powered transcription
- **🎨 Gradient Background**: Purple to blue radial gradient matching the UI theme (#667eea to #764ba2)
- **✨ Clean & Modern**: Instantly recognizable at any size, from 16x16 to 256x256

## Files Generated

### Icon Files
- `favicon.ico` - Multi-resolution ICO file (16x16, 32x32, 48x48)
- `favicon.png` - Standard 32x32 PNG favicon
- `favicon.svg` - Scalable vector version (best quality)
- `apple-touch-icon.png` - 256x256 for iOS/Apple devices

### Multi-resolution PNG Files
- `favicon-16x16.png` - For smaller displays
- `favicon-32x32.png` - Standard size
- `favicon-48x48.png` - Medium size
- `favicon-64x64.png` - Larger displays
- `favicon-128x128.png` - High-DPI displays
- `favicon-256x256.png` - Retina displays

### Manifest
- `site.webmanifest` - PWA manifest for installable web app support

## Regenerating Favicons

If you need to regenerate the favicons with modifications:

```bash
cd /home/gjovanov/grox/text2speech
python3 generate-favicon.py
```

Then rebuild the web container:

```bash
docker compose build web && docker compose up -d web
```

## Customization

To modify the favicon design, edit `generate-favicon.py`:

- **Colors**: Adjust RGB values in the gradient and icon elements
- **Size**: Modify the `sizes` array to add/remove resolutions
- **Design**: Change the drawing commands in `create_favicon()` function

## Browser Compatibility

The favicon setup supports:
- ✅ Chrome/Edge (all versions)
- ✅ Firefox (all versions)
- ✅ Safari (desktop and iOS)
- ✅ Opera
- ✅ PWA installation on mobile devices
- ✅ High-DPI/Retina displays

## Theme Color

The project uses `#667eea` as the theme color, which:
- Matches the UI gradient
- Sets the browser toolbar color on mobile devices
- Provides consistent branding across platforms
