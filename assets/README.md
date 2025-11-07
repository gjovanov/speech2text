# Architecture Diagrams

This folder contains the Mermaid source files (`.mmd`) for all architecture diagrams used in the main README.

## Viewing Diagrams

The diagrams are embedded in the main README.md using Mermaid syntax and will render automatically on GitHub.

## Generating PNG Files

If you need PNG versions of the diagrams, you can generate them using one of these methods:

### Method 1: Using Mermaid CLI (npm)

```bash
npm install -g @mermaid-js/mermaid-cli

# Generate all diagrams
mmdc -i system-architecture.mmd -o system-architecture.png -b transparent
mmdc -i data-flow.mmd -o data-flow.png -b transparent
mmdc -i docker-deployment.mmd -o docker-deployment.png -b transparent
mmdc -i realtime-streaming-flow.mmd -o realtime-streaming-flow.png -b transparent
mmdc -i comparison-mode-flow.mmd -o comparison-mode-flow.png -b transparent
mmdc -i integration-workflow.mmd -o integration-workflow.png -b transparent
```

### Method 2: Using Kroki API (Python)

```bash
python3 generate_diagrams.py
```

This script uses the Kroki online service to convert Mermaid diagrams to PNG.

### Method 3: Online Editors

Visit these sites and paste the `.mmd` file contents:
- **Mermaid Live Editor**: https://mermaid.live/
- **Kroki**: https://kroki.io/

Then download the PNG export.

## Available Diagrams

1. **system-architecture.mmd** - Overall system architecture showing all components
2. **data-flow.mmd** - Sequence diagram of data flow between components
3. **docker-deployment.mmd** - Docker containerization architecture
4. **realtime-streaming-flow.mmd** - User flow for real-time streaming mode
5. **comparison-mode-flow.mmd** - User flow for comparison mode
6. **integration-workflow.mmd** - Integration patterns for FFmpeg/WebRTC

## Note

PNG files are not committed to version control to keep the repository size small. GitHub renders Mermaid diagrams natively in Markdown files, so PNG files are only needed for external documentation or presentations.
