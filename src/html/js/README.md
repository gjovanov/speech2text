# RAG System Modules

This directory contains the modular RAG (Retrieval-Augmented Generation) correction system for the German transcription project.

## File Structure

```
js/
├── rag-system.js          # Core RAG classes (BrowserRAG, GermanTextProcessor)
├── llm-correctors.js      # LLM correction classes (Local & API-based)
├── rag-ui.js              # UI functions for manual/AI correction
├── rag-init.js            # Initialization and global bindings
└── README.md              # This file
```

## Module Descriptions

### `rag-system.js` (200 lines)
**Core RAG functionality**

Classes:
- `BrowserRAG` - Manages knowledge base with semantic embeddings
  - `initialize()` - Load multilingual-e5-small model
  - `addCorrection()` - Add correction with embedding
  - `retrieveSimilar()` - Semantic search for similar examples
  - `getStats()` - Get KB statistics
  - `clear()` - Delete all corrections

- `GermanTextProcessor` - Detects error patterns
  - `detectErrorPatterns()` - Find capitalization/compound word errors

### `llm-correctors.js` (300 lines)
**LLM correction implementations**

Classes:
- `LocalLLMCorrector` - Browser-based correction
  - Uses Qwen2.5-0.5B-Instruct (or Flan-T5 fallback)
  - ~500MB download, runs entirely in browser
  - No API key needed, private

- `APILLMCorrector` - Cloud-based correction
  - Supports OpenAI, Anthropic, Mistral
  - Higher accuracy, requires API key
  - Methods: `callOpenAI()`, `callAnthropic()`, `callMistral()`

Functions:
- `buildGermanCorrectionPrompt()` - Build German prompt with RAG examples

### `rag-ui.js` (400 lines)
**User interface functions**

Manual Editing:
- `manualEdit()` - Make text contentEditable
- `saveManualEdit()` - Save manual corrections

AI Correction:
- `aiCorrect()` - Main RAG correction workflow
- `acceptCorrection()` - Accept AI suggestion
- `rejectCorrection()` - Reject AI suggestion
- `editBeforeAccept()` - Edit AI suggestion before saving

Knowledge Base Management:
- `toggleKBPanel()` - Show/hide KB panel
- `refreshKBStats()` - Update stats display
- `renderKBList()` - Display recent corrections
- `exportKnowledgeBase()` - Export to JSON
- `importKnowledgeBase()` - Import from JSON
- `clearKnowledgeBase()` - Delete all

Settings Modal:
- `openRAGSettings()` - Show settings modal
- `saveRAGSettings()` - Save configuration
- `updateProviderSettings()` - Update provider UI

### `rag-init.js` (100 lines)
**Initialization and glue code**

Functions:
- `initializeRAGSystem()` - Create RAG instances
- `saveCorrection()` - Save to localStorage + RAG
- `setupGlobalFunctions()` - Export to window for onclick
- `initRAG()` - Main entry point

## Usage in HTML

```html
<!-- Load Transformers.js CDN first -->
<script type="module">
    import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
    env.allowLocalModels = false;
    window.transformersPipeline = pipeline;
</script>

<!-- Load RAG system -->
<script type="module">
    import { initRAG } from './js/rag-init.js';
    window.addEventListener('load', initRAG);
</script>
```

## Benefits of Modularization

### Before:
- `vod-transcription.html`: **2444 lines** (monolithic)
- `live-transcription.html`: **2100 lines** (monolithic)
- Total: ~4500 lines, all duplicated

### After:
- `vod-transcription.html`: **1347 lines** (-45%)
- `live-transcription.html`: **~1200 lines** (estimated)
- Shared modules: **1000 lines** (reusable)
- Total: ~3500 lines (-22% overall)

### Advantages:
1. **DRY (Don't Repeat Yourself)** - Single source of truth
2. **Easier Maintenance** - Fix bugs in one place
3. **Better Testing** - Test modules independently
4. **Cleaner Code** - Separation of concerns
5. **Faster Development** - Reuse across pages
6. **Smaller Files** - Easier to navigate and understand

## Architecture

```
┌─────────────────────────────────────┐
│  vod-transcription.html             │
│  live-transcription.html            │
│                                     │
│  (UI, WebSockets, Audio Playback)  │
└──────────────┬──────────────────────┘
               │
               │ imports
               ▼
       ┌───────────────┐
       │  rag-init.js  │
       └───────┬───────┘
               │
       ┌───────┴────────────────┐
       │                        │
       ▼                        ▼
┌──────────────┐        ┌──────────────┐
│ rag-system.js│        │llm-correctors│
│              │        │      .js     │
│ - BrowserRAG │        │- LocalLLM    │
│ - TextProc   │        │- APILLM      │
└──────────────┘        └──────────────┘
       │                        │
       └────────┬───────────────┘
                │
                ▼
        ┌───────────────┐
        │   rag-ui.js   │
        │               │
        │ - UI funcs    │
        │ - KB mgmt     │
        │ - Settings    │
        └───────────────┘
```

## Global Variables

These are set by `rag-init.js` and available throughout the page:

```javascript
window.ragSystem       // BrowserRAG instance
window.textProcessor   // GermanTextProcessor instance
window.llmCorrector    // LocalLLMCorrector or APILLMCorrector instance

// All UI functions are also available on window for onclick handlers
window.aiCorrect()
window.manualEdit()
window.toggleKBPanel()
// ... etc
```

## Development

To modify the RAG system:

1. **Edit modules** - Make changes in `js/` files
2. **Restart server** - `docker compose restart web`
3. **Test** - Visit http://localhost/vod-transcription.html
4. **Check console** - Look for RAG initialization messages

No need to edit multiple HTML files anymore!
