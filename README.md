# inf-mem

[![CI](https://github.com/Olivesz/inf-mem/actions/workflows/ci.yml/badge.svg)](https://github.com/Olivesz/inf-mem/actions/workflows/ci.yml)

A persistent memory layer for LLMs. Runs locally on your machine and works across ChatGPT and Claude — context that actually sticks.

---

## The problem

LLMs forget everything the moment a conversation ends. You re-explain your projects, your preferences, your background — over and over. `inf-mem` fixes that by maintaining a personal memory store that lives outside any single conversation.

## How it works

```
You type a prompt on ChatGPT or Claude
            ↓
Chrome extension intercepts it
            ↓
Fetches relevant memories from local backend (semantic search)
            ↓
Shows review panel: "injecting these 3 memories — send?"
            ↓
Injects memory context into your prompt, sends it
            ↓
After response arrives → Groq extracts new memories → stored locally
```

Memory persists across every conversation, forever, entirely on your machine.

---

## Memory types

| Type | What it stores | Example |
|---|---|---|
| `FACT` | Who you are | "User studies CS at UPenn" |
| `PROJECT` | Active work | "Working on a comic OCR pipeline in Python" |
| `PREFERENCE` | How you like AI to respond | "Prefers concise answers, no bullet points" |
| `EPISODE` | Significant past context | "Had a long debug session on ChromaDB embeddings" |

---

## Architecture

```
Chrome Extension (MV3)
├── content/chatgpt.js    — intercepts prompts on ChatGPT
├── content/claude.js     — intercepts prompts on Claude
├── content/shared.js     — shared utilities, panel injection
├── background.js         — service worker, routes to backend
└── popup/                — memory browser + settings

Local Python Backend (FastAPI @ localhost:7842)
├── POST /retrieve        — semantic search for relevant memories
├── POST /extract         — Groq extracts memories from prompt+response
├── POST /compress        — Groq compresses verbose prompts
├── GET  /memories        — list all stored memories
├── DELETE /memories/{id} — delete a memory
└── GET  /health          — status check

Storage (in ~/.inf_mem/)
├── memories.db           — SQLite: text, type, keywords, importance
└── embeddings.npy        — numpy: 384-dim vectors (BAAI/bge-small-en-v1.5)
```

**LLM**: [Groq](https://console.groq.com) free tier — llama-3.1-8b-instant. Fast, free, no GPU.
**Embeddings**: [fastembed](https://github.com/qdrant/fastembed) via ONNX Runtime — no PyTorch needed. Downloads ~130MB model once on first use.

---

## Quickstart

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create. Free tier: 14,400 requests/day.

### 2. Configure

```bash
cp config.example.yaml config.yaml
# Open config.yaml and paste your Groq key
```

### 3. Install and start backend

```bash
make install
make start
make status   # should show: {"status": "ok", "groq_connected": true}
```

On first use, fastembed downloads the embedding model (~130MB) to ~/.cache/fastembed/. One-time only.

### 4. Load the Chrome extension

1. Open chrome://extensions/
2. Enable Developer mode (top right toggle)
3. Click Load unpacked → select the extension/ folder
4. Pin the extension from the puzzle-piece icon

### 5. Use it

Go to chatgpt.com or claude.ai and chat normally. On each prompt send:
- If relevant memories exist → review panel slides in showing what will be injected
- Click "Send with memory" to inject, or "Skip" to send as-is
- After response → extraction runs silently in the background

Click the extension icon to browse and manage your memory store.

---

## Makefile reference

| Command | What it does |
|---|---|
| make install | Install Python dependencies |
| make start | Start backend in background |
| make dev | Start with auto-reload (foreground) |
| make stop | Stop background server |
| make status | Check backend health |
| make memories | Print all stored memories as JSON |
| make clear | Delete all memories (asks for confirmation) |
| make clean | Remove __pycache__ files |

---

## Prompt compression (optional)

Toggle in the extension popup. When on, Groq rewrites verbose/informal prompts to be more concise before sending — reducing tokens enough to offset the injection overhead.

---

## Project structure

```
inf-mem/
├── backend/
│   ├── main.py           # FastAPI app, endpoint routing
│   ├── memory_store.py   # SQLite + numpy storage layer
│   ├── retriever.py      # fastembed encoding + cosine similarity search
│   ├── extractor.py      # Groq-powered memory extraction (ADD/UPDATE/NOOP)
│   ├── compressor.py     # Groq-powered prompt compression
│   └── models.py         # Pydantic schemas
├── extension/
│   ├── manifest.json     # MV3 manifest
│   ├── background.js     # Service worker (routes extension to backend)
│   ├── content/
│   │   ├── shared.js     # Panel injection, memory formatting
│   │   ├── chatgpt.js    # ChatGPT DOM interception
│   │   └── claude.js     # Claude DOM interception
│   ├── panel/
│   │   └── panel.css     # Review panel styles
│   └── popup/
│       ├── popup.html    # Extension popup
│       ├── popup.css     # Popup styles
│       └── popup.js      # Popup logic
├── config.example.yaml   # Config template (copy to config.yaml)
├── config.yaml           # Your config with Groq key (gitignored)
├── requirements.txt
└── Makefile
```

---

## Roadmap

- [ ] Memory consolidation: merge similar memories, decay stale ones over time
- [ ] A-MEM-style Zettelkasten linking between related memories
- [ ] Firefox extension support
- [ ] Gemini and Perplexity content scripts
- [ ] Memory export/import (JSON)

## License

MIT — see [LICENSE](LICENSE).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q  (models, store round-trips, retrieval, compression — no network)
```
