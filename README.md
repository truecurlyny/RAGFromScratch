# RAG From Scratch

A minimal, fully local, fully free RAG (Retrieval-Augmented Generation) pipeline.
No cloud APIs, no API keys, no billing. Everything runs on your machine.

Test corpus: four short documents about a fictional product, "Nimbus Vault,"
with specific facts (prices, error codes, retention windows) that a general-purpose
LLM cannot possibly know — so if the assistant answers correctly, you know
retrieval actually worked, rather than the model guessing from general knowledge.

> **Learning by building.** This is a study project. If you want the *concepts*
> behind every moving part — the two kinds of model, what Ollama actually is,
> quantization, why the query must use the same embedding model as the chunks —
> read [RAG-Learning-Notes.md](RAG-Learning-Notes.md) alongside the code. The
> notes are written from a first-timer's point of view, one concept at a time.

## Pipeline

```
data/*.txt  --chunk-->  overlapping text chunks
                --embed (sentence-transformers)-->  vectors
                --store (ChromaDB, local, persisted to disk)-->  vector index
                --query: embed question, retrieve top-k chunks-->  context
                --augment: stuff context into a prompt-->
                --generate (Ollama, local LLM)-->  answer
```

Each stage is its own file in `src/`, written without a framework (no LangChain),
so every step is visible: `chunking.py` -> `ingest.py` -> `query.py`.

## Repository layout

```
RAGFromScratch/
├── data/                    corpus: 4 .txt files about fictional "Nimbus Vault"
├── src/
│   ├── chunking.py          Stage 1 — split docs into overlapping word-windows
│   ├── ingest.py            Stages 2-3 — embed each chunk, store it in ChromaDB
│   └── query.py             Stages 4-6 — retrieve, augment the prompt, generate
├── RAG-Learning-Notes.md    plain-language notes on every concept in the pipeline
├── requirements.txt         Python dependencies
└── README.md                you are here
```

`chroma_db/` (the vector index) is created locally by `ingest.py` and is
git-ignored — you build it yourself in step 4 below.

Tip: `python src/chunking.py` runs on its own with **no** models or Ollama
required, so you can inspect real chunks before setting anything else up.

## Prerequisites

This was built and tested on macOS (Apple Silicon), but nothing here is
Mac-specific except the install commands. You need:

- **Python 3.10 or newer** — check with `python3 --version`. If it's missing,
  install it with `brew install python@3.12` or from https://www.python.org.
- **Homebrew** (macOS) — used below to install Ollama. Get it at
  https://brew.sh. On Linux/Windows, install Python and Ollama directly
  instead; the `pip` and `python` steps are identical.
- **~3 GB of free disk space** — the LLM (`llama3.2`) is ~2 GB and the
  embedding model is ~80 MB, both downloaded once and cached locally.
- **No internet after the first run, and no API keys** — everything runs
  on your machine, so once the models are pulled you can go fully offline.

## Setup (macOS, Apple Silicon)

1. **Install Ollama** (the local LLM server):
   ```
   brew install ollama
   ```
   or download the app from https://ollama.com. Start it once (either
   `ollama serve` in a terminal, or launch the Ollama app — it runs in
   the background).

2. **Pull a model.** On a 32GB M-series Mac, a good starting point:
   ```
   ollama pull llama3.2
   ```
   This is a 3B model, fast and fits comfortably in memory. For better
   answer quality (still fast on your hardware), you can instead use:
   ```
   ollama pull qwen2.5:7b
   ```
   If you switch models, update `LLM_MODEL` in `src/query.py` to match.

3. **Create a virtual environment and install Python dependencies:**
   ```
   cd RAGFromScratch
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Build the vector index** (run this once, and again any time you
   change files in `data/`):
   ```
   python src/ingest.py
   ```
   First run will download the embedding model (~80MB) and may take a
   minute; after that it's fast.

5. **Ask questions.** This starts an interactive prompt — type a question
   at the `You:` line and press Enter; type `quit` (or `exit`) to leave.
   ```
   python src/query.py
   ```
   Try questions like:
   - "How much does the Business plan cost and how many devices does it support?"
   - "What does error NV-204 mean and how do I fix it?"
   - "Is SMS two-factor authentication still supported?"
   - "How long is version history kept on each plan?"

   Before each answer it prints the retrieved chunks with a distance score
   (lower = more similar), so you can watch retrieval and generation as two
   separate, inspectable steps.

## Using your own documents

Drop `.txt` files into `data/`, then re-run `python src/ingest.py`. PDF/DOCX
support isn't included here on purpose — the point of this project is to see
chunking/embedding/retrieval clearly. Add a PDF-to-text step separately if
you want to index PDFs.

## Tuning knobs worth experimenting with

- `chunk_size` / `overlap` in `src/chunking.py` — smaller chunks give more
  precise retrieval but less surrounding context per chunk.
- `TOP_K` in `src/query.py` — how many chunks get stuffed into the prompt.
- `LLM_MODEL` in `src/query.py` — any model you've pulled with `ollama pull`.
