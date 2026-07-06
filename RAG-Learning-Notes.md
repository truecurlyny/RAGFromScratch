# RAG From Scratch — Learning Notes

> These are my study notes from building a local RAG pipeline step-by-step in Claude Code.
> I'm a first-timer to the whole topic. I learned it BY BUILDING, one concept at a time —
> explain, quiz me, run and inspect each step, then move on.

> These notes now go the WHOLE way: problem → concepts → build → run → a correct grounded
> answer. Read them next to the code in src/ (the code is finished and runs — see README to
> set it up). The notes are the WHY; the code is the WHAT.

---

## 0. The project in one line
A **fully local, fully free RAG** (Retrieval-Augmented Generation) pipeline on a Mac —
no cloud APIs, no keys. Test corpus = 4 text files about a **fictional** product
"Nimbus Vault" with made-up facts (prices, error codes). If the LLM answers correctly,
we KNOW retrieval worked (it couldn't have guessed fictional facts).

---

## 1. What problem does RAG solve?
An LLM only knows its training data. Ask about private/fictional facts and it either says
"I don't know" or **makes something up (hallucinates)**.

**RAG's trick:** before asking the LLM, *find* the relevant documents yourself, *paste them
into the prompt*, and say "answer using ONLY this."
- **R**etrieval = find the right text
- **A**ugment = add it to the prompt
- **G**eneration = LLM writes the answer

All the craft is in making retrieval fast + accurate. No model retraining needed.

---

## 2. The two models (KEY distinction)
There are **two different KINDS of model** doing two different jobs:

| | Embedding model | LLM (generation model) |
|---|---|---|
| Name here | `all-MiniLM-L6-v2` | `llama3.2` (or `qwen2.5:7b`) |
| Job | text → **vector** (list of numbers) | prompt → **text** (the answer) |
| Role | "reader" / fingerprints meaning | "writer" / predicts next word |
| Size | ~80 MB | ~2 GB |
| Runs via | **sentence-transformers** library (in-process) | **Ollama** server (localhost:11434) |
| Used in stages | 2 (embed chunks) AND 4 (embed question) | 6 (generate) only |

⚠️ `llama3.2` vs `qwen2.5:7b` are NOT "two models" in this sense — they're two interchangeable
**options for the single LLM slot** (like V6 vs V8 for one engine bay).

---

## 3. What is Ollama / an "LLM server"?
- An LLM file is just a **binary bag of numbers** (weights) on disk — inert on its own.
- To USE it, a program must **load** it into memory and **run** the math (inference) to
  predict the next word. That program is an **inference engine / runtime**.
- **Ollama** = that runtime, made easy. Two jobs:
  1. **Model manager** — `ollama pull llama3.2` downloads the weights for you.
  2. **Local server** — runs in the background at `http://localhost:11434`; programs send
     it a prompt and get text back. Loads the model ONCE and keeps it "warm."
- Our code (`query.py`) calls `ollama.chat(...)` → sends prompt over localhost → Ollama runs
  llama3.2 → returns the answer. Python is the customer; Ollama is the kitchen.
- Ollama mimics the **OpenAI API format**, so OpenAI code can point at local Ollama.

**Library vs server — why the difference?** Size + usage pattern:
- Embedding model (80 MB) loads in ~1s → just import it as a library, no server needed.
- LLM (2 GB) is slow to load + memory-heavy → put it behind a server that loads it ONCE and
  stays warm, so every question after is fast.
- Rule of thumb: **light model → library; heavy model → server.**

---

## 4. Context window
The **context window** = the model's own limit on how much text (input + output) it can
"see" at once, measured in **tokens** (~word-pieces). It's a property of the MODEL, not the
engine. llama3.2 ≈ 128,000 tokens. Everything we send (system prompt + retrieved chunks +
question) must fit inside it — a reason we chunk and keep TOP_K small.

---

## 5. How is an LLM stored? (file format)
- NOT a language/code — a **binary file of numbers** (weights) grouped into **tensors** (grids).
- **Quantization** = storing each number with less precision to shrink size + speed it up:
  - float32 = 32 bits (~12 GB for 3B params)
  - float16 = 16 bits (~6 GB)
  - **int4 = 4 bits (~2 GB)** ← our llama3.2 is 4-bit quantized (why it fits a laptop)
- **File formats:**
  - **GGUF** → used by Ollama/llama.cpp (our LLM). One file: quantized weights + metadata.
  - **safetensors** → Hugging Face/PyTorch world (our embedding model).
- On disk: llama3.2 lives in `~/.ollama/models/blobs/` (a 1.9 GB file named by its sha256
  hash — "content-addressed storage," the hash also verifies integrity).

---

## 6. llama.cpp vs Ollama
- **llama.cpp** = the low-level **C/C++ inference engine** (by Georgi Gerganov, the "gg" in
  GGUF). Made LLMs run fast on normal hardware (CPU, Apple Silicon). Powerful but manual.
- **Ollama** = a **convenience wrapper built ON TOP of llama.cpp**. Adds: easy install,
  `ollama pull`, model registry, a warm background server, OpenAI-style API, prompt templates.
- Analogy: llama.cpp is the **engine**; Ollama is the **whole car** around it.
  (Also like **git vs GitHub**.)
- Upgrade to my mental model: an LLM is a file; the engine doesn't just *read* it, it
  **RUNS** it (loads → executes the math → serves). Read = passive; run = the real work.

---

## 7. Engines & wrappers landscape
**Inference engines (low-level):** llama.cpp (local/CPU/Mac), MLX (Apple-tuned), vLLM
(server scale, NVIDIA), TensorRT-LLM (NVIDIA), ExLlamaV2, TGI.
**Wrappers/apps (user-facing):** Ollama (CLI+server, us), LM Studio (GUI), Jan (GUI),
GPT4All (GUI), text-generation-webui, koboldcpp.
Most consumer wrappers sit on the **same** engine (llama.cpp); they differ mostly in
interface. Big fork = **local-scale** (llama.cpp/MLX) vs **server-scale** (vLLM/TGI).

---

## 8. Embedding models available in Ollama (optional alt path)
We use `all-MiniLM-L6-v2` via sentence-transformers, but Ollama CAN also serve embeddings:
`nomic-embed-text` (768 dims, popular default), `mxbai-embed-large` (1024), `all-minilm`
(384, same family as ours), `bge-m3` (1024, multilingual), `snowflake-arctic-embed`.
Call with `ollama.embed(...)` (not `.chat`). NOTE the differing dims (384/768/1024) → see #9.

---

## 9. THE MOST IMPORTANT PRINCIPLE (Q4)
**You MUST embed the query with the SAME embedding model used on the chunks.**
- Each embedding model invents its OWN coordinate system + may output a different number of
  dimensions (384 vs 768 vs 1024).
- Vectors from model A and model B are like maps at different scales/origins → measuring
  "distance" between them is meaningless → **retrieval returns garbage, and it fails SILENTLY**
  (no error, just wrong).
- That's why `EMBEDDING_MODEL` is defined identically in `ingest.py` and `query.py`.
  Change it → you must re-run ingest to rebuild the whole index.

---

## 10. Full object inventory
```
GENERATION SIDE
  LLM               - llama3.2               (a file: 2GB quantized weights, GGUF)
  Engine            - llama.cpp              (C/C++ program that runs the math)
  Wrapper           - Ollama                 (server on top of llama.cpp; localhost:11434)
RETRIEVAL SIDE
  Embedding model   - all-MiniLM-L6-v2       (file: ~80MB, safetensors; text → 384-num vector)
  Embedding runtime - sentence-transformers  (Python library; runs it in-process)
  Vector database   - ChromaDB               (stores vectors on disk in ./chroma_db; finds nearest)
OUR CODE (no framework)
  Chunker           - src/chunking.py        (splits docs into overlapping word-windows)
  Ingest script     - src/ingest.py          (Stages 2+3: embed → store)
  Query script      - src/query.py           (Stages 4-6: embed question → retrieve → prompt → generate)
DATA
  Corpus            - data/*.txt             (4 files, fictional "Nimbus Vault")
  Vector index      - chroma_db/             (generated by ingest.py)
ENVIRONMENT (plumbing, not RAG)
  Language          - Python 3.12 (venv via brew python@3.12)   |   Package mgr - pip
```

---

## 11. Full stage list
```
INDEXING (runs ONCE — ingest.py)
  Stage 1  Chunk     - split docs into overlapping word-windows   (chunking.py, no model)
  Stage 2  Embed     - each chunk → vector                        (all-MiniLM via sentence-transformers)
  Stage 3  Store     - save vectors + text                        (ChromaDB → ./chroma_db)
QUERYING (runs PER question — query.py)
  Stage 4  Retrieve  - embed question, find nearest chunks         (all-MiniLM + ChromaDB)
  Stage 5  Augment   - stuff retrieved chunks into a prompt        (query.py, no model) ← the "A" in RAG
  Stage 6  Generate  - LLM writes the answer                       (llama3.2 via Ollama)
```
Central efficiency idea: embed all documents ONCE (Stages 1–3), then answer unlimited
questions cheaply (Stages 4–6). Embedding model appears in BOTH Stage 2 and Stage 4.

---

## 12. Embeddings — WHY numbers capture meaning (the core magic)
An embedding model turns text → a fixed list of numbers (a **vector**). Ours (all-MiniLM)
outputs **384 numbers** per chunk, no matter how long the text is.

The key idea: those 384 numbers are **coordinates in a 384-dimensional space**. The model is
trained so that text with **similar meaning lands close together**, and unrelated text lands
far apart. "How much is the Business plan?" and "Business Plan — $18.99/month" end up as
neighbors even though they barely share any words.

- It's **meaning**, not keyword matching. "cost" and "$18.99" have no letters in common, but
  their vectors sit close.
- We can't picture 384-D, but "distance between two points" is the same formula in any number
  of dimensions — just more terms. 2-D/3-D intuition still carries.
- Where does the "meaning" come from? **Training.** The model saw billions of sentences and
  learned which words/phrases show up in similar contexts; that pattern is baked into its
  weights → similar contexts produce similar vectors.

Mental picture: a giant map where every possible sentence has an address, and things that
mean the same thing live in the same neighborhood. Embedding = looking up the address.
⚠️ This is exactly WHY #9 matters — model A and model B draw DIFFERENT maps, so an address
from A is meaningless on B's map.

---

## 13. ChromaDB — how "nearest" actually works
ChromaDB is the **vector database**: it stores every chunk's vector (+ the original text +
metadata), and given a query vector it hands back the closest ones.

- "Closest" = smallest **distance** between the query vector and each stored vector. Lower =
  more similar — that's the `distance=` number `query.py` prints for each retrieved chunk.
- Naive method: compare the query against EVERY stored vector (totally fine for our 10 chunks).
  Big vector DBs build an index (e.g. HNSW, a graph you hop through) so they don't scan
  millions one-by-one — but the idea is identical: find the nearest points.
- Chroma persists to disk in `./chroma_db/`, so we embed ONCE and reuse forever (the
  Stage 1–3 / 4–6 split from #11).

So retrieval = embed the question with the SAME model (#9) → ask Chroma for the TOP_K nearest
vectors → get back the human-readable chunks they belong to.

---

## 14. Running the whole thing end-to-end
**INDEXING (once):**
```
python src/ingest.py
```
Loads all-MiniLM → chunks `data/` (10 chunks for our 4 files) → embeds each into a 384-num
vector → writes them to `chroma_db/`. Ends with "Indexed 10 chunks".

**QUERYING (per question):**
```
python src/query.py
```
Ask: *"How much does the Business plan cost and how many devices does it support?"*
What happens, visibly:
1. `query.py` embeds the question with all-MiniLM (Stage 4).
2. Chroma returns the TOP_K=4 nearest chunks — printed with source + distance:
   ```
   [retrieved chunks]
     - 02_pricing.txt (chunk 1, distance=0.845)
     - 02_pricing.txt (chunk 0, distance=0.901)
     - 01_product_overview.txt (chunk 1, distance=1.417)
     - 01_product_overview.txt (chunk 2, distance=1.420)
   ```
   (The pricing file wins by a wide margin — distance ~0.85 vs ~1.42 for the
   product-overview chunks. Lower = closer in meaning.)
3. Those chunks get stuffed into the prompt under "Context:" (Stage 5).
4. llama3.2 via Ollama writes the answer, citing the source file (Stage 6):
   ```
   Assistant: The Business Plan is $18.99/month per user (minimum 3 users) and
   supports up to 10 linked devices per user. (02_pricing.txt)
   ```

**The PROOF it's really RAG:** $18.99 and "10 devices" are FICTIONAL Nimbus Vault facts. Base
llama3.2 can't know them — it got them right ONLY because retrieval put them in the prompt.
Blank the context (or point it at unrelated chunks) and it says it doesn't know, or guesses.

---

## 15. Where it all landed (recap)
- Python 3.12 venv + deps (torch, chromadb, sentence-transformers)
- Ollama running; `llama3.2` pulled (2 GB)
- `chroma_db/` built by `ingest.py` (Stages 1–3)
- `query.py` answers questions end-to-end (Stages 4–6), grounded + citing sources

Full loop, start to finish: **the problem RAG solves → the two models → Ollama/servers →
context window → file formats + quantization → embeddings → the vector DB → build the index →
ask a question → a correct, source-cited answer about facts the model never saw in training.**

**Ideas to push further (not built — next time I tinker):**
- Swap the LLM (`qwen2.5:7b`) or the embedder (`nomic-embed-text`) and compare answers —
  re-run `ingest.py` if you change the EMBEDDER (#9), otherwise the index is stale.
- Tune `chunk_size` / `overlap` (chunking.py) and `TOP_K` (query.py); watch retrieval shift.
- Print the actual prompt sent to the LLM to SEE the augmentation with your own eyes.
- Add a PDF→text step so `data/` can hold real documents, not just `.txt`.
