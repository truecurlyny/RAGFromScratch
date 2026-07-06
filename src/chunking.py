"""
Step 1 of the RAG pipeline: load raw text files and split them into
overlapping chunks.

No frameworks here on purpose — this is the part of RAG that is easiest
to treat as a black box, so it's written out by hand.

Chunking strategy: a sliding window over words (not characters), with
overlap so a sentence that falls on a chunk boundary still shows up
whole in at least one chunk.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str       # filename the chunk came from
    chunk_index: int   # position of this chunk within that file


def load_documents(data_dir: str | Path) -> list[tuple[str, str]]:
    """Read every .txt file in data_dir. Returns list of (filename, content)."""
    data_dir = Path(data_dir)
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append((path.name, text))
    return docs


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """
    Split `text` into overlapping chunks of `chunk_size` words, moving
    forward by (chunk_size - overlap) words each step.

    Example: chunk_size=120, overlap=30 means each new chunk repeats the
    last 30 words of the previous chunk, so context isn't lost at the seam.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_chunks(data_dir: str | Path, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Load all documents in data_dir and return a flat list of Chunk objects."""
    all_chunks: list[Chunk] = []
    for filename, text in load_documents(data_dir):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(Chunk(text=piece, source=filename, chunk_index=i))
    return all_chunks


if __name__ == "__main__":
    # Quick manual check: run `python src/chunking.py` from the project root
    chunks = build_chunks("data")
    print(f"Loaded {len(chunks)} chunks from data/")
    for c in chunks[:3]:
        print(f"\n--- {c.source} chunk {c.chunk_index} ---")
        print(c.text[:200], "...")
