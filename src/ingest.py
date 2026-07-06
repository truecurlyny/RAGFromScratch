"""
Step 2 and 3 of the RAG pipeline: embed each chunk and store it in a
local vector database (ChromaDB, persisted to disk in ./chroma_db).

Run this once (and again any time data/ changes):
    python src/ingest.py

Embedding model: all-MiniLM-L6-v2 (sentence-transformers). It's small
(~80MB), runs fast on CPU, and needs no API key or internet access after
the first download.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import build_chunks

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "nimbus_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Chunking documents in '{DATA_DIR}/' ...")
    chunks = build_chunks(DATA_DIR)
    if not chunks:
        raise SystemExit(f"No .txt files found in {DATA_DIR}/ — nothing to ingest.")
    print(f"  -> {len(chunks)} chunks")

    print("Embedding chunks ...")
    # Stage 2: each chunk of text -> a fixed-length vector of 384 numbers that
    # encodes its meaning. .tolist() converts the numpy array to plain Python
    # lists, which is the format ChromaDB stores.
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to persistent Chroma store at '{DB_DIR}/' ...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh index each run: drop the old collection if it exists.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Stage 3: store each vector alongside its original text and metadata, so
    # retrieval can hand the human-readable chunk (not just the vector) back to
    # the LLM. `ids` must be unique per chunk; source + index guarantees that.
    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done. Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
