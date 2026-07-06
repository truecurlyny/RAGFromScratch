"""
Steps 4, 5, and 6 of the RAG pipeline:
    4. Retrieve  - embed the question and find the nearest chunks in ChromaDB
    5. Augment   - stuff those chunks into a prompt as "context"  (the "A" in RAG)
    6. Generate  - a local LLM (via Ollama) writes an answer grounded in that context

Prerequisites (see README.md):
    1. Ollama installed and running (`ollama serve`, or the Ollama app)
    2. A model pulled, e.g. `ollama pull llama3.2`
    3. `python src/ingest.py` already run at least once

Run:
    python src/query.py
"""

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "nimbus_docs"
# MUST match ingest.py: the question has to be embedded with the SAME model
# used on the chunks, or the vectors live in different coordinate systems and
# "nearest" becomes meaningless (retrieval silently returns garbage).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"   # change to whatever you `ollama pull`ed
TOP_K = 4                # how many chunks to retrieve and stuff into the prompt

SYSTEM_PROMPT = """You are a support assistant for the product Nimbus Vault.
Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say so plainly instead of guessing.
Cite which source file each fact came from in parentheses, e.g. (02_pricing.txt).
"""


def retrieve(collection, model, question: str, top_k: int = TOP_K):
    """Stage 4: turn the question into a vector and ask Chroma for the nearest chunks."""
    # encode() expects a list of texts and returns a list of vectors, so we wrap
    # the single question in a list and Chroma searches with that one vector.
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    # Chroma answers in batch form (one row per query vector); we only sent one,
    # so we take index [0] of each parallel list.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]        # lower distance = more similar
    return list(zip(docs, metas, distances))


def build_prompt(question: str, retrieved) -> str:
    """Stage 5 (Augment): fold the retrieved chunks into the prompt as context."""
    context_blocks = []
    for doc, meta, _dist in retrieved:
        context_blocks.append(f"[{meta['source']}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Chroma store at '{DB_DIR}/' ...")
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found. Run `python src/ingest.py` first."
        )

    print(f"Ready. Using Ollama model '{LLM_MODEL}'. Type a question (or 'quit').\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        retrieved = retrieve(collection, embed_model, question)

        print("\n[retrieved chunks]")
        for doc, meta, dist in retrieved:
            print(f"  - {meta['source']} (chunk {meta['chunk_index']}, distance={dist:.3f})")
        print()

        prompt = build_prompt(question, retrieved)

        # Stage 6 (Generate): the LLM writes the answer. The system prompt tells
        # it to answer ONLY from the context above, which is what makes this RAG
        # rather than the model free-associating from its training data.
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response["message"]["content"]
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
