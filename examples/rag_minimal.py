"""Minimal RAG: chunk -> store -> retrieve -> chat. Runs fully offline.

Uses `local_llm_kit.rag` (chunk_text, SimpleVectorStore, build_rag_prompt)
in lexical (TF-IDF) mode — no model download, no embeddings needed, and
keyword queries rank sensibly and deterministically. Pass `embed_fn=...`
(a real embedding backend such as EmbeddingClient, Ollama, or an
OpenAI-compatible endpoint) for semantic retrieval instead.

Usage:
    python examples/rag_minimal.py
    python examples/rag_minimal.py --query "Where is the Eiffel Tower?"
    python examples/rag_minimal.py --backend ollama --model llama3.2:3b --query "..."
"""
import argparse

from local_llm_kit import LLM
from local_llm_kit.rag import SimpleVectorStore, build_rag_prompt, chunk_text


DOCS = [
    "Paris is the capital of France. The Eiffel Tower is located in Paris on the Champ de Mars.",
    "Local LLM Kit is a Python toolkit for running local language models with an OpenAI-like API.",
    "The Louvre museum in Paris houses the Mona Lisa.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal RAG demo (offline)")
    ap.add_argument("--model", "-m", default="echo", help="Model path/name")
    ap.add_argument("--backend", "-b", default="echo", help="Backend (default: echo)")
    ap.add_argument("--query", "-q", default="Where is the Eiffel Tower?",
                    help="Question to ask")
    ap.add_argument("--top-k", type=int, default=2, help="Chunks to retrieve")
    args = ap.parse_args()

    store = SimpleVectorStore()  # lexical TF-IDF mode: no embeddings needed
    for i, doc in enumerate(DOCS):
        chunks = chunk_text(doc, chunk_size=200, chunk_overlap=20)
        store.add(chunks, metadatas=[{"source": "doc-%d" % i}] * len(chunks))

    hits = store.search(args.query, k=args.top_k)
    print("Retrieved context:")
    for h in hits:
        print("- [%s] %s" % (h["metadata"].get("source", "?"), h["text"]))
    print()

    messages = build_rag_prompt(args.query, hits)
    llm = LLM(model_path=args.model, backend=args.backend)
    resp = llm.chat(messages=messages)
    print("Answer:", resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
