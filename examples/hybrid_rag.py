#!/usr/bin/env python3
"""Example: Production-Grade Hybrid RAG with BM25, Dense Embeddings, and RRF.

Demonstrates:
1. High-level RAGPipeline document ingestion and chunking.
2. Hybrid Search combining Dense Cosine Similarity and Okapi BM25.
3. Reciprocal Rank Fusion (RRF) scoring for exact technical keyword matching.
4. Metadata filtering and JSON index persistence.

Run:
    python examples/hybrid_rag.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_llm_kit import LLM
from local_llm_kit.embeddings import hash_embed
from local_llm_kit.rag import RAGPipeline, SimpleVectorStore


def deterministic_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return hash_embed(list(texts), dim=32)


def main():
    print("=== Local LLM Kit: Hybrid RAG (BM25 + Dense RRF) ===\n")

    # 1. Initialize store with offline embedder
    store = SimpleVectorStore(embed_fn=deterministic_embed)

    # 2. Add domain documents with metadata
    docs = [
        "Kubernetes deployment manifests manage replica sets and pods in cloud clusters.",
        "PostgreSQL 16 index optimization: B-Tree vs GIN indexes for JSONB columns.",
        "Error ERR_CONN_REFUSED: Database connection dropped due to connection pool saturation.",
        "Python typing system: TypeVar, ParamSpec, and Protocol for static type checking.",
    ]
    metas = [
        {"category": "devops", "env": "prod"},
        {"category": "database", "env": "prod"},
        {"category": "troubleshooting", "env": "prod"},
        {"category": "python", "env": "dev"},
    ]

    store.add(docs, metadatas=metas, ids=["doc-k8s", "doc-pg", "doc-err", "doc-py"])
    print(f"Indexed {len(store)} documents.")

    # 3. Hybrid search (combines semantic embeddings and exact BM25 keyword matching)
    query = "ERR_CONN_REFUSED saturation"
    print(f"\nSearching with mode='hybrid' for: {query!r}...")
    hits = store.search(query, k=2, mode="hybrid")
    for hit in hits:
        print(f"  [{hit['id']}] (score: {hit['score']:.4f}) {hit['text']}")

    # 4. Metadata filtering
    print("\nFiltering by metadata: {'env': 'dev'}...")
    dev_hits = store.search("typing", k=2, filter={"env": "dev"})
    for hit in dev_hits:
        print(f"  [{hit['id']}] {hit['text']}")

    # 5. Persistence: save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        save_path = tmp.name

    store.save(save_path)
    print(f"\nSaved index to {save_path}")

    reloaded = SimpleVectorStore.load(save_path, embed_fn=deterministic_embed)
    print(f"Reloaded store from disk: {len(reloaded)} documents verified.")

    # 6. High-level RAGPipeline answering
    llm = LLM("llama3.2:3b", backend="echo")
    pipeline = RAGPipeline(llm=llm, store=reloaded)
    result = pipeline.query("What causes ERR_CONN_REFUSED?", k=1, mode="hybrid")
    print("\nRAGPipeline Query Output:")
    print(f"Answer: {result['answer']}")
    print(f"Sources cited: {len(result['sources'])}")


if __name__ == "__main__":
    main()
