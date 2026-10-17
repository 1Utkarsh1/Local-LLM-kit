"""Tests for BM25, Reciprocal Rank Fusion, Hybrid Search, and RAGPipeline."""

import unittest
from unittest.mock import MagicMock

from local_llm_kit.embeddings import hash_embed
from local_llm_kit.rag import (
    RAGPipeline,
    SimpleVectorStore,
    bm25_scores,
    reciprocal_rank_fusion,
)


def fake_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return hash_embed(list(texts), dim=32)


class TestHybridRAG(unittest.TestCase):
    def test_bm25_scores_computation(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "the dog chased the cat up the tree",
            "python programming with local models and rag",
        ]
        scores = bm25_scores("python rag", docs)
        self.assertEqual(len(scores), 3)
        # Doc 2 must score highest for "python rag"
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[2], scores[1])

    def test_reciprocal_rank_fusion(self):
        # Two ranking lists
        rank_a = [0, 1, 2]
        rank_b = [1, 0, 2]
        fused = reciprocal_rank_fusion([rank_a, rank_b], rrf_k=60)
        self.assertIn(0, fused)
        self.assertIn(1, fused)
        self.assertIn(2, fused)
        # Item 0 and 1 have same fused score, both higher than 2
        self.assertAlmostEqual(fused[0], fused[1])
        self.assertGreater(fused[0], fused[2])

    def test_simple_vector_store_delete(self):
        store = SimpleVectorStore(embed_fn=fake_embed)
        store.add(["doc a", "doc b", "doc c"], ids=["a", "b", "c"])
        self.assertEqual(len(store), 3)

        deleted = store.delete(["b"])
        self.assertEqual(deleted, 1)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.ids, ["a", "c"])
        self.assertEqual(store.texts, ["doc a", "doc c"])

    def test_simple_vector_store_metadata_filtering(self):
        store = SimpleVectorStore(embed_fn=fake_embed)
        store.add(
            ["report 2024", "report 2025", "internal wiki"],
            metadatas=[
                {"year": 2024, "dept": "finance"},
                {"year": 2025, "dept": "finance"},
                {"year": 2025, "dept": "engineering"},
            ],
            ids=["r1", "r2", "w1"],
        )

        # Filter by dict exact match
        hits = store.search("report", k=5, filter={"dept": "finance", "year": 2024})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "r1")

        # Filter by callable predicate
        hits_fn = store.search("report", k=5, filter=lambda m: m.get("year", 0) >= 2025)
        self.assertEqual(len(hits_fn), 2)
        ids_returned = {h["id"] for h in hits_fn}
        self.assertEqual(ids_returned, {"r2", "w1"})

    def test_simple_vector_store_hybrid_mode(self):
        store = SimpleVectorStore(embed_fn=fake_embed)
        store.add(
            [
                "PostgreSQL database query optimization and indexes",
                "Deep learning neural networks and transformer backends",
                "Kubernetes container deployment and helm charts",
            ],
            ids=["sql", "ml", "k8s"],
        )

        # Hybrid search combining semantic embedding and BM25 lexical token match
        hits = store.search("PostgreSQL indexes", k=2, mode="hybrid")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "sql")

    def test_rag_pipeline_end_to_end(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "According to [1], sky is blue."}}
            ]
        }

        pipeline = RAGPipeline(llm=mock_llm, embed_fn=fake_embed)
        chunk_ids = pipeline.add_documents(
            ["The sky on Mars looks butterscotch, but on Earth the sky is blue."],
            chunk_size=100,
            chunk_overlap=20,
        )
        self.assertGreater(len(chunk_ids), 0)

        result = pipeline.query("What color is the sky on Earth?", k=2, mode="hybrid")
        self.assertIn("sky is blue", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(mock_llm.chat.called)


if __name__ == "__main__":
    unittest.main()
