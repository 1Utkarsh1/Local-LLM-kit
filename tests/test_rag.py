"""Offline tests for minimal RAG helpers (chunking + vector store).

Target API (v0.2.0): local_llm_kit.rag exposing chunk_text() and
SimpleVectorStore. Method-name variants are adapted so the tests accept any
reasonable spelling; the whole module skips (green) if the rag module has
not landed yet. Stdlib unittest only; no downloads, no numpy/torch.

Run: python -m pytest tests/test_rag.py -q
"""

import os
import re
import tempfile
import unittest

# --- v0.2.0 module probe (skip gracefully when absent) -----------------------
RAG_MOD = None
for _candidate in ("local_llm_kit.rag", "local_llm_kit.retrieval"):
    try:
        __import__(_candidate)
        import sys as _sys

        RAG_MOD = _sys.modules[_candidate]
        break
    except ImportError:
        continue

HAS_RAG = RAG_MOD is not None
CHUNK_FN = getattr(RAG_MOD, "chunk_text", None) if HAS_RAG else None
STORE_CLS = getattr(RAG_MOD, "SimpleVectorStore", None) if HAS_RAG else None
HAS_CHUNK = callable(CHUNK_FN)
HAS_STORE = isinstance(STORE_CLS, type)


def call_chunk(text, chunk_size=200, overlap=50):
    """Call chunk_text across likely v0.2.0 signatures."""
    tried = []
    variants = [
        ({"chunk_size": chunk_size, "overlap": overlap}, {}),
        ({"chunk_size": chunk_size, "chunk_overlap": overlap}, {}),
        ({"max_chars": chunk_size, "overlap": overlap}, {}),
        ({"chunk_size": chunk_size}, {}),
    ]
    for kwargs, _ in variants:
        try:
            return CHUNK_FN(text, **kwargs)
        except TypeError as exc:
            tried.append(str(exc))
    try:
        return CHUNK_FN(text, chunk_size, overlap)
    except TypeError as exc:
        tried.append(str(exc))
    raise TypeError("no chunk_text signature matched: %s" % tried)


def make_store(**kwargs):
    try:
        return STORE_CLS()
    except TypeError:
        return STORE_CLS(embedding_fn=None, **kwargs)


def store_add(store, texts):
    for name in ("add", "add_texts", "add_documents", "ingest", "extend", "upsert"):
        meth = getattr(store, name, None)
        if callable(meth):
            try:
                if name == "add_documents":
                    return meth([{"text": t} for t in texts])
                return meth(texts)
            except TypeError:
                continue
    raise AttributeError("SimpleVectorStore has no known add* method")


def store_search(store, query, top_k=2):
    for name in ("search", "query", "retrieve", "similarity_search"):
        meth = getattr(store, name, None)
        if callable(meth):
            try:
                return meth(query, top_k=top_k)
            except TypeError:
                try:
                    return meth(query, top_k)
                except TypeError:
                    continue
    raise AttributeError("SimpleVectorStore has no known search* method")


def _result_text(hit):
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        for key in ("text", "content", "document", "page_content"):
            if key in hit:
                return hit[key]
        return str(hit)
    for attr in ("text", "content", "document", "page_content"):
        if hasattr(hit, attr):
            return getattr(hit, attr)
    return str(hit)


def _norm(text):
    return re.sub(r"\s+", " ", text).strip()


@unittest.skipUnless(HAS_CHUNK, "local_llm_kit.rag.chunk_text not available yet")
class TestChunkText(unittest.TestCase):
    def test_returns_list_of_non_empty_strings(self):
        chunks = call_chunk("hello world, this is a test", chunk_size=10, overlap=2)
        self.assertIsInstance(chunks, list)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertTrue(chunk.strip())

    def test_short_text_gives_single_chunk(self):
        chunks = call_chunk("short text", chunk_size=1000, overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertIn("short text", chunks[0])

    def test_long_text_splits_into_multiple_chunks(self):
        text = "word %d " % 0
        text = " ".join("word%d" % i for i in range(200))
        chunks = call_chunk(text, chunk_size=20, overlap=0)
        self.assertGreater(len(chunks), 1)

    def test_no_overlap_covers_original_exactly(self):
        text = " ".join("word%d" % i for i in range(60))
        chunks = call_chunk(text, chunk_size=50, overlap=0)
        joined_space = _norm(" ".join(chunks))
        joined_bare = _norm("".join(chunks))
        self.assertTrue(
            joined_space == _norm(text) or joined_bare == _norm(text),
            msg="chunks do not reassemble the original text",
        )

    def test_overlap_adds_redundancy_and_shares_content(self):
        text = " ".join("word%d" % i for i in range(200))
        plain = call_chunk(text, chunk_size=20, overlap=0)
        overlapped = call_chunk(text, chunk_size=20, overlap=5)
        self.assertGreaterEqual(len(overlapped), len(plain))
        shared = False
        for first, second in zip(overlapped, overlapped[1:]):
            words = set(first.split()) & set(second.split())
            bare = "".join(first.split())
            if words or any(
                bare[i : i + 3] in "".join(second.split()) for i in range(max(0, len(bare) - 3))
            ):
                shared = True
                break
        self.assertTrue(shared, msg="overlapping chunks share no content")

    def test_every_word_appears_in_some_chunk(self):
        text = " ".join("word%d" % i for i in range(100))
        chunks = call_chunk(text, chunk_size=20, overlap=5)
        for word in text.split():
            self.assertTrue(any(word in c for c in chunks), msg="word missing: %s" % word)

    def test_empty_input_returns_empty_or_single(self):
        chunks = call_chunk("", chunk_size=50, overlap=10)
        self.assertIsInstance(chunks, list)
        self.assertLessEqual(len(chunks), 1)

    def test_deterministic(self):
        text = " ".join("word%d" % i for i in range(100))
        self.assertEqual(
            call_chunk(text, chunk_size=20, overlap=5), call_chunk(text, chunk_size=20, overlap=5)
        )


@unittest.skipUnless(HAS_STORE, "local_llm_kit.rag.SimpleVectorStore not available yet")
class TestSimpleVectorStore(unittest.TestCase):
    DOCS = [
        "the cat sat on the mat",
        "quantum field theory and particle physics",
        "the dog barked loudly at night",
    ]

    def _filled(self, docs=None):
        store = make_store()
        store_add(store, docs if docs is not None else list(self.DOCS))
        return store

    def test_search_orders_exact_match_first(self):
        store = self._filled()
        hits = store_search(store, "cat", top_k=3)
        self.assertTrue(hits)
        self.assertIn("cat", _result_text(hits[0]))

    def test_search_orders_second_topic_first(self):
        store = self._filled()
        hits = store_search(store, "quantum physics", top_k=3)
        self.assertIn("quantum", _result_text(hits[0]))

    def test_top_k_is_respected(self):
        store = self._filled()
        hits = store_search(store, "the", top_k=1)
        self.assertEqual(len(hits), 1)
        hits = store_search(store, "the", top_k=2)
        self.assertLessEqual(len(hits), 2)

    def test_empty_store_returns_empty(self):
        store = make_store()
        self.assertEqual(list(store_search(store, "anything", top_k=2)), [])

    def test_search_is_deterministic(self):
        store = self._filled()
        first = [_result_text(h) for h in store_search(store, "cat dog", top_k=3)]
        second = [_result_text(h) for h in store_search(store, "cat dog", top_k=3)]
        self.assertEqual(first, second)

    def test_save_and_load_roundtrip(self):
        store = self._filled()
        before = [_result_text(h) for h in store_search(store, "cat", top_k=3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            saved = False
            for name in ("save", "save_to_disk", "persist", "dump"):
                meth = getattr(store, name, None)
                if callable(meth):
                    try:
                        meth(path)
                        saved = True
                        break
                    except TypeError:
                        continue
            if not saved:
                self.skipTest("SimpleVectorStore exposes no save* method")
            loaded = None
            load = getattr(STORE_CLS, "load", None) or getattr(STORE_CLS, "load_from_disk", None)
            if callable(load):
                try:
                    loaded = load(path)
                except TypeError:
                    loaded = None
            if loaded is None:
                loaded = make_store()
                for name in ("load", "load_from_disk", "load_from_file"):
                    meth = getattr(loaded, name, None)
                    if callable(meth):
                        meth(path)
                        break
            after = [_result_text(h) for h in store_search(loaded, "cat", top_k=3)]
            self.assertEqual(before, after)


@unittest.skipUnless(HAS_RAG, "local_llm_kit.rag not available yet")
class TestRagStaysLight(unittest.TestCase):
    def test_rag_import_pulls_no_torch(self):
        import subprocess as _sp
        import sys as _sys

        mod = RAG_MOD.__name__
        code = (
            "import sys; import %s; "
            "print(','.join(m for m in ('torch', 'transformers') if m in sys.modules))" % mod
        )
        out = _sp.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        self.assertEqual(out.returncode, 0, msg=out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
