"""
Minimal RAG (retrieval-augmented generation) helpers for local-llm-kit.

This module is intentionally small, dependency-light and fully offline:

* :func:`chunk_text` -- split documents into overlapping character chunks,
  preferring to break on separators (paragraph / line / word boundaries).
* :class:`SimpleVectorStore` -- tiny in-memory vector store with cosine
  search (``numpy`` fast path when available, pure-Python fallback),
  plus JSON save/load.
* :func:`retrieve` -- wire any ``embed`` function to a store.
* :func:`build_rag_prompt` -- build chat ``messages`` grounding the model
  in retrieved passages, with ``[1]``-style citations.

Everything runs on the standard library. ``numpy`` is used only if it is
already installed. For offline tests, use the deterministic hash-based
embed function from :mod:`local_llm_kit.embeddings`::

    from local_llm_kit.embeddings import hash_embed
    from local_llm_kit.rag import SimpleVectorStore, retrieve, build_rag_prompt

    def fake_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        return hash_embed(list(texts), dim=32)

    store = SimpleVectorStore(embed_fn=fake_embed)
    store.add(["the sky is blue", "grass is green"])
    hits = retrieve("what color is the sky?", store, fake_embed, k=1)
    messages = build_rag_prompt("what color is the sky?", hits)

Python 3.9 compatible.
"""

import json
import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore

HAS_NUMPY = np is not None

__all__ = [
    "HAS_NUMPY",
    "DEFAULT_SEPARATORS",
    "DEFAULT_SYSTEM_PROMPT",
    "chunk_text",
    "SimpleVectorStore",
    "retrieve",
    "build_rag_prompt",
    "bm25_scores",
    "reciprocal_rank_fusion",
    "RAGPipeline",
]

logger = logging.getLogger(__name__)

#: Default separators for :func:`chunk_text`, tried in priority order.
#: The empty string means "hard cut anywhere" and is always allowed last.
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Cite every factual claim with its source number like [1] or [2]. "
    "If the context does not contain the answer, say you don't know."
)

DocInput = Union[str, Dict[str, Any]]
EmbedFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Cosine similarity (local copies so rag works standalone)
# ---------------------------------------------------------------------------


def _cosine(a: List[float], b: List[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    denom = na * nb
    if denom == 0.0:
        return 0.0
    return dot / denom


def _cosine_matrix(query: List[float], matrix: List[List[float]]) -> List[float]:
    """Similarity of one query vector against each matrix row."""
    if not matrix:
        return []
    if HAS_NUMPY and np is not None:
        try:
            mat = np.asarray(matrix, dtype=float)
            q = np.asarray(query, dtype=float)
            if mat.ndim == 2 and q.ndim == 1 and mat.shape[1] == q.shape[0]:
                qn = float(np.linalg.norm(q))
                norms = np.linalg.norm(mat, axis=1)
                dots = mat.dot(q).tolist()
                out = []
                for d, n in zip(dots, norms.tolist()):
                    denom = float(n) * qn
                    out.append(float(d / denom) if denom else 0.0)
                return out
        except Exception:
            logger.debug("numpy RAG cosine path failed, using fallback")
    return [_cosine(query, row) for row in matrix]


def _as_vector(value: Any, name: str = "vector") -> List[float]:
    if HAS_NUMPY and np is not None and isinstance(value, np.ndarray):
        try:
            if value.ndim != 1:
                raise ValueError("expected 1-D vector")
            value = value.tolist()
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be a 1-D numeric vector" % name) from exc
    if isinstance(value, dict) and "embedding" in value:
        return _as_vector(value["embedding"], name=name)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a sequence of numbers" % name)
    if len(value) == 0:
        raise ValueError("%s must be non-empty" % name)
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if HAS_NUMPY and np is not None:
                try:
                    if isinstance(v, (np.integer, np.floating)):
                        out.append(float(v))
                        continue
                except Exception:
                    pass
            raise ValueError("%s must contain only numbers" % name)
        out.append(float(v))
    return out


def _as_matrix(value: Any, n: int) -> Optional[List[List[float]]]:
    if HAS_NUMPY and np is not None and isinstance(value, np.ndarray):
        try:
            if value.ndim == 2 and value.shape[0] == n:
                return [_as_vector(row) for row in value.tolist()]
            if value.ndim == 1 and n == 1:
                return [_as_vector(value.tolist())]
        except ValueError:
            return None
        return None
    if isinstance(value, dict):
        if "embeddings" in value:
            return _as_matrix(value["embeddings"], n)
        if "data" in value and isinstance(value["data"], (list, tuple)):
            if len(value["data"]) != n:
                return None
            try:
                return [_as_vector(item) for item in value["data"]]
            except ValueError:
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) == n:
        try:
            rows = [_as_vector(row) for row in value]
        except ValueError:
            return None
        if rows and any(len(r) != len(rows[0]) for r in rows):
            return None
        return rows
    return None


def _embed_texts(fn: EmbedFn, texts: List[str]) -> List[List[float]]:
    """Run ``texts`` through any supported embed-function shape."""
    if hasattr(fn, "embed_documents") and callable(fn.embed_documents):
        matrix = _as_matrix(fn.embed_documents(list(texts)), len(texts))
        if matrix is not None:
            return matrix
    if hasattr(fn, "embed") and callable(fn.embed):
        try:
            matrix = _as_matrix(fn.embed(list(texts)), len(texts))
        except TypeError:
            matrix = None
        if matrix is not None:
            return matrix
        return [_as_vector(fn.embed(t)) for t in texts]
    if hasattr(fn, "encode") and callable(fn.encode):
        try:
            matrix = _as_matrix(fn.encode(list(texts)), len(texts))
        except TypeError:
            matrix = None
        if matrix is not None:
            return matrix
        return [_as_vector(fn.encode(t)) for t in texts]
    if not callable(fn):
        raise TypeError(
            "embed_fn must be callable or expose embed()/encode()/"
            "embed_documents(), got %r" % type(fn).__name__
        )
    try:
        matrix = _as_matrix(fn(list(texts)), len(texts))
    except TypeError:
        matrix = None
    if matrix is not None:
        return matrix
    # Per-text fallback (also covers single-string embed fns).
    rows = []
    for t in texts:
        try:
            rows.append(_as_vector(fn(t)))
            continue
        except TypeError:
            pass
        if hasattr(fn, "embed_query") and callable(fn.embed_query):
            rows.append(_as_vector(fn.embed_query(t)))
        else:
            raise
    return rows


def _embed_one(fn: EmbedFn, text: str) -> List[float]:
    return _embed_texts(fn, [text])[0]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[Sequence[str]] = None,
    strip: bool = True,
) -> List[str]:
    """Split ``text`` into overlapping character chunks.

    A sliding window of ``chunk_size`` characters advances by
    ``chunk_size - chunk_overlap`` characters. When a window would cut
    mid-text, the cut is moved back to the last occurrence of the
    highest-priority separator (e.g. ``"\\n\\n"`` before ``"\\n"`` before
    ``" "``), so chunks prefer to break on paragraph/line/word
    boundaries. If no separator is found, the chunk is hard-cut.

    Args:
        text: Document text to split.
        chunk_size: Maximum characters per chunk (must be >= 1).
        chunk_overlap: Characters shared between consecutive chunks
            (must satisfy ``0 <= chunk_overlap < chunk_size``).
        separators: Separator strings in priority order. Defaults to
            ``["\\n\\n", "\\n", " ", ""]`` (``""`` = allow hard cuts).
        strip: Strip whitespace from chunks and drop empty ones.

    Returns:
        List of chunk strings (``[]`` for empty/whitespace-only input).

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: For invalid ``chunk_size``/``chunk_overlap``.
    """
    if not isinstance(text, str):
        raise TypeError("chunk_text expects a string, got %r" % type(text).__name__)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1, got %r" % (chunk_size,))
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0, got %r" % (chunk_overlap,))
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap (%d) must be smaller than chunk_size (%d)" % (chunk_overlap, chunk_size)
        )
    seps = list(DEFAULT_SEPARATORS if separators is None else separators)
    n = len(text)
    if n == 0:
        return []
    if not strip and n <= chunk_size:
        return [text]
    if strip and not text.strip():
        return []

    chunks: List[str] = []
    pos = 0
    min_keep = max(1, chunk_size // 4)
    while pos < n:
        window_end = pos + chunk_size
        if window_end >= n:
            piece = text[pos:n]
            if strip:
                piece = piece.strip()
            if piece or not strip:
                chunks.append(piece)
            break
        cut = window_end  # default: hard cut
        for sep in seps:
            if not sep:
                continue
            idx = text.rfind(sep, pos, window_end)
            if idx > pos and (idx - pos) >= min_keep:
                cut = min(idx + len(sep), window_end)
                if cut > pos:
                    break
                cut = window_end
        else:
            cut = window_end
        piece = text[pos:cut]
        if strip:
            piece = piece.strip()
        if piece or not strip:
            chunks.append(piece)
        # Advance with overlap, always making progress.
        nxt = cut - chunk_overlap
        pos = nxt if nxt > pos else cut
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def _lexical_tokens(text: str) -> List[str]:
    """Lowercase word tokens for the lexical (TF-IDF) fallback."""
    return re.findall(r"\w+", text.lower())


def _lexical_scores(query: str, docs: Sequence[str]) -> List[float]:
    """TF-IDF cosine-ish scores for a query against raw doc texts.

    Pure stdlib, deterministic, no embeddings required. Used when a
    :class:`SimpleVectorStore` has no ``embed_fn`` and no precomputed
    vectors, so keyword queries still rank sensibly offline.
    """
    qtokens = _lexical_tokens(query)
    if not qtokens or not docs:
        return [0.0] * len(docs)
    doc_tokens = [_lexical_tokens(d) for d in docs]
    n = len(docs)
    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}
    qtf: Dict[str, int] = {}
    for tok in qtokens:
        qtf[tok] = qtf.get(tok, 0) + 1
    qnorm = math.sqrt(math.fsum((qtf[t] * idf.get(t, 0.0)) ** 2 for t in qtf))
    scores = []
    for toks in doc_tokens:
        dtf: Dict[str, int] = {}
        for tok in toks:
            dtf[tok] = dtf.get(tok, 0) + 1
        dot = math.fsum(qtf[t] * idf.get(t, 0.0) * dtf.get(t, 0) * idf.get(t, 0.0) for t in qtf)
        dnorm = math.sqrt(math.fsum((dtf[t] * idf.get(t, 0.0)) ** 2 for t in dtf))
        denom = qnorm * dnorm
        scores.append(dot / denom if denom else 0.0)
    return scores


def bm25_scores(
    query: str,
    docs: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Calculate Okapi BM25 scores for a query across documents.

    Pure standard library, deterministic, zero external dependencies.

    Args:
        query: Query string.
        docs: List of document strings.
        k1: Term frequency saturation parameter. Default: 1.5.
        b: Document length normalization parameter. Default: 0.75.

    Returns:
        List of float BM25 scores matching docs length.
    """
    qtokens = _lexical_tokens(query)
    if not qtokens or not docs:
        return [0.0] * len(docs)
    doc_tokens = [_lexical_tokens(d) for d in docs]
    n = len(docs)
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_lens) / float(n) if n > 0 else 1.0

    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1

    scores = [0.0] * n
    for q in qtokens:
        if q not in df:
            continue
        nq = df[q]
        idf = math.log((n - nq + 0.5) / (nq + 0.5) + 1.0)
        for i, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf = toks.count(q)
            if tf <= 0:
                continue
            denom = tf + k1 * (1.0 - b + b * (doc_lens[i] / avgdl))
            scores[i] += idf * ((tf * (k1 + 1.0)) / denom)
    return scores


_bm25_scores = bm25_scores


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    rrf_k: int = 60,
) -> Dict[int, float]:
    """Combine multiple ranked lists of document indices using Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists: Sequence of ranked item lists (each item is an index or id).
        rrf_k: Smoothing constant. Default: 60.

    Returns:
        Dict mapping index/id to fused RRF score.
    """
    rrf_scores: Dict[int, float] = {}
    for rlist in ranked_lists:
        for rank, item in enumerate(rlist):
            score = 1.0 / (rrf_k + rank + 1)
            rrf_scores[item] = rrf_scores.get(item, 0.0) + score
    return rrf_scores


_reciprocal_rank_fusion = reciprocal_rank_fusion


class SimpleVectorStore:
    """Tiny in-memory vector store with cosine search and JSON persistence.

    Two modes (chosen automatically):
      * **Dense** (default when ``embed_fn`` or precomputed ``embeddings``
        are given): cosine similarity over embedding vectors, with a
        ``numpy`` fast path when available.
      * **Lexical** (no ``embed_fn``, no vectors): TF-IDF ranking over the
        raw texts. Zero dependencies, deterministic, and good enough for
        keyword search, tests, and small offline demos. For semantic
        retrieval, pass ``embed_fn`` (e.g. from
        :mod:`local_llm_kit.embeddings`).

    Args:
        embed_fn: Optional default embedding function used by :meth:`add`
            (when ``embeddings`` are not given) and :meth:`search` (when
            the query is a string). Any shape accepted: batched
            ``fn([str]) -> [[float]]``, per-text ``fn(str) -> [float]``,
            or an object exposing ``embed`` / ``encode`` /
            ``embed_documents`` / ``embed_query``.
        dim: Optional expected embedding dimension; inferred from the
            first added vectors when omitted (dense mode only).

    Example (offline)::

        def fake_embed(texts):
            return hash_embed(list(texts), dim=32)

        store = SimpleVectorStore(embed_fn=fake_embed)
        store.add(["the sky is blue"], metadatas=[{"source": "doc1"}])
        hits = store.search("what color is the sky?", k=1)
    """

    version = 1

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        dim: Optional[int] = None,
    ):
        if dim is not None and dim < 1:
            raise ValueError("dim must be >= 1, got %r" % (dim,))
        self.embed_fn = embed_fn
        self.dim = dim
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self._id_counter = 0

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return "%s(size=%d, dim=%r)" % (type(self).__name__, len(self), self.dim)

    def clear(self) -> None:
        """Remove all stored entries."""
        self.texts = []
        self.metadatas = []
        self.ids = []
        self.vectors = []
        self._id_counter = 0

    def _next_id(self) -> str:
        existing = set(self.ids)
        while True:
            candidate = "doc-%d" % self._id_counter
            self._id_counter += 1
            if candidate not in existing:
                return candidate

    def _check_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        checked = [_as_vector(v, name="embedding") for v in vectors]
        for vec in checked:
            if self.dim is None:
                self.dim = len(vec)
            elif len(vec) != self.dim:
                raise ValueError(
                    "Inconsistent embedding dimension: expected %d, got %d" % (self.dim, len(vec))
                )
        return checked

    def add(
        self,
        texts: Union[str, Sequence[str]],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Add texts (and optional metadata) to the store.

        Args:
            texts: A single string or a list of strings.
            embeddings: Precomputed vectors (one per text). When omitted,
                they are computed with the store's ``embed_fn`` — or, when
                the store has no ``embed_fn`` either, entries are stored
                for lexical (TF-IDF) search instead.
            metadatas: Optional list of metadata dicts (one per text).
            ids: Optional list of unique ids (generated as ``"doc-N"``
                when omitted).

        Returns:
            The list of ids for the added entries.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        for t in items:
            if not isinstance(t, str):
                raise TypeError("add expects strings, got %r" % type(t).__name__)

        if embeddings is None:
            if self.embed_fn is None:
                vectors = [[] for _ in items]  # lexical mode: no vectors
            else:
                vectors = self._check_vectors(_embed_texts(self.embed_fn, items))
        else:
            vectors = self._check_vectors(list(embeddings))
            if len(vectors) != len(items):
                raise ValueError("Got %d embeddings for %d texts" % (len(vectors), len(items)))

        if metadatas is None:
            metas: List[Dict[str, Any]] = [{} for _ in items]
        else:
            metas = [dict(m) if m else {} for m in metadatas]
            if len(metas) != len(items):
                raise ValueError("Got %d metadatas for %d texts" % (len(metas), len(items)))

        if ids is None:
            new_ids = [self._next_id() for _ in items]
            self.ids.extend([])  # no-op, keeps intent explicit
        else:
            new_ids = list(ids)
            if len(new_ids) != len(items):
                raise ValueError("Got %d ids for %d texts" % (len(new_ids), len(items)))
            existing = set(self.ids)
            for i in new_ids:
                if not isinstance(i, str) or not i:
                    raise ValueError("ids must be non-empty strings")
                if i in existing:
                    raise ValueError("Duplicate id %r" % (i,))
                existing.add(i)

        self.texts.extend(items)
        self.metadatas.extend(metas)
        self.ids.extend(new_ids)
        self.vectors.extend(vectors)
        return new_ids

    def _has_dense_vectors(self) -> bool:
        """True when every stored entry carries a non-empty vector."""
        return bool(self.vectors) and all(len(v) > 0 for v in self.vectors)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove entries matching the given ids.

        Args:
            ids: Sequence of entry ids to delete.

        Returns:
            Number of entries removed.
        """
        to_delete = set(ids)
        keep = [i for i, doc_id in enumerate(self.ids) if doc_id not in to_delete]
        if len(keep) == len(self.ids):
            return 0
        removed = len(self.ids) - len(keep)
        self.texts = [self.texts[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        self.ids = [self.ids[i] for i in keep]
        self.vectors = [self.vectors[i] for i in keep]
        return removed

    def _filter_candidates(
        self,
        filter_spec: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]],
    ) -> List[int]:
        if filter_spec is None:
            return list(range(len(self.texts)))
        candidates = []
        for i, meta in enumerate(self.metadatas):
            if isinstance(filter_spec, dict):
                match = all(meta.get(k) == v for k, v in filter_spec.items())
            elif callable(filter_spec):
                match = bool(filter_spec(meta))
            else:
                raise TypeError("filter must be a dict or a callable predicate")
            if match:
                candidates.append(i)
        return candidates

    def _ranked_hits(
        self,
        scores: Sequence[float],
        k: int,
        candidates: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        indices = list(range(len(scores))) if candidates is None else list(candidates)
        order = sorted(indices, key=lambda i: (scores[i], -i), reverse=True)
        hits = []
        for i in order[: min(k, len(order))]:
            hits.append(
                {
                    "index": i,
                    "id": self.ids[i],
                    "text": self.texts[i],
                    "metadata": dict(self.metadatas[i]),
                    "score": float(scores[i]),
                }
            )
        return hits

    def search(
        self,
        query: Union[str, Sequence[float]],
        k: int = 4,
        mode: str = "auto",
        filter: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the store for the ``k`` most similar entries.

        Args:
            query: A query string or a raw query vector (dense mode only).
            k: Number of hits to return (capped at the store size).
            mode: Search mode: ``"auto"`` (default, dense if vectors available, else lexical),
                ``"dense"`` (cosine similarity), ``"lexical"`` (BM25 keyword search),
                or ``"hybrid"`` (Reciprocal Rank Fusion over dense cosine and BM25 scores).
            filter: Optional metadata filter dict (exact matches) or callable predicate.

        Returns:
            Hits ordered by decreasing score. Each hit is a dict with
            ``index``, ``id``, ``text``, ``metadata`` and ``score`` keys.
            ``[]`` when the store is empty or no documents match filter.
        """
        if k < 1:
            raise ValueError("k must be >= 1, got %r" % (k,))
        if not self.texts:
            return []

        candidates = self._filter_candidates(filter)
        if not candidates:
            return []

        active_mode = mode.lower() if isinstance(mode, str) else "auto"
        if active_mode == "auto":
            if isinstance(query, str) and not self._has_dense_vectors():
                active_mode = "lexical"
            else:
                active_mode = "dense"

        if active_mode == "lexical":
            if not isinstance(query, str):
                raise ValueError("Lexical search requires a string query.")
            cand_texts = [self.texts[i] for i in candidates]
            raw_scores = _bm25_scores(query, cand_texts)
            all_scores = [0.0] * len(self.texts)
            for cand_idx, score in zip(candidates, raw_scores):
                all_scores[cand_idx] = score
            return self._ranked_hits(all_scores, k, candidates=candidates)

        if active_mode == "hybrid":
            if not isinstance(query, str):
                raise ValueError("Hybrid search requires a string query.")
            if self.embed_fn is None or not self._has_dense_vectors():
                return self.search(query, k=k, mode="lexical", filter=filter)

            query_vec = _embed_one(self.embed_fn, query)
            dense_scores = _cosine_matrix(query_vec, self.vectors)
            dense_order = sorted(candidates, key=lambda i: (dense_scores[i], -i), reverse=True)

            cand_texts = [self.texts[i] for i in candidates]
            bm25_raw = _bm25_scores(query, cand_texts)
            bm25_map = {cand_idx: score for cand_idx, score in zip(candidates, bm25_raw)}
            bm25_order = sorted(candidates, key=lambda i: (bm25_map.get(i, 0.0), -i), reverse=True)

            rrf_map = _reciprocal_rank_fusion([dense_order, bm25_order], rrf_k=60)
            all_scores = [0.0] * len(self.texts)
            for cand_idx in candidates:
                all_scores[cand_idx] = rrf_map.get(cand_idx, 0.0)
            return self._ranked_hits(all_scores, k, candidates=candidates)

        # Dense mode
        if isinstance(query, str):
            if self.embed_fn is None and not self._has_dense_vectors():
                cand_texts = [self.texts[i] for i in candidates]
                raw_scores = _lexical_scores(query, cand_texts)
                all_scores = [0.0] * len(self.texts)
                for cand_idx, score in zip(candidates, raw_scores):
                    all_scores[cand_idx] = score
                return self._ranked_hits(all_scores, k, candidates=candidates)
            if self.embed_fn is None:
                raise ValueError(
                    "String queries need an embed_fn: pass a query vector "
                    "or construct the store with embed_fn=...."
                )
            query_vec = _embed_one(self.embed_fn, query)
        else:
            query_vec = _as_vector(query, name="query")

        if not self._has_dense_vectors():
            raise ValueError(
                "Vector queries need stored embeddings: add entries with "
                "embeddings=... or construct the store with embed_fn=...."
            )
        if self.dim is not None and len(query_vec) != self.dim:
            raise ValueError(
                "Query dimension %d does not match store dimension %d" % (len(query_vec), self.dim)
            )
        scores = _cosine_matrix(query_vec, self.vectors)
        return self._ranked_hits(scores, k, candidates=candidates)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the store to a JSON-compatible dict."""
        return {
            "version": self.version,
            "dim": self.dim,
            "ids": list(self.ids),
            "texts": list(self.texts),
            "metadatas": [dict(m) for m in self.metadatas],
            "vectors": [list(v) for v in self.vectors],
        }

    def save(self, path: str) -> str:
        """Save the store to a JSON file. Returns the path."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str, embed_fn: Optional[EmbedFn] = None) -> "SimpleVectorStore":
        """Load a store previously saved with :meth:`save`.

        Raises:
            ValueError: If the file is not a JSON object, or any of the
                ``ids``/``texts``/``metadatas``/``vectors`` fields is null,
                not a list, or has a different length than the others, or a
                vector entry is not a numeric vector, or ``dim`` is neither
                an integer nor null.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid vector store file: %r" % (path,))

        def _collection(name: str) -> List[Any]:
            if name not in data:
                return []
            value = data[name]
            if value is None:
                raise ValueError("Corrupt vector store file %r: field %r is null" % (path, name))
            if not isinstance(value, list):
                raise ValueError(
                    "Corrupt vector store file %r: field %r must be a list, got %s"
                    % (path, name, type(value).__name__)
                )
            return value

        ids = _collection("ids")
        texts = _collection("texts")
        metadatas = _collection("metadatas")
        vectors = _collection("vectors")
        if not (len(ids) == len(texts) == len(metadatas) == len(vectors)):
            raise ValueError("Corrupt vector store file %r: mismatched field lengths" % (path,))
        dim = data.get("dim")
        if isinstance(dim, float) and dim.is_integer():
            dim = int(dim)  # __init__ allows int-valued floats; save() writes them back
        if isinstance(dim, bool) or (dim is not None and not isinstance(dim, int)):
            raise ValueError(
                "Corrupt vector store file %r: 'dim' must be an integer or null, got %s"
                % (path, type(dim).__name__)
            )
        store = cls(embed_fn=embed_fn, dim=dim)
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("Corrupt vector store file %r: ids must be non-empty strings" % path)
        if any(not isinstance(value, str) for value in texts):
            raise ValueError("Corrupt vector store file %r: texts must be strings" % path)
        if any(value is not None and not isinstance(value, dict) for value in metadatas):
            raise ValueError(
                "Corrupt vector store file %r: metadata must be objects or null" % path
            )
        store.ids = [str(i) for i in ids]
        store.texts = [str(t) for t in texts]
        store.metadatas = [dict(m) if isinstance(m, dict) else {} for m in metadatas]
        # Empty vectors mark lexical-mode entries (no embed_fn at save time).
        store.vectors = []
        for vec in vectors:
            if not isinstance(vec, list):
                raise ValueError(
                    "Corrupt vector store file %r: each vector must be a list, got %s"
                    % (path, type(vec).__name__)
                )
            store.vectors.append([] if not vec else store._check_vectors([vec])[0])
        # Keep generated ids collision-free.
        store._id_counter = len(store.ids)
        return store


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------


def retrieve(
    query: Union[str, Sequence[float]],
    store: SimpleVectorStore,
    embed_fn: Optional[EmbedFn] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    """Retrieve the ``k`` most relevant docs for ``query``.

    Args:
        query: Query string or precomputed query vector.
        store: The :class:`SimpleVectorStore` to search.
        embed_fn: Embedding function for string queries. Defaults to the
            store's own ``embed_fn``.
        k: Number of hits to return.

    Returns:
        Ranked hits (see :meth:`SimpleVectorStore.search`).
    """
    fn = embed_fn if embed_fn is not None else store.embed_fn
    if isinstance(query, str):
        if fn is None:
            # Lexical mode: TF-IDF over raw texts, no embeddings needed.
            return store.search(query, k=k)
        return store.search(_embed_one(fn, query), k=k)
    return store.search(query, k=k)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _doc_text(doc: DocInput) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "content", "page_content", "body"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
        # Fall back to a compact JSON rendering of the hit.
        try:
            return json.dumps(doc, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(doc)
    return str(doc)


def build_rag_prompt(
    query: str,
    docs: Optional[Sequence[DocInput]] = None,
    system_prompt: Optional[str] = None,
    context_header: str = "Context:",
    question_label: str = "Question:",
    instruction: str = (
        "Answer the question using the context above. "
        "Cite sources with their numbers like [1] or [2]."
    ),
    max_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build chat ``messages`` grounding the model in retrieved passages.

    Retrieved docs are rendered as numbered sources (``[1]``, ``[2]``,
    ...) so the model can cite them, and the system prompt instructs the
    model to cite every factual claim.

    Args:
        query: The user's question (non-empty string).
        docs: Retrieved passages as strings or hit dicts (as returned by
            :meth:`SimpleVectorStore.search` / :func:`retrieve`).
        system_prompt: Override for the default citation system prompt.
        context_header: Header printed above the numbered sources.
        question_label: Label printed before the question.
        instruction: Instruction appended after the question.
        max_chars: Optional cap on the joined context length; longer
            contexts are truncated with a ``...[truncated]`` marker.

    Returns:
        A ``messages`` list (``system`` + ``user``) ready for
        ``LLM.chat(messages)``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    items = list(docs or [])
    system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    if not items:
        user = (
            "%s %s\n\n(No context was retrieved. Answer from general "
            "knowledge if you can, otherwise say you don't know.)" % (question_label, query.strip())
        )
    else:
        parts = []
        for i, doc in enumerate(items, 1):
            parts.append("[%d] %s" % (i, _doc_text(doc).strip()))
        context = "\n\n".join(parts)
        if max_chars is not None and max_chars >= 0 and len(context) > max_chars:
            context = context[:max_chars].rstrip() + "\n...[truncated]"
        user = "%s\n%s\n\n%s %s\n\n%s" % (
            context_header,
            context,
            question_label,
            query.strip(),
            instruction,
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# High-Level RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """High-level Retrieval-Augmented Generation pipeline.

    Connects document chunking, indexing (vector/hybrid/BM25), retrieval,
    and LLM grounded generation with source citations.

    Args:
        llm: A :class:`local_llm_kit.LLM` instance.
        store: Optional :class:`SimpleVectorStore`. If omitted, a new store is created.
        embed_fn: Optional embedding function.
        system_prompt: System prompt instructing citation and grounding.
    """

    def __init__(
        self,
        llm: Any,
        store: Optional[SimpleVectorStore] = None,
        embed_fn: Optional[EmbedFn] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.store = store if store is not None else SimpleVectorStore(embed_fn=embed_fn)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def add_documents(
        self,
        texts: Union[str, Sequence[str]],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Chunk and index documents into the vector store.

        Args:
            texts: Single text document or list of documents.
            metadatas: Optional metadata dicts (one per document).
            chunk_size: Character size per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            List of generated chunk IDs added to the store.
        """
        raw_docs = [texts] if isinstance(texts, str) else list(texts)
        if not raw_docs:
            return []

        all_chunks: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for idx, doc in enumerate(raw_docs):
            doc_meta = (
                dict(metadatas[idx])
                if metadatas and idx < len(metadatas) and metadatas[idx]
                else {}
            )
            chunks = chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = dict(doc_meta)
                meta["doc_index"] = idx
                meta["chunk_index"] = c_idx
                all_metas.append(meta)

        return self.store.add(all_chunks, metadatas=all_metas)

    def query(
        self,
        question: str,
        k: int = 4,
        mode: str = "hybrid",
        filter: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]] = None,
        **chat_kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve relevant context and generate a grounded, cited answer.

        Args:
            question: User question string.
            k: Number of chunks to retrieve.
            mode: Search mode: ``"hybrid"`` (default), ``"dense"``, or ``"lexical"``.
            filter: Optional metadata filter dict or callable.
            **chat_kwargs: Additional kwargs passed to ``llm.chat()``.

        Returns:
            Dict containing ``answer``, ``sources`` (retrieved hits), and ``raw_response``.
        """
        hits = self.store.search(question, k=k, mode=mode, filter=filter)
        messages = build_rag_prompt(
            query=question,
            docs=hits,
            system_prompt=self.system_prompt,
        )
        resp = self.llm.chat(messages=messages, **chat_kwargs)
        choices = resp.get("choices", [])
        answer = choices[0].get("message", {}).get("content", "") if choices else ""

        return {
            "answer": answer,
            "sources": hits,
            "raw_response": resp,
        }
