"""
Embeddings support for local-llm-kit.

This module provides a small, dependency-light embedding layer used by the
rest of the toolkit (notably the minimal RAG helpers in
:mod:`local_llm_kit.rag`).

Design goals
------------
* **Stdlib-only core** -- ``numpy`` is used when available (fast cosine
  path) but everything works without it via a pure-Python fallback.
* **Backend agnostic** -- works with any backend object that exposes one of
  ``embed()``, ``encode()`` (sentence-transformers style),
  ``embed_documents()`` / ``embed_query()``, OpenAI-style
  ``{"data": [{"embedding": ...}]}`` dicts / ``numpy`` arrays, or a plain
  callable. Backends without embedding support fail with a clear
  :class:`EmbeddingError` instead of an obscure ``AttributeError``.
* **Offline testable** -- :func:`hash_embed` (and ``backend="hash"``)
  provides deterministic hash-based fake vectors, so tests never need a
  model download.

Quick start (offline, no model needed)::

    from local_llm_kit.embeddings import hash_embed, EmbeddingClient

    vectors = hash_embed(["hello world", "goodbye world"], dim=64)

    client = EmbeddingClient(backend="hash", dim=64)
    vectors = client.embed_documents(["hello world", "goodbye world"])
    query_vec = client.embed_query("hello")
    score = client.similarity(query_vec, vectors[0])

With a real backend (any object exposing ``embed``)::

    from local_llm_kit.embeddings import EmbeddingClient

    client = EmbeddingClient(model_path="my-model", backend=my_backend)
    vectors = client.embed(["first text", "second text"])

Python 3.9 compatible.
"""

import hashlib
import importlib
import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore

HAS_NUMPY = np is not None

__all__ = [
    "HAS_NUMPY",
    "EmbeddingError",
    "cosine_similarity",
    "batch_cosine_similarity",
    "hash_embed",
    "embed",
    "EmbeddingClient",
]

logger = logging.getLogger(__name__)

Vector = List[float]
EmbedFn = Callable[..., Any]

_SENTINEL = object()

#: Backend names that resolve to the deterministic hash-based fake backend.
_HASH_BACKEND_NAMES = {"hash", "fake", "mock-hash", "test", "deterministic"}

#: Best-effort mapping of backend name -> (module, class) for lazy
#: construction. Missing/unavailable backends degrade to EmbeddingError.
_NAMED_BACKENDS = {
    "transformers": ("transformers", "TransformersBackend"),
    "llamacpp": ("llamacpp", "LlamaCppBackend"),
    "llama-cpp": ("llamacpp", "LlamaCppBackend"),
    "llama.cpp": ("llamacpp", "LlamaCppBackend"),
    "ollama": ("ollama", "OllamaBackend"),
    "openai": ("openai_compat", "OpenAICompatBackend"),
    "openai-compatible": ("openai_compat", "OpenAICompatBackend"),
    "openai_compat": ("openai_compat", "OpenAICompatBackend"),
    "server": ("openai_compat", "OpenAICompatBackend"),
    "mock": ("mock", "MockBackend"),
    "sentence-transformers": ("sentence_transformers", "SentenceTransformersBackend"),
    "sentence_transformers": ("sentence_transformers", "SentenceTransformersBackend"),
}


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced by the configured backend."""


# ---------------------------------------------------------------------------
# Numeric helpers (numpy-optional)
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    """Return True for real numeric scalars (bool excluded)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if HAS_NUMPY and np is not None:
        try:
            return isinstance(value, (np.integer, np.floating))
        except Exception:
            return False
    return False


def _extract_vector_or_none(value: Any) -> Optional[Vector]:
    """Coerce a single vector to ``List[float]`` or return None."""
    if HAS_NUMPY and np is not None and isinstance(value, np.ndarray):
        try:
            if value.ndim != 1 or value.shape[0] == 0:
                return None
            value = value.tolist()
        except Exception:
            return None
    if isinstance(value, dict):
        if "embedding" in value:
            return _extract_vector_or_none(value["embedding"])
        return None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        out = []
        for v in value:
            if not _is_number(v):
                return None
            out.append(float(v))
        return out
    return None


def _extract_vector(value: Any, name: str = "vector") -> Vector:
    vec = _extract_vector_or_none(value)
    if vec is None:
        raise ValueError("%s must be a non-empty sequence of numbers, got %r" % (name, value))
    return vec


def _extract_matrix(value: Any, n: int) -> Optional[List[Vector]]:
    """Coerce an embedding response to a list of ``n`` vectors or None."""
    if HAS_NUMPY and np is not None and isinstance(value, np.ndarray):
        try:
            if value.ndim == 2 and value.shape[0] == n:
                rows = []
                for row in value.tolist():
                    vec = _extract_vector_or_none(row)
                    if vec is None:
                        return None
                    rows.append(vec)
                return rows
            if value.ndim == 1 and n == 1:
                vec = _extract_vector_or_none(value.tolist())
                return [vec] if vec is not None else None
        except Exception:
            return None
        return None
    if isinstance(value, dict):
        if "embeddings" in value:
            return _extract_matrix(value["embeddings"], n)
        if "data" in value and isinstance(value["data"], (list, tuple)):
            if len(value["data"]) != n:
                return None
            rows = []
            for item in value["data"]:
                vec = _extract_vector_or_none(item)
                if vec is None:
                    return None
                rows.append(vec)
            return rows
        if "embedding" in value and n == 1:
            vec = _extract_vector_or_none(value["embedding"])
            return [vec] if vec is not None else None
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            return None
        rows = []
        for row in value:
            vec = _extract_vector_or_none(row)
            if vec is None:
                return None
            rows.append(vec)
        if rows and any(len(r) != len(rows[0]) for r in rows):
            return None
        return rows
    return None


def _l2_normalize(vec: Vector) -> Vector:
    norm = math.sqrt(math.fsum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors in ``[-1, 1]``.

    Uses a ``numpy`` fast path when available, otherwise pure Python.
    Zero vectors yield ``0.0`` instead of ``NaN``.

    Raises:
        ValueError: If the inputs are not non-empty numeric vectors of
            equal length.
    """
    va = _extract_vector(a, name="a")
    vb = _extract_vector(b, name="b")
    if len(va) != len(vb):
        raise ValueError(
            "cosine_similarity requires equal-length vectors, got %d and %d" % (len(va), len(vb))
        )
    if HAS_NUMPY and np is not None:
        try:
            qa = np.asarray(va, dtype=float)
            qb = np.asarray(vb, dtype=float)
            denom = float(np.linalg.norm(qa) * np.linalg.norm(qb))
            if denom == 0.0:
                return 0.0
            return float(np.dot(qa, qb) / denom)
        except Exception:
            logger.debug("numpy cosine path failed, using pure-Python fallback")
    dot = math.fsum(x * y for x, y in zip(va, vb))
    na = math.sqrt(math.fsum(x * x for x in va))
    nb = math.sqrt(math.fsum(x * x for x in vb))
    denom = na * nb
    if denom == 0.0:
        return 0.0
    return dot / denom


def batch_cosine_similarity(
    query: Sequence[float], matrix: Sequence[Sequence[float]]
) -> List[float]:
    """Cosine similarity between one query vector and each row of a matrix.

    Uses a vectorized ``numpy`` path when available, otherwise falls back
    to :func:`cosine_similarity` per row.
    """
    rows = list(matrix)
    if not rows:
        return []
    if HAS_NUMPY and np is not None:
        try:
            mat = np.asarray([list(r) for r in rows], dtype=float)
            q = np.asarray(list(query), dtype=float)
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
            logger.debug("numpy batch cosine path failed, using fallback")
    return [cosine_similarity(query, row) for row in rows]


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (offline tests / dev)
# ---------------------------------------------------------------------------


def hash_embed(
    texts: Union[str, Sequence[str]],
    dim: int = 128,
    normalize: bool = False,
    seed: int = 0,
) -> List[Vector]:
    """Deterministic hash-based fake embeddings.

    Each text maps to a fixed pseudo-random vector derived from its SHA-256
    hash. Useful for offline tests and development -- **not** semantic.

    Args:
        texts: A single string or a sequence of strings.
        dim: Embedding dimension (must be >= 1).
        normalize: If True, L2-normalize each vector to unit length.
        seed: Salt mixed into the hash so independent test spaces can be
            built (``seed`` is part of the digest).

    Returns:
        A list of ``dim``-dimensional float vectors, one per input text.
    """
    if isinstance(texts, str):
        items = [texts]
    else:
        items = list(texts)
    if dim < 1:
        raise ValueError("dim must be >= 1, got %r" % (dim,))
    vectors = []
    for text in items:
        if not isinstance(text, str):
            raise TypeError("hash_embed expects strings, got %r" % type(text).__name__)
        vec: Vector = []
        counter = 0
        while len(vec) < dim:
            digest = hashlib.sha256(("%d:%s:%d" % (seed, text, counter)).encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                if len(vec) >= dim:
                    break
                unit = int.from_bytes(digest[i : i + 4], "big") / 0xFFFFFFFF
                vec.append(unit * 2.0 - 1.0)
            counter += 1
        if normalize:
            vec = _l2_normalize(vec)
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# Backend interop
# ---------------------------------------------------------------------------


class _HashBackend:
    """Deterministic hash-based backend (tests / offline development)."""

    def __init__(self, dim: int = 128):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim

    def embed(self, texts: Union[str, Sequence[str]]) -> Union[Vector, List[Vector]]:
        if isinstance(texts, str):
            return hash_embed([texts], dim=self.dim)[0]
        return hash_embed(list(texts), dim=self.dim)

    def embed_query(self, text: str) -> Vector:
        return hash_embed([text], dim=self.dim)[0]

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return hash_embed(list(texts), dim=self.dim)


class _CallableBackend:
    """Adapter wrapping a plain ``embed_fn`` callable as a backend."""

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def embed(self, texts: Any) -> Any:
        return self._fn(texts)


def _has_embed_capability(obj: Any) -> bool:
    return any(
        callable(getattr(obj, name, None))
        for name in ("embed", "encode", "embed_documents", "embed_query")
    )


def _call_embed_fn(obj: Any, batch: List[str]) -> List[Vector]:
    """Call ``batch`` (list of str) through any supported backend protocol."""
    notes = []

    embed = getattr(obj, "embed", None)
    if callable(embed):
        try:
            result = embed(list(batch))
        except TypeError as exc:  # e.g. single-string-only signature
            notes.append("embed(list): %s" % exc)
            result = _SENTINEL
        except Exception as exc:
            raise EmbeddingError(
                "Backend %r failed to embed %d texts: %s" % (type(obj).__name__, len(batch), exc)
            ) from exc
        if result is not _SENTINEL:
            matrix = _extract_matrix(result, len(batch))
            if matrix is not None:
                return matrix
            notes.append("embed(list) returned unexpected shape")
        # Per-item fallback.
        try:
            return [_extract_vector(embed(t), name="embedding") for t in batch]
        except (TypeError, ValueError) as exc:
            notes.append("embed(text): %s" % exc)

    encode = getattr(obj, "encode", None)
    if callable(encode):
        try:
            result = encode(list(batch))
        except TypeError as exc:
            notes.append("encode(list): %s" % exc)
            result = _SENTINEL
        except Exception as exc:
            raise EmbeddingError(
                "Backend %r failed to encode %d texts: %s" % (type(obj).__name__, len(batch), exc)
            ) from exc
        if result is not _SENTINEL:
            matrix = _extract_matrix(result, len(batch))
            if matrix is not None:
                return matrix
            notes.append("encode(list) returned unexpected shape")
        try:
            return [_extract_vector(encode(t), name="encoding") for t in batch]
        except (TypeError, ValueError) as exc:
            notes.append("encode(text): %s" % exc)

    embed_documents = getattr(obj, "embed_documents", None)
    embed_query = getattr(obj, "embed_query", None)
    if callable(embed_documents) or callable(embed_query):
        try:
            if callable(embed_documents):
                matrix = _extract_matrix(embed_documents(list(batch)), len(batch))
                if matrix is not None:
                    return matrix
            if callable(embed_query):
                return [_extract_vector(embed_query(t), name="embedding") for t in batch]
        except (TypeError, ValueError) as exc:
            notes.append("embed_documents/embed_query: %s" % exc)
        except Exception as exc:
            raise EmbeddingError(
                "Backend %r failed to embed %d texts: %s" % (type(obj).__name__, len(batch), exc)
            ) from exc

    detail = "; ".join(notes) if notes else "no embed/encode method found"
    raise EmbeddingError(
        "Backend %r does not expose usable embeddings (%s). "
        "Provide a backend with embed(), encode(), or "
        "embed_documents()/embed_query(), or use backend='hash' for "
        "deterministic offline embeddings." % (type(obj).__name__, detail)
    )


def _construct_named_backend(
    name: str, model_path: Optional[str], backend_kwargs: Dict[str, Any]
) -> Any:
    """Best-effort lazy construction of a backend from its string name."""
    key = name.strip().lower()
    if key in _HASH_BACKEND_NAMES:
        return _HashBackend(dim=int(backend_kwargs.pop("dim", 128)))
    entry = _NAMED_BACKENDS.get(key)
    if entry is None:
        raise EmbeddingError(
            "Unknown embedding backend %r. Pass a backend instance exposing "
            "embed(), a plain embed callable, or one of: %s."
            % (name, sorted(list(_NAMED_BACKENDS) + list(_HASH_BACKEND_NAMES)))
        )
    module_name, class_name = entry
    prefixes = []
    if __package__:
        prefixes.append(__package__)
    prefixes.append("local_llm_kit")
    last_error: Optional[Exception] = None
    for prefix in prefixes:
        full_name = "%s.backends.%s" % (prefix, module_name)
        try:
            module = importlib.import_module(full_name)
        except ImportError as exc:
            last_error = exc
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            last_error = ImportError("module %r has no %r" % (full_name, class_name))
            continue
        try:
            instance = cls(model_path or "", **backend_kwargs)
        except Exception as exc:
            raise EmbeddingError(
                "Could not construct backend %r for embeddings: %s" % (name, exc)
            ) from exc
        if not _has_embed_capability(instance):
            raise EmbeddingError(
                "Backend %r does not support embeddings (no embed()/encode() " "method)." % name
            )
        return instance
    raise EmbeddingError(
        "Embedding backend %r is not available (%s). Install the matching "
        "optional dependency, pass a backend instance directly, or use "
        "backend='hash' for deterministic offline embeddings." % (name, last_error)
    )


def _resolve_backend(
    model_path: Optional[str],
    backend: Any,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    dim: Optional[int] = None,
) -> Any:
    """Normalize the ``backend`` argument to a usable backend object."""
    kwargs = dict(backend_kwargs or {})
    if dim is not None:
        kwargs.setdefault("dim", dim)
    if backend is None:
        raise EmbeddingError(
            "No embedding backend configured. Pass backend=<object with "
            "embed()>, backend=<embed callable>, backend='hash' for "
            "deterministic offline embeddings, or a backend name such as "
            "'transformers'/'ollama' (requires that backend to be installed "
            "and to implement embed())."
        )
    if isinstance(backend, str):
        return _construct_named_backend(backend, model_path, kwargs)
    if _has_embed_capability(backend):
        return backend
    if isinstance(backend, type):
        raise EmbeddingError("backend must be an instance, not the class %r." % backend.__name__)
    if callable(backend):
        return _CallableBackend(backend)
    raise EmbeddingError(
        "backend of type %r does not expose embed()/encode()/embed_query() "
        "and is not callable. Use backend='hash' for deterministic offline "
        "embeddings." % type(backend).__name__
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed(
    texts: Union[str, Sequence[str]],
    model_path: Optional[str] = None,
    backend: Any = None,
    batch_size: int = 32,
    normalize: bool = False,
    dim: Optional[int] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Vector]:
    """Embed one or more texts with the given model/backend.

    Args:
        texts: A single string or a sequence of strings.
        model_path: Model id or path handed to the backend (when the
            backend is given by name and needs constructing).
        backend: A backend instance exposing ``embed()`` (or ``encode()`` /
            ``embed_documents()``), a plain ``embed`` callable, or a
            backend name (``"hash"`` for deterministic offline vectors).
        batch_size: Number of texts sent to the backend per call.
        normalize: If True, L2-normalize every returned vector.
        dim: Dimension hint (used for ``backend="hash"`` and to validate
            real backend output when given).
        backend_kwargs: Extra keyword arguments for backend construction.

    Returns:
        A list of float vectors, one per input text (``[]`` for empty
        input).

    Raises:
        EmbeddingError: If no usable backend is configured or the backend
            fails / lacks embedding support.
        ValueError: For invalid ``batch_size``/``dim`` or inconsistent
            vector dimensions.
        TypeError: For non-string inputs.
    """
    client = EmbeddingClient(
        model_path=model_path,
        backend=backend,
        batch_size=batch_size,
        normalize=normalize,
        dim=dim,
        backend_kwargs=backend_kwargs,
    )
    if isinstance(texts, str):
        return client.embed([texts])
    return client.embed(list(texts))


class EmbeddingClient:
    """Stateful embedding client with batching and caching.

    Args:
        model_path: Model id or path (used when constructing a named
            backend; otherwise informational).
        backend: Backend instance exposing ``embed()`` (or ``encode()`` /
            ``embed_documents()``), a plain embed callable, or a backend
            name (``"hash"`` for deterministic offline vectors).
        batch_size: Number of texts sent to the backend per call.
        max_cache_size: Maximum cached vectors (FIFO eviction).
            ``None`` or ``<= 0`` disables caching.
        normalize: If True, L2-normalize every returned vector.
        backend_kwargs: Extra keyword arguments for backend construction.
        dim: Expected embedding dimension. Used for ``backend="hash"``
            and to validate real backend output.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: Any = None,
        batch_size: int = 32,
        max_cache_size: Optional[int] = 4096,
        normalize: bool = False,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        dim: Optional[int] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1, got %r" % (batch_size,))
        if dim is not None and dim < 1:
            raise ValueError("dim must be >= 1, got %r" % (dim,))
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_cache_size = max_cache_size
        self.normalize = normalize
        self.dim = dim
        self.backend = _resolve_backend(model_path, backend, backend_kwargs, dim)
        self._cache: "OrderedDict[str, Vector]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Number of currently cached vectors."""
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        return "%s(model_path=%r, backend=%s, dim=%r, cached=%d)" % (
            type(self).__name__,
            self.model_path,
            type(self.backend).__name__,
            self.dim,
            len(self),
        )

    # -- cache ----------------------------------------------------------
    def clear_cache(self) -> None:
        """Drop all cached vectors."""
        with self._lock:
            self._cache.clear()

    def _cache_get(self, text: str) -> Optional[Vector]:
        if self.max_cache_size is None or self.max_cache_size <= 0:
            return None
        with self._lock:
            vec = self._cache.get(text)
            if vec is None:
                return None
            self._cache.move_to_end(text)
            return list(vec)

    def _cache_put(self, text: str, vec: Vector) -> None:
        if self.max_cache_size is None or self.max_cache_size <= 0:
            return
        with self._lock:
            self._cache[text] = list(vec)
            self._cache.move_to_end(text)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

    def _check_dim(self, vec: Vector) -> None:
        if self.dim is None:
            self.dim = len(vec)
        elif len(vec) != self.dim:
            raise ValueError(
                "Inconsistent embedding dimension: expected %d, got %d" % (self.dim, len(vec))
            )

    # -- embedding ------------------------------------------------------
    def embed(self, texts: Union[str, Sequence[str]]) -> List[Vector]:
        """Embed texts, using the cache and batching backend calls.

        Always returns a list of vectors (even for a single string input;
        use :meth:`embed_query` for a single vector).
        """
        if isinstance(texts, str):
            items = [texts]
        else:
            items = list(texts)
        for t in items:
            if not isinstance(t, str):
                raise TypeError("embed expects strings, got %r" % type(t).__name__)
        if not items:
            return []

        cached: Dict[str, Vector] = {}
        to_fetch: List[str] = []
        for t in items:
            if t in cached:
                continue
            vec = self._cache_get(t)
            if vec is not None:
                cached[t] = vec
            elif t not in to_fetch:
                to_fetch.append(t)

        fetched: Dict[str, Vector] = {}
        for start in range(0, len(to_fetch), self.batch_size):
            batch = to_fetch[start : start + self.batch_size]
            logger.debug(
                "Embedding batch of %d texts with %s",
                len(batch),
                type(self.backend).__name__,
            )
            vectors = _call_embed_fn(self.backend, batch)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    "Backend returned %d vectors for %d texts" % (len(vectors), len(batch))
                )
            for text, vec in zip(batch, vectors):
                vec = _extract_vector(vec, name="embedding")
                if self.normalize:
                    vec = _l2_normalize(vec)
                self._check_dim(vec)
                fetched[text] = vec
                self._cache_put(text, vec)

        return [list(cached[t] if t in cached else fetched[t]) for t in items]

    def embed_documents(self, texts: Union[str, Sequence[str]]) -> List[Vector]:
        """Embed a batch of documents. Returns one vector per text."""
        if isinstance(texts, str):
            return self.embed([texts])
        return self.embed(list(texts))

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string. Returns one vector."""
        if not isinstance(text, str):
            raise TypeError("embed_query expects a string, got %r" % type(text).__name__)
        return self.embed([text])[0]

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity between two vectors (numpy fast path if present)."""
        return cosine_similarity(a, b)
