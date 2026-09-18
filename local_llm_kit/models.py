"""
Model download / discovery helpers for local-llm-kit.

Stdlib-only core (Python 3.9 compatible). Optional ``huggingface_hub``
is used lazily when installed; otherwise downloads fall back to
``urllib`` against ``https://huggingface.co``.

Covers:
  - ``is_gguf_path()`` / ``is_hf_repo_id()`` / ``is_url()``
  - ``download_model()`` / ``download_gguf()``
  - ``list_cached_models()``
  - ``resolve_model()`` -> ``(local_path_or_id, suggested_backend)``
  - ``RECOMMENDED_MODELS`` starter table (small CPU-friendly GGUFs +
    small HF instruct models)

Backend suggestion mirrors ``LLM._init_backend``: anything that looks
like a GGUF (``.gguf`` suffix, case-insensitive, or legacy ``ggml``)
suggests ``"llamacpp"``, everything else suggests ``"transformers"``.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "RECOMMENDED_MODELS",
    "is_gguf_path",
    "is_hf_repo_id",
    "is_url",
    "suggested_backend_for",
    "download_model",
    "download_gguf",
    "list_cached_models",
    "resolve_model",
    "list_recommended_models",
    "format_recommended_models",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Suffixes treated as llama.cpp models.
GGUF_SUFFIXES = (".gguf",)

#: Legacy llama.cpp suffixes (kept for backend auto-detect compat).
LEGACY_LLAMACPP_SUBSTRINGS = ("ggml",)

#: File types collected when scanning caches.
CACHE_FILE_PATTERNS = ("*.gguf", "*.ggml", "*.bin", "*.safetensors", "*.pt")

#: Default HF revision used for downloads.
DEFAULT_REVISION = "main"

#: ``owner/name`` pattern for Hugging Face repo ids.
_HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?" r"/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)

#: Small, CPU-friendly starter models. Filenames for GGUF rows are the
#: canonical ``Q4_K_M`` quant where the upstream repo publishes one; if a
#: filename has been renamed upstream, browse the repo page and pass the
#: exact filename to :func:`download_model`.
RECOMMENDED_MODELS: List[Dict[str, Any]] = [
    {
        "name": "Qwen2.5 0.5B Instruct (GGUF Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "backend": "llamacpp",
        "params": "0.5B",
        "approx_size": "~400 MB",
        "ram": "4 GB+",
        "use": "Smallest practical chat/test model for CPU.",
    },
    {
        "name": "Llama 3.2 1B Instruct (GGUF Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "backend": "llamacpp",
        "params": "1B",
        "approx_size": "~800 MB",
        "ram": "4-8 GB",
        "use": "Best quality-per-MB tiny Meta instruct model.",
    },
    {
        "name": "TinyLlama 1.1B Chat (GGUF Q4_K_M)",
        "repo_id": "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
        "backend": "llamacpp",
        "params": "1.1B",
        "approx_size": "~700 MB",
        "ram": "4-8 GB",
        "use": "Fast tiny chat baseline, very widely mirrored.",
    },
    {
        "name": "SmolLM2 135M Instruct (HF transformers)",
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "filename": None,
        "backend": "transformers",
        "params": "135M",
        "approx_size": "~270 MB",
        "ram": "2-4 GB",
        "use": "Smallest instruct model for offline transformers smoke tests.",
    },
    {
        "name": "SmolLM2 360M Instruct (HF transformers)",
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "filename": None,
        "backend": "transformers",
        "params": "360M",
        "approx_size": "~700 MB",
        "ram": "4 GB+",
        "use": "Better tiny instruct model, still CPU-runnable.",
    },
    {
        "name": "Qwen2.5 0.5B Instruct (HF transformers)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "filename": None,
        "backend": "transformers",
        "params": "0.5B",
        "approx_size": "~1 GB",
        "ram": "4-8 GB",
        "use": "Strong small instruct model; needs more RAM than GGUF.",
    },
    {
        "name": "TinyLlama 1.1B Chat (HF transformers)",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "filename": None,
        "backend": "transformers",
        "params": "1.1B",
        "approx_size": "~2.2 GB",
        "ram": "8 GB",
        "use": "Full-precision fallback when GGUF stack is unavailable.",
    },
]

# ---------------------------------------------------------------------------
# Small predicates
# ---------------------------------------------------------------------------


def is_url(spec: str) -> bool:
    """Return True if *spec* looks like an ``http(s)://`` URL."""
    if not isinstance(spec, str):
        return False
    s = spec.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def is_gguf_path(spec: str) -> bool:
    """Return True if *spec* looks like a GGUF model path/URL/name.

    Case-insensitive ``.gguf`` suffix check after stripping query
    strings (``?``) and fragments (``#``) so URLs such as
    ``https://.../model.gguf?download=true`` still match.
    """
    if not isinstance(spec, str):
        return False
    s = spec.strip().split("?", 1)[0].split("#", 1)[0].strip()
    return s.lower().endswith(GGUF_SUFFIXES)


def _looks_like_llamacpp(spec: str, filename: Optional[str] = None) -> bool:
    """Internal: GGUF suffix or legacy ``ggml`` substring (mirrors llm.py)."""
    hay = spec or ""
    if filename:
        hay += " " + filename
    low = hay.lower()
    if is_gguf_path(spec) or (filename and is_gguf_path(filename)):
        return True
    return any(sub in low for sub in LEGACY_LLAMACPP_SUBSTRINGS)


def is_hf_repo_id(spec: str) -> bool:
    """Return True if *spec* looks like a Hugging Face repo id.

    Accepts ``owner/name`` (and deeper ``owner/name/sub...`` namespaces).
    Returns False for URLs, local paths that exist, strings with
    whitespace, or strings ending in ``.gguf`` (use ``repo/file.gguf``
    or ``repo_id + filename`` for those).
    """
    if not isinstance(spec, str):
        return False
    s = spec.strip()
    if not s or "://" in s or any(c.isspace() for c in s):
        return False
    if s.lower().endswith(GGUF_SUFFIXES):
        return False
    if os.path.exists(os.path.expanduser(s)):
        return False
    if "/" not in s or s.startswith("/") or s.endswith("/"):
        return False
    return _HF_REPO_RE.match(s) is not None


def suggested_backend_for(spec: str, filename: Optional[str] = None) -> str:
    """Return ``"llamacpp"`` for GGUF/ggml specs, else ``"transformers"``.

    Mirrors ``LLM._init_backend`` auto-detection so
    :func:`resolve_model` agrees with :class:`LLM`.
    """
    if not isinstance(spec, str):
        spec = str(spec)
    return "llamacpp" if _looks_like_llamacpp(spec, filename) else "transformers"


# ---------------------------------------------------------------------------
# Cache locations
# ---------------------------------------------------------------------------


def default_cache_dir() -> str:
    """Return the default download cache directory.

    Respects ``LOCAL_LLM_KIT_CACHE`` env var, else
    ``~/.cache/local-llm-kit``.
    """
    env = os.environ.get("LOCAL_LLM_KIT_CACHE", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath(os.path.join(os.path.expanduser("~"), ".cache", "local-llm-kit"))


def _hf_hub_cache_dir() -> str:
    """Return the Hugging Face hub cache dir (env-aware)."""
    for var in ("HUGGINGFACE_HUB_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        val = os.environ.get(var, "").strip()
        if val:
            base = os.path.abspath(os.path.expanduser(val))
            # HF_HOME points at the parent of `hub/`.
            if var == "HF_HOME" and os.path.basename(base).lower() != "hub":
                return os.path.join(base, "hub")
            return base
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _dest_for_repo_file(repo_id: str, filename: str, cache_dir: Optional[str]) -> str:
    """Map ``(repo_id, filename)`` to a path inside our cache dir."""
    base = os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else default_cache_dir()
    # Nested layout keeps caches browsable: <cache>/<owner>/<name>/<file>
    rel = os.path.join(*repo_id.strip().split("/"), os.path.basename(filename))
    return os.path.join(base, rel)


# ---------------------------------------------------------------------------
# Download internals (stdlib urllib + lazy huggingface_hub)
# ---------------------------------------------------------------------------

#: Maximum single download size accepted by the urllib fallback (5 GB).
_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024 * 1024


def _download_url(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Download *url* to *dest_path* with stdlib urllib. Returns dest path."""
    # Local import keeps module import light.
    import urllib.request

    dest = os.path.abspath(dest_path)
    _assert_within_cache(dest, os.path.dirname(dest))
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm-kit/0.2.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
            total = resp.getheader("Content-Length")
            total_n = int(total) if total and total.isdigit() else 0
            if total_n > _MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    "Refusing to download %s: %d bytes exceeds the %d byte limit"
                    % (url, total_n, _MAX_DOWNLOAD_BYTES)
                )
            done = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done > _MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        "Download exceeded the %d byte limit; aborting %s"
                        % (_MAX_DOWNLOAD_BYTES, url)
                    )
                if total_n:
                    pct = done * 100 // total_n
                    logger.info("Downloading %s: %d%% (%d/%d bytes)", url, pct, done, total_n)
    except Exception as exc:
        # Remove partial file so a retry/cache scan never sees it as valid.
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        raise RuntimeError("Failed to download {}: {}".format(url, exc)) from exc
    logger.info("Saved %s -> %s", url, dest)
    return dest


def _assert_within_cache(path: str, cache_root: str) -> None:
    """Reject paths escaping the cache root (path-traversal mitigation)."""
    root = os.path.abspath(cache_root)
    target = os.path.abspath(path)
    if target != root and not target.startswith(root + os.sep):
        raise ValueError("Refusing to write outside the cache directory: %r" % path)


def _try_hf_hub_download(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str],
    revision: str = DEFAULT_REVISION,
) -> Optional[str]:
    """Try ``huggingface_hub.hf_hub_download``; return path or None.

    Returns None when ``huggingface_hub`` is not installed (caller falls
    back to urllib). Re-raises a clear RuntimeError when installed but
    the download itself fails.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    kwargs = {"repo_id": repo_id, "filename": filename, "revision": revision}
    if cache_dir:
        # NOTE: this is the HF hub cache root, files land under
        # models--<org>--<name>/snapshots/... upstream layout.
        kwargs["cache_dir"] = os.path.abspath(os.path.expanduser(cache_dir))
    try:
        return str(hf_hub_download(**kwargs))
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub failed to download '{}/{}' (revision={!r}): {}. "
            "Check the repo id, filename and your network/token.".format(
                repo_id, filename, revision, exc
            )
        ) from exc


def _download_hf_via_urllib(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str],
    revision: str = DEFAULT_REVISION,
) -> str:
    """Fallback downloader hitting ``huggingface.co/.../resolve/...``."""
    url = "https://huggingface.co/{}/resolve/{}/{}".format(
        repo_id.strip(), revision.strip() or DEFAULT_REVISION, filename.strip()
    )
    return _download_url(url, _dest_for_repo_file(repo_id, filename, cache_dir))


def _split_shorthand(spec: str) -> Tuple[str, Optional[str]]:
    """Split conveniences: ``repo:file`` and ``owner/name/file.gguf``.

    Returns ``(repo_or_path, filename_or_None)``. Plain repo ids, local
    paths and URLs pass through unchanged (filename None).
    """
    s = spec.strip()
    # "owner/name:file.gguf" shorthand (but not Windows "C:\\..." / URLs).
    if "://" not in s and len(s) > 2 and s[1] != ":":
        if ":" in s and s.count("/") == 1:
            head, tail = s.split(":", 1)
            if head and tail and is_hf_repo_id(head.strip()) and not os.path.exists(s):
                return head.strip(), tail.strip().lstrip("/")
    # "owner/name/file.gguf" (or deeper) reference.
    parts = s.split("/")
    if "://" not in s and len(parts) >= 3 and is_gguf_path(parts[-1]):
        repo = "/".join(parts[:-1])
        if is_hf_repo_id(repo):
            return repo, parts[-1].strip()
    return s, None


# ---------------------------------------------------------------------------
# Public download API
# ---------------------------------------------------------------------------


def download_model(
    repo_or_path: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = DEFAULT_REVISION,
) -> str:
    """Download (or locate) a model and return its local path.

    Args:
        repo_or_path: local file/dir (returned as-is), ``http(s)`` URL,
            HF repo id (``"owner/name"``, requires *filename*), or the
            shorthands ``"owner/name:file.gguf"`` /
            ``"owner/name/file.gguf"``. Only ``https`` URLs and
            ``huggingface.co`` hosts are accepted for direct URL downloads.
        filename: file inside the repo (e.g. ``"model-q4_k_m.gguf"``).
            Required for HF repo ids unless the shorthand above is used.
            Leading directories/``..`` are stripped (basename only) so a
            malicious filename can't escape the cache directory.
        cache_dir: cache root (default
            ``$LOCAL_LLM_KIT_CACHE`` or ``~/.cache/local-llm-kit``).
        revision: HF branch/tag/commit (default ``"main"``).

    Uses ``huggingface_hub.hf_hub_download`` when that package is
    installed, else falls back to a stdlib ``urllib`` download from
    ``https://huggingface.co/<repo>/resolve/<revision>/<file>``.
    Install the fast path with ``pip install huggingface_hub``.
    """
    if not isinstance(repo_or_path, str) or not repo_or_path.strip():
        raise ValueError("repo_or_path must be a non-empty string.")
    spec = repo_or_path.strip()
    if "\x00" in spec:
        raise ValueError("Invalid model spec (NUL byte).")

    # 1. Existing local path wins (file or HF snapshot dir).
    expanded = os.path.abspath(os.path.expanduser(spec))
    if os.path.exists(expanded):
        return expanded

    # 2. Shorthands: "repo:file", "repo/file.gguf".
    spec, short_file = _split_shorthand(spec)
    if short_file and filename is None:
        filename = short_file

    # 3. Direct URL download (https-only, host allowlist).
    if is_url(spec):
        from urllib.parse import urlparse

        parsed = urlparse(spec)
        if parsed.scheme != "https":
            raise ValueError("Direct model downloads require https, got %r." % parsed.scheme)
        if parsed.username or parsed.password:
            raise ValueError("URLs with embedded credentials are not accepted.")
        allowed_hosts = ("huggingface.co", "cdn-lfs.huggingface.co", "hf.co")
        if parsed.hostname not in allowed_hosts:
            raise ValueError(
                "Direct downloads are only accepted from %s, got host %r. "
                "Use download_model('owner/name', filename=...) for Hugging Face repos."
                % (", ".join(allowed_hosts), parsed.hostname)
            )
        name = filename or os.path.basename(spec.split("?", 1)[0].split("#", 1)[0]) or "model.gguf"
        base = os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else default_cache_dir()
        return _download_url(spec, os.path.join(base, "urls", os.path.basename(name)))

    # 4. Hugging Face repo id.
    if is_hf_repo_id(spec):
        if not filename or not filename.strip():
            raise ValueError(
                "Downloading repo '{}' requires a filename, e.g. "
                "download_model('{}', filename='model-q4_k_m.gguf'). "
                "Browse https://huggingface.co/{} to find the exact file, "
                "or use download_gguf() for GGUF repos.".format(spec, spec, spec)
            )
        # Basename-only: a crafted filename must not escape the cache dir.
        filename = os.path.basename(filename.strip().lstrip("/"))
        if not filename or filename in (".", ".."):
            raise ValueError("Invalid filename: %r." % filename)
        hit = _try_hf_hub_download(spec, filename, cache_dir, revision or DEFAULT_REVISION)
        if hit is not None:
            return os.path.abspath(hit)
        logger.info(
            "huggingface_hub not installed; falling back to urllib. "
            "For faster cached downloads: pip install huggingface_hub"
        )
        return os.path.abspath(
            _download_hf_via_urllib(spec, filename, cache_dir, revision or DEFAULT_REVISION)
        )

    raise ValueError(
        "Cannot interpret {!r} as a local path, URL, or Hugging Face repo id "
        "(expected 'owner/name'). If it is a repo file, pass "
        "filename='...' or use the 'owner/name:file.gguf' shorthand.".format(repo_or_path)
    )


def download_gguf(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    revision: str = DEFAULT_REVISION,
) -> str:
    """Download a single ``.gguf`` file from a Hugging Face repo.

    Thin validated wrapper over :func:`download_model` for the
    llama.cpp path (``LlamaCppBackend(model_path, ...)`` expects the
    returned file path).
    """
    if not isinstance(repo_id, str) or not is_hf_repo_id(repo_id.strip()):
        raise ValueError(
            "repo_id must be a Hugging Face repo id like 'owner/name', got {!r}.".format(repo_id)
        )
    if not isinstance(filename, str) or not is_gguf_path(filename.strip()):
        raise ValueError("filename must end in '.gguf', got {!r}.".format(filename))
    return download_model(repo_id.strip(), filename.strip(), cache_dir, revision)


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------


def _iter_model_files(root: str) -> List[str]:
    """Collect model files + HF snapshot dirs under *root* (best effort)."""
    found: List[str] = []
    base = Path(root).expanduser()
    if not base.is_dir():
        return found
    try:
        for pat in CACHE_FILE_PATTERNS:
            try:
                for p in base.rglob(pat):
                    try:
                        if p.is_file():
                            found.append(os.path.abspath(str(p)))
                    except OSError:
                        continue
            except (OSError, PermissionError):
                continue
        # HF snapshot dirs (usable directly as Transformers model_path).
        try:
            for cfg in base.rglob("config.json"):
                try:
                    if cfg.is_file() and "snapshots" in cfg.parts:
                        found.append(os.path.abspath(str(cfg.parent)))
                except OSError:
                    continue
        except (OSError, PermissionError):
            pass
    except (OSError, PermissionError):
        return found
    return found


def list_cached_models(
    cache_dir: Optional[str] = None,
    search_cwd: bool = True,
    search_hf_cache: bool = True,
) -> List[str]:
    """List locally cached/downloaded models.

    Scans, in order:
      1. the Hugging Face hub cache (``~/.cache/huggingface/hub`` or
         ``$HF_HOME``/``$HUGGINGFACE_HUB_CACHE``),
      2. the local-llm-kit cache (``$LOCAL_LLM_KIT_CACHE`` or
         ``~/.cache/local-llm-kit``),
      3. ``*.gguf`` files in the current working directory.

    Returns a sorted, de-duplicated list of absolute paths (GGUF/model
    files, or HF snapshot directories containing ``config.json``).
    Missing directories are skipped silently.
    """
    seen: Dict[str, None] = {}

    def _add(paths: List[str]) -> None:
        for p in paths:
            try:
                key = os.path.abspath(p)
            except (OSError, ValueError):
                continue
            if key not in seen:
                seen[key] = None

    if search_hf_cache:
        _add(_iter_model_files(_hf_hub_cache_dir()))
    _add(
        _iter_model_files(
            os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else default_cache_dir()
        )
    )
    if search_cwd:
        try:
            for p in Path.cwd().glob("*.gguf"):
                try:
                    if p.is_file():
                        _add([str(p)])
                except OSError:
                    continue
        except OSError:
            pass
    return sorted(seen.keys())


def _find_cached_repo_file(
    repo_id: str,
    filename: Optional[str],
    cache_dir: Optional[str],
) -> Optional[str]:
    """Search caches for a previously downloaded repo file (or snapshot)."""
    roots: List[str] = []
    roots.append(
        os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else default_cache_dir()
    )
    roots.append(_hf_hub_cache_dir())
    want = os.path.basename(filename) if filename else None
    # Our nested layout first: <cache>/<owner>/<name>/<file>.
    if want:
        direct = os.path.join(roots[0], *repo_id.split("/"), want)
        if os.path.isfile(direct):
            return os.path.abspath(direct)
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        try:
            if want:
                for p in base.rglob(want):
                    try:
                        if p.is_file():
                            return os.path.abspath(str(p))
                    except OSError:
                        continue
            else:
                # Any snapshot dir for this repo (HF hub layout is
                # models--<org>--<name>/snapshots/<rev>/...).
                marker = "models--" + "--".join(repo_id.split("/"))
                cand = base / marker
                if cand.is_dir():
                    snaps = cand / "snapshots"
                    if snaps.is_dir():
                        revs = sorted([d for d in snaps.iterdir() if d.is_dir()])
                        if revs:
                            return os.path.abspath(str(revs[0]))
        except (OSError, PermissionError):
            continue
    return None


# ---------------------------------------------------------------------------
# Resolve: spec -> (path_or_id, backend)
# ---------------------------------------------------------------------------


def resolve_model(
    spec: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    download: bool = False,
    revision: str = DEFAULT_REVISION,
) -> Tuple[str, str]:
    """Resolve *spec* to ``(local_path_or_id, suggested_backend)``.

    - Existing local file/dir -> ``(abspath, backend)``.
    - ``http(s)`` URL (no download) -> ``(url, backend)``; with
      ``download=True`` the URL is fetched into the cache first.
    - HF repo id (or ``repo:file`` / ``repo/file.gguf`` shorthand):
      returns the cached file/snapshot when found; with
      ``download=True`` downloads it; otherwise returns the repo id
      (or ``repo_id/filename``) untouched so
      ``TransformersBackend``/``from_pretrained`` can still load it
      directly from the Hub.
    - Suggested backend is ``"llamacpp"`` for ``.gguf`` (or legacy
      ``ggml``) specs, else ``"transformers"`` — matching
      ``LLM._init_backend``.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("spec must be a non-empty string.")
    raw = spec.strip()

    expanded = os.path.abspath(os.path.expanduser(raw))
    if os.path.exists(expanded):
        return expanded, suggested_backend_for(raw, filename)

    norm, short_file = _split_shorthand(raw)
    if short_file and filename is None:
        filename = short_file

    if is_url(norm):
        if download:
            path = download_model(norm, filename, cache_dir, revision)
            return path, suggested_backend_for(path, filename)
        return norm, suggested_backend_for(norm, filename)

    if is_hf_repo_id(norm):
        hit = _find_cached_repo_file(norm, filename, cache_dir)
        if hit:
            return hit, suggested_backend_for(hit, filename)
        if download:
            if not filename:
                raise ValueError("download=True for repo '{}' needs filename='...'.".format(norm))
            path = download_model(norm, filename, cache_dir, revision)
            return path, suggested_backend_for(path, filename)
        # Not cached: hand back something the backends accept directly.
        model_id = norm if not filename else norm + "/" + filename.strip().lstrip("/")
        return model_id, suggested_backend_for(model_id, filename)

    # Unknown but local-looking string: return as-is with a best guess so
    # error messages from the backend (not here) stay authoritative.
    return raw, suggested_backend_for(raw, filename)


# ---------------------------------------------------------------------------
# Starter-model table
# ---------------------------------------------------------------------------


def list_recommended_models() -> List[Dict[str, Any]]:
    """Return a copy of the recommended starter-models table."""
    return [dict(entry) for entry in RECOMMENDED_MODELS]


def format_recommended_models() -> str:
    """Render the starter-models table as Markdown (for CLI/docs)."""
    lines = [
        "| Model | Repo | File | Backend | Size |",
        "|---|---|---|---|---|",
    ]
    for m in RECOMMENDED_MODELS:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                m.get("name", "?"),
                m.get("repo_id", "-"),
                m.get("filename") or "(any/snapshot)",
                m.get("backend", "?"),
                m.get("approx_size", "?"),
            )
        )
    return "\n".join(lines) + "\n"
