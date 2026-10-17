"""
Find the local LLM servers that are actually running, and report what they can do.

The most common failure in local-AI code is not a bad model call, it is not
knowing *what* is listening *where*: which port a desktop app chose, whether the
server finished loading its model, whether the loaded model can call tools, and
whether a model is genuinely on-device or quietly routed to a remote host.

This module answers those questions with the standard library only. It probes
every known local endpoint concurrently, reads each server's own capability
metadata where that server exposes it, and never raises: a dead port is a
result, not an exception.

Example:
    from local_llm_kit import discover

    report = discover()
    for server in report.servers:
        print(server.base_url, server.provider, server.latency_ms)
    first = report.first_live()

Probed endpoints come from :data:`local_llm_kit.discovery.ENDPOINTS`; pass your
own with ``discover(extra_urls=["http://localhost:9000/v1"])``.

Nothing here performs inference, downloads weights, or sends credentials.
Probes are unauthenticated metadata reads, including POST /api/show for Ollama.
"""

import concurrent.futures
import ipaddress
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ENDPOINTS",
    "Endpoint",
    "ModelRecord",
    "ServerReport",
    "DiscoveryReport",
    "normalize_base_url",
    "probe_endpoint",
    "discover",
    "find_local_server",
]

#: Sent with probes so servers and local logs can identify the caller.
_USER_AGENT = "local-llm-kit-discovery/0.3"

#: Default per-request timeout. Loopback probes either answer immediately or not.
DEFAULT_TIMEOUT = 0.75

#: Cap on how many models get per-model enrichment (a `/api/show` round trip each).
_MAX_ENRICHED_MODELS = 12


@dataclass(frozen=True)
class Endpoint:
    """A place a local LLM server is known to listen."""

    provider: str
    base_url: str
    #: Extra base URLs that are common for this provider (probed too).
    note: str = ""


#: Known local endpoints, most commonly used first. The order is the probe
#: order, so users who run the popular tools get answers fastest.
#:
#: Ports are the upstream defaults as of October 2026, checked against each
#: project's own source or documentation:
#:
#: * ``llama-server`` appears twice on purpose. llama.cpp changed its default
#:   listen port from 8080 to 9931 in October 2026
#:   (ggml-org/llama.cpp#30159, merged 2026-10-09), while the published Docker
#:   image still binds 8080 (same PR). MLX-LM also defaults to 8080, which is
#:   exactly why the provider is identified from the response and not the port.
#: * ``apfel`` deliberately shares Ollama's port 11434, so both entries collapse
#:   into one probe and the payload decides which server is really there.
#: * Docker Model Runner serves OpenAI compatibility under an ``/engines``
#:   prefix rather than at the root (Docker model-runner API reference).
#: * ``fm-serve`` (Apple's macOS 27 system tool) has been measured on
#:   127.0.0.1:1976 by third-party testing. It is probed because a refused
#:   loopback connection costs nothing, but Apple publishes no primary
#:   documentation for it and its ``/v1/models`` shape is unverified.
#: * RamaLama takes the first free port in 8080-8180, so it is only caught by
#:   the 8080 probe when it happens to land there.
ENDPOINTS: Tuple[Endpoint, ...] = (
    Endpoint("ollama", "http://127.0.0.1:11434/v1"),
    Endpoint("lmstudio", "http://127.0.0.1:1234/v1"),
    Endpoint("llama-server", "http://127.0.0.1:9931/v1"),
    Endpoint("llama-server", "http://127.0.0.1:8080/v1"),
    Endpoint("vllm", "http://127.0.0.1:8000/v1"),
    Endpoint("localai", "http://127.0.0.1:8080/v1"),
    Endpoint("sglang", "http://127.0.0.1:30000/v1"),
    Endpoint("jan", "http://127.0.0.1:1337/v1"),
    Endpoint("gpt4all", "http://127.0.0.1:4891/v1"),
    Endpoint("mlx-lm", "http://127.0.0.1:8080/v1"),
    Endpoint("koboldcpp", "http://127.0.0.1:5001/v1"),
    Endpoint("text-generation-webui", "http://127.0.0.1:5000/v1"),
    Endpoint("docker-model-runner", "http://127.0.0.1:12434/engines/v1"),
    Endpoint("apfel", "http://127.0.0.1:11434/v1"),
    Endpoint("fm-serve", "http://127.0.0.1:1976/v1"),
    Endpoint("llamafile", "http://127.0.0.1:8080/v1"),
)

#: Provider names whose model listing carries capability metadata we can read.
#: See the module docstring: capabilities come from each server's own API.
_OLLAMA_ROOT_SUFFIXES = ("/v1", "/api")

#: Substrings in a model id or host that indicate off-device routing.
_REMOTE_MARKERS = ("cloud",)


def normalize_base_url(url: str) -> str:
    """Return a canonical ``scheme://host[:port]/v1`` base URL.

    Accepts what users actually paste: a bare host/port, a URL with or without
    ``/v1``, or a full ``.../v1/chat/completions`` endpoint. Query strings and
    fragments are dropped.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("base_url must be a non-empty string.")
    text = url.strip()
    if "://" not in text:
        text = "http://" + text
    parts = urllib.parse.urlsplit(text)
    if not parts.hostname:
        raise ValueError("base_url must include a host: %r" % (url,))
    if parts.scheme not in ("http", "https"):
        raise ValueError("base_url must use http or https.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Discovery probes do not accept credentials in URLs.")
    netloc = "[%s]" % parts.hostname if ":" in parts.hostname else parts.hostname
    if parts.port is not None:
        netloc = "%s:%d" % (netloc, parts.port)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/embeddings", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/v1"):
        path = path + "/v1" if path else "/v1"
    return "%s://%s%s" % (parts.scheme or "http", netloc, path)


def is_loopback_url(url: str) -> bool:
    """Whether an HTTP endpoint explicitly addresses this machine."""
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False
        host = parts.hostname or ""
        if host.lower() == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _server_root(base_url: str) -> str:
    """Strip a trailing ``/v1`` (or ``/api``) so native API paths can be built."""
    root = base_url.rstrip("/")
    for suffix in _OLLAMA_ROOT_SUFFIXES:
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


@dataclass
class ModelRecord:
    """One model as reported by a server, with whatever capabilities it exposes."""

    id: str
    #: Capability names the server reported (empty when the server does not say).
    capabilities: List[str] = field(default_factory=list)
    #: Context length the server reported for this model, if it reported one.
    context_length: Optional[int] = None
    #: True when the server routes this model off-device (for example Ollama's
    #: ``:cloud`` models, which run on ollama.com rather than on this machine).
    remote: bool = False
    #: Server-native metadata explicitly identifies local execution.
    local_verified: bool = False
    remote_host: Optional[str] = None
    #: Raw provider payload, kept for ``--json`` output and debugging.
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "capabilities": list(self.capabilities),
            "context_length": self.context_length,
            "remote": self.remote,
            "local_verified": self.local_verified,
            "remote_host": self.remote_host,
        }

    @property
    def supports_tools(self) -> bool:
        return any("tool" in c.lower() for c in self.capabilities)

    @property
    def supports_vision(self) -> bool:
        return any(c.lower() in ("vision", "image") for c in self.capabilities)

    @property
    def supports_embeddings(self) -> bool:
        return any("embed" in c.lower() for c in self.capabilities)

    @property
    def supports_thinking(self) -> bool:
        return any(c.lower() in ("thinking", "reasoning") for c in self.capabilities)


@dataclass
class ServerReport:
    """Result of probing one endpoint. A dead endpoint is a normal result."""

    base_url: str
    provider: str
    live: bool = False
    status: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    models: List[ModelRecord] = field(default_factory=list)
    #: Server-wide context window when the server reports one (llama.cpp,
    #: apfel and LM Studio do; Ollama reports it per model instead).
    context_window: Optional[int] = None
    #: Server-wide vision support when the server reports it.
    vision: Optional[bool] = None
    #: True when the server says it is still loading its model.
    loading: bool = False
    #: Human-readable notes: detection evidence, degradations, next steps.
    notes: List[str] = field(default_factory=list)

    @property
    def remote_models(self) -> List[ModelRecord]:
        """Models on this server that would leave the machine if called."""
        return [m for m in self.models if m.remote]

    @property
    def local_models(self) -> List[ModelRecord]:
        if not is_loopback_url(self.base_url):
            return []
        return [m for m in self.models if m.local_verified and not m.remote]

    @property
    def has_local_models(self) -> bool:
        """True when this live server offers at least one on-device model.

        A server may offer both kinds — Ollama happily lists downloaded models
        next to ``:cloud`` ones — so this is the useful "can I run offline?"
        question, whereas :attr:`all_local` is the stricter "is everything
        here on-device?" question.
        """
        return self.live and bool(self.local_models)

    @property
    def all_local(self) -> bool:
        """True when the server is live and exposes no off-device models."""
        return self.live and bool(self.models) and len(self.local_models) == len(self.models)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "provider": self.provider,
            "live": self.live,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "loading": self.loading,
            "context_window": self.context_window,
            "vision": self.vision,
            "supports_tools": self.supports_tools,
            "has_local_models": self.has_local_models,
            "all_local": self.all_local,
            "notes": list(self.notes),
            "models": [m.to_dict() for m in self.models],
        }

    @property
    def supports_tools(self) -> bool:
        return any(m.supports_tools for m in self.models)

    @property
    def supports_embeddings(self) -> bool:
        return any(m.supports_embeddings for m in self.models)


@dataclass
class DiscoveryReport:
    """Everything :func:`discover` found, including what it could not reach."""

    servers: List[ServerReport] = field(default_factory=list)
    elapsed_ms: float = 0.0
    probed: int = 0

    @property
    def live(self) -> List[ServerReport]:
        return [s for s in self.servers if s.live]

    @property
    def offline(self) -> List[ServerReport]:
        return [s for s in self.servers if not s.live]

    def first_live(self, require_local: bool = False) -> Optional[ServerReport]:
        """Return the first live server, preferring ones with on-device models.

        With *require_local*, only servers offering at least one on-device model
        qualify.
        """
        if require_local:
            for server in self.live:
                if server.has_local_models:
                    return server
            return None
        for server in self.live:
            if server.has_local_models:
                return server
        return self.live[0] if self.live else None

    def on_device(self) -> List[ServerReport]:
        """Live servers that offer at least one on-device model."""
        return [s for s in self.live if s.has_local_models]

    def local_only(self) -> List[ServerReport]:
        """Live servers whose model list is entirely on-device."""
        return [s for s in self.live if s.all_local]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": "local_llm_kit.discovery",
            "probed": self.probed,
            "elapsed_ms": self.elapsed_ms,
            "live_count": len(self.live),
            "servers": [s.to_dict() for s in self.servers],
        }


# ---------------------------------------------------------------------------
# HTTP (stdlib only)
# ---------------------------------------------------------------------------


def _get_json(url: str, timeout: float) -> Tuple[Optional[int], Any, Optional[str]]:
    """GET *url* and decode JSON. Returns ``(status, payload, error)``.

    Any failure (DNS, refused connection, timeout, non-JSON body such as the
    plain-text page Ollama serves on ``/``) is reported through *error* rather
    than raised, because a probe that cannot answer is a result we want to keep.
    """
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            try:
                return response.status, json.loads(body.decode("utf-8", errors="replace")), None
            except ValueError as exc:
                return response.status, None, "invalid JSON: %s" % exc
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # pragma: no cover - body already consumed
            detail = ""
        return exc.code, None, "HTTP %s%s" % (exc.code, (": " + detail) if detail else "")
    except urllib.error.URLError as exc:
        return None, None, str(getattr(exc, "reason", exc))
    except (TimeoutError, OSError) as exc:
        return None, None, str(exc)


def _post_json(
    url: str, payload: Dict[str, Any], timeout: float
) -> Tuple[Optional[int], Any, Optional[str]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            try:
                return (
                    response.status,
                    json.loads(response.read().decode("utf-8", errors="replace")),
                    None,
                )
            except ValueError as exc:
                return response.status, None, "invalid JSON: %s" % exc
    except urllib.error.HTTPError as exc:
        return exc.code, None, "HTTP %s" % exc.code
    except urllib.error.URLError as exc:
        return None, None, str(getattr(exc, "reason", exc))
    except (TimeoutError, OSError) as exc:
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Model listing + provider fingerprinting
# ---------------------------------------------------------------------------


def _looks_remote(model_id: str, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Decide whether a listed model would run off this machine.

    Ollama is explicit: ``/api/tags`` marks cloud-routed models with
    ``remote_host``/``remote_model``. As a fallback, an ``:cloud`` tag on an
    otherwise-local server is treated as off-device too, because that is the
    conventional naming for provider-routed models.
    """
    host = payload.get("remote_host")
    if isinstance(host, str) and host:
        return True, host
    if payload.get("remote_model"):
        return True, host or "remote"
    if model_id.lower().endswith(":cloud"):
        return True, host
    return False, None


def _records_from_openai_models(payload: Any) -> List[ModelRecord]:
    """Parse an OpenAI ``/v1/models`` payload into :class:`ModelRecord` objects.

    Also reads the extra fields servers add on top of the OpenAI shape:
    ``capabilities`` and ``context_window`` (apfel), ``meta.n_ctx_train``
    (llama.cpp), and ``details.context_length`` (Ollama's OpenAI-compat view).
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records: List[ModelRecord] = []
    for item in data:
        if isinstance(item, str):
            records.append(ModelRecord(id=item))
            continue
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name") or item.get("model")
        if not isinstance(model_id, str) or not model_id:
            continue
        raw_caps = item.get("capabilities")
        capabilities = [str(c) for c in raw_caps] if isinstance(raw_caps, list) else []
        context_length = item.get("context_window")
        if not isinstance(context_length, int):
            meta = item.get("meta")
            context_length = meta.get("n_ctx_train") if isinstance(meta, dict) else None
        if not isinstance(context_length, int):
            details = item.get("details")
            context_length = details.get("context_length") if isinstance(details, dict) else None
        remote, remote_host = _looks_remote(model_id, item)
        records.append(
            ModelRecord(
                id=model_id,
                capabilities=capabilities,
                context_length=context_length if isinstance(context_length, int) else None,
                remote=remote,
                remote_host=remote_host,
                raw=item,
            )
        )
    return records


def _identify_provider(base_url: str, models_payload: Any, hint: str) -> str:
    """Name the server behind *base_url* from its own response shape.

    Users point this at arbitrary ports, so the provider is inferred from
    evidence in the payload rather than assumed from the port number alone.
    Returns ``""`` when the payload carries no distinguishing field, which tells
    the caller to try the provider-specific marker endpoints instead.
    """
    data = models_payload.get("data") if isinstance(models_payload, dict) else None
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    if isinstance(first.get("details"), dict) and "family" in first.get("details", {}):
        return "ollama"
    if isinstance(first.get("meta"), dict) and "n_ctx_train" in first.get("meta", {}):
        return "llama-server"
    if "capabilities" in first and "context_window" in first:
        return "apfel"
    owned_by = str(first.get("owned_by", "")).lower()
    if "llamacpp" in owned_by or "llama.cpp" in owned_by:
        return "llama-server"
    if "lmstudio" in owned_by or "lm studio" in owned_by:
        return "lmstudio"
    if "vllm" in owned_by:
        return "vllm"
    if owned_by in ("library", "ollama"):
        return "ollama"
    return ""


def _refine_provider(root: str, timeout: float) -> str:
    """Identify a server by trying each provider's own marker endpoint.

    Used when the OpenAI-compatible model list carries no distinguishing field —
    LM Studio's ``/v1/models`` reports a generic owner, for example. Only
    read-only metadata endpoints are touched.
    """
    status, payload, _error = _get_json(root + "/api/tags", timeout)
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("models"), list):
        return "ollama"
    status, payload, _error = _get_json(root + "/props", timeout)
    if status == 200 and isinstance(payload, dict) and "default_generation_settings" in payload:
        return "llama-server"
    for path in ("/api/v1/models", "/api/v0/models"):
        status, payload, _error = _get_json(root + path, timeout)
        if status == 200 and isinstance(payload, dict) and isinstance(payload.get("models"), list):
            return "lmstudio"
    return ""


# ---------------------------------------------------------------------------
# Capability enrichment (each server's own metadata endpoint)
# ---------------------------------------------------------------------------


def _enrich_ollama(report: ServerReport, root: str, timeout: float) -> None:
    """Read Ollama's native ``/api/tags`` and ``/api/show`` for real capabilities.

    ``/api/tags`` is the only place Ollama exposes ``remote_host`` (whether a
    model runs on ollama.com) and per-model ``capabilities``; ``/api/show`` is
    where the true ``context_length`` lives — the OpenAI-compatible
    ``/v1/models`` view reports neither.
    """
    status, payload, error = _get_json(root + "/api/tags", timeout)
    if status != 200 or not isinstance(payload, dict):
        report.notes.append("ollama /api/tags unavailable (%s)" % (error or status))
        return
    models = payload.get("models")
    if not isinstance(models, list):
        return
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name:
            catalog[name] = item
    records: List[ModelRecord] = []
    for model_id, item in catalog.items():
        capabilities = [str(c) for c in item.get("capabilities") or [] if isinstance(c, str)]
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        raw_context = details.get("context_length")
        remote, remote_host = _looks_remote(model_id, item)
        records.append(
            ModelRecord(
                id=model_id,
                capabilities=capabilities,
                context_length=raw_context if isinstance(raw_context, int) else None,
                remote=remote,
                local_verified=not remote,
                remote_host=remote_host,
                raw=item,
            )
        )
    if records:
        report.models = records
    report.notes.append("capabilities read from ollama /api/tags")


def _enrich_llama_server(report: ServerReport, root: str, timeout: float) -> None:
    """Read llama.cpp ``/props`` for the live context size and modality flags."""
    status, payload, error = _get_json(root + "/props", timeout)
    if status != 200 or not isinstance(payload, dict):
        report.notes.append("llama.cpp /props unavailable (%s)" % (error or status))
        return
    settings = payload.get("default_generation_settings")
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    if isinstance(n_ctx, int) and n_ctx > 0:
        report.context_window = n_ctx
        for model in report.models:
            if model.context_length is None:
                model.context_length = n_ctx
    modalities = payload.get("modalities")
    if isinstance(modalities, dict) and "vision" in modalities:
        report.vision = bool(modalities["vision"])
    if payload.get("is_sleeping"):
        report.notes.append("server is sleeping (idle sleep is enabled)")
    model_path = payload.get("model_path")
    if isinstance(model_path, str) and model_path:
        for model in report.models:
            model.local_verified = not model.remote
        report.notes.append("model file: %s" % model_path)


def _enrich_lmstudio(report: ServerReport, root: str, timeout: float) -> None:
    """Read LM Studio's native model API for vision/tool/context capabilities.

    LM Studio exposes ``capabilities.vision`` and
    ``capabilities.trained_for_tool_use`` plus ``max_context_length`` on its
    native endpoint only; its OpenAI-compatible ``/v1/models`` lists bare ids.
    Both the current ``/api/v1/models`` and the older ``/api/v0/models`` are
    tried.
    """
    for path in ("/api/v1/models", "/api/v0/models"):
        status, payload, _error = _get_json(root + path, timeout)
        if status != 200 or not isinstance(payload, dict):
            continue
        entries = payload.get("models")
        if not isinstance(entries, list):
            entries = payload.get("data")
        if not isinstance(entries, list):
            continue
        records: List[ModelRecord] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            model_id = item.get("key") or item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            capabilities: List[str] = []
            caps = item.get("capabilities")
            if isinstance(caps, dict):
                if caps.get("vision"):
                    capabilities.append("vision")
                if caps.get("trained_for_tool_use"):
                    capabilities.append("tools")
                reasoning = caps.get("reasoning")
                if isinstance(reasoning, dict):
                    capabilities.append("reasoning")
            model_type = item.get("type")
            if isinstance(model_type, str) and model_type:
                capabilities.append(model_type)
            context_length = item.get("max_context_length")
            if not isinstance(context_length, int):
                instances = item.get("loaded_instances")
                if isinstance(instances, list) and instances and isinstance(instances[0], dict):
                    config = instances[0].get("config")
                    if isinstance(config, dict):
                        context_length = config.get("context_length")
            records.append(
                ModelRecord(
                    id=model_id,
                    capabilities=capabilities,
                    context_length=context_length if isinstance(context_length, int) else None,
                    raw=item,
                )
            )
        if records:
            for model in records:
                model.local_verified = bool(model.raw.get("loaded_instances"))
            report.models = records
            report.notes.append("capabilities read from lmstudio %s" % path)
            return
    report.notes.append("lmstudio native model API unavailable")


def _enrich_per_model_context(report: ServerReport, root: str, timeout: float) -> None:
    """Fill in context length per model via Ollama ``/api/show`` when missing."""
    missing = [m for m in report.models if m.context_length is None][:_MAX_ENRICHED_MODELS]
    for model in missing:
        status, payload, _error = _post_json(root + "/api/show", {"model": model.id}, timeout)
        if status != 200 or not isinstance(payload, dict):
            continue
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, list) and capabilities:
            model.capabilities = [str(c) for c in capabilities]
        info = payload.get("model_info")
        if isinstance(info, dict):
            for key, value in info.items():
                if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                    model.context_length = value
                    break


#: ``provider -> enricher``. Enrichers read that server's own metadata API and
#: mutate the report in place; a failure only appends a note.
_ENRICHERS = {
    "ollama": _enrich_ollama,
    "llama-server": _enrich_llama_server,
    "lmstudio": _enrich_lmstudio,
    "apfel": lambda report, root, timeout: None,
}


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def probe_endpoint(
    base_url: str,
    provider: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    enrich: bool = True,
) -> ServerReport:
    """Probe one endpoint and return its report (never raises for I/O errors).

    Args:
        base_url: Anything :func:`normalize_base_url` accepts.
        provider: Expected provider name, used as a hint in ``notes``. The real
            provider is inferred from the server's response.
        timeout: Per-request timeout in seconds.
        enrich: Read provider-specific capability metadata (an extra request or
            two). Disable for the fastest possible liveness check.

    Raises:
        ValueError: only for a malformed *base_url* argument.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number of seconds.")
    normalized = normalize_base_url(base_url)
    root = _server_root(normalized)
    report = ServerReport(base_url=normalized, provider=provider)

    started = time.perf_counter()
    status, payload, error = _get_json(normalized + "/models", timeout)
    report.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    report.status = status

    if status == 200 and isinstance(payload, dict):
        report.live = True
        report.models = _records_from_openai_models(payload)
        report.provider = _identify_provider(normalized, payload, provider)
        if not report.provider:
            report.provider = _refine_provider(root, timeout) or provider or "unknown"
            if report.provider != provider:
                report.notes.append("provider identified as %s" % report.provider)
        if not report.models:
            report.notes.append("server is live but lists no models")
    elif status is not None and status >= 500:
        report.error = error
        report.notes.append("server answered %s; it may still be loading a model" % status)
    else:
        report.error = error or ("HTTP %s" % status if status else "unreachable")
        return report

    if enrich:
        enricher = _ENRICHERS.get(report.provider)
        if enricher is not None:
            try:
                enricher(report, root, timeout)
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                report.notes.append("capability lookup failed: %s" % exc)
        if report.provider == "ollama":
            try:
                _enrich_per_model_context(report, root, timeout)
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                report.notes.append("context lookup failed: %s" % exc)

    if report.live and any(not m.local_verified and not m.remote for m in report.models):
        report.notes.append(
            "Execution locality is unknown for some models; a loopback endpoint can proxy a remote service."
        )
    remote = report.remote_models
    if remote:
        hosts = sorted({m.remote_host or "remote" for m in remote})
        report.notes.append(
            "%d of %d model(s) are routed off-device (%s)"
            % (len(remote), len(report.models), ", ".join(hosts))
        )
    return report


def _probe_all(
    endpoints: Sequence[Endpoint], timeout: float, enrich: bool, workers: int
) -> List[ServerReport]:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(workers, len(endpoints)))
    ) as pool:
        futures = [
            pool.submit(probe_endpoint, endpoint.base_url, endpoint.provider, timeout, enrich)
            for endpoint in endpoints
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - a probe must never break discovery
                results.append(ServerReport(base_url="unknown", provider="unknown", error=str(exc)))
    return results


def discover(
    extra_urls: Optional[Sequence[str]] = None,
    include_known: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    enrich: bool = True,
    max_workers: int = 8,
) -> DiscoveryReport:
    """Probe local LLM endpoints concurrently and report what is running.

    Args:
        extra_urls: Additional base URLs to probe (not required to be in
            :data:`ENDPOINTS`).
        include_known: Probe the built-in :data:`ENDPOINTS` catalog.
        timeout: Per-request timeout in seconds.
        enrich: Also read each server's own capability metadata.
        max_workers: Probe concurrency.

    Returns:
        A :class:`DiscoveryReport`. Dead endpoints appear with ``live=False``,
        so callers can show a full picture rather than only successes.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number of seconds.")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    endpoints: List[Endpoint] = []
    seen = set()
    for endpoint in list(ENDPOINTS) if include_known else []:
        normalized = normalize_base_url(endpoint.base_url)
        if normalized in seen:
            continue  # several providers share a well-known port (11434, 8080)
        seen.add(normalized)
        endpoints.append(Endpoint(endpoint.provider, normalized))
    for url in extra_urls or []:
        normalized = normalize_base_url(url)
        if normalized not in seen:
            seen.add(normalized)
            endpoints.append(Endpoint("custom", normalized))

    started = time.perf_counter()
    if not endpoints:
        return DiscoveryReport(servers=[], elapsed_ms=0.0, probed=0)
    reports = _probe_all(endpoints, timeout, enrich, max_workers)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return DiscoveryReport(servers=reports, elapsed_ms=elapsed, probed=len(endpoints))


def find_local_server(
    require_local: bool = True,
    extra_urls: Optional[Sequence[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    include_known: bool = True,
) -> Optional[ServerReport]:
    """Return a live server that can serve an on-device model, or None.

    With *require_local* (the default) only servers offering at least one
    on-device model qualify, so a caller cannot accidentally fall through to a
    provider-routed model.
    """
    report = discover(
        extra_urls=extra_urls, include_known=include_known, timeout=timeout, enrich=True
    )
    candidates = report.on_device() if require_local else report.live
    return candidates[0] if candidates else None
