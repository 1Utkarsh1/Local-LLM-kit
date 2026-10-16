# Competitor / alternative capability research for `local-llm-kit`

**Prepared:** 2026-10-17 · **Baseline:** `local-llm-kit` 0.2.0 working tree (`local_llm_kit/`, core dep = `typing-extensions` only)
**Evidence ledger:** `ledger_competitors.jsonl` — **211 substantively reviewed items** (94 source-code, 75 docs, 17 package-metadata, 8 issues, 8 PRs, 5 example-code, 2 test-code, 1 release-notes, 1 repo-metadata)
**Primary-source policy:** every version number and quote below comes from a file, docs page, GitHub API record or PyPI record fetched during this session. Where a claim comes from a `main`/`master` branch rather than a tagged release, that is stated.

---

## 0. Headline answer

The mature alternatives are **not** ahead of `local-llm-kit` on features a *local* library needs — they are ahead on **transport hygiene** (retries, timeouts split by phase, typed errors), **model-residency control** (`keep_alive` / `ttl`, readiness probes), and **making unreliable local JSON usable** (deterministic repair). Everything else they own — routers, failover, budgets, tracing SDKs, constrained-decoding engines, conversation databases — buys its value at a dependency cost that is incompatible with a stdlib-core library.

Three findings dominate:

1. **A 503 during a local cold start is a documented, still-open failure mode**, and it is not fixed by retry-on-5xx: Ollama queues requests while loading and returns 503 when overloaded (`docs/faq.mdx`), llama.cpp returns `503 {"error": {"code": 503, "message": "Loading model", …}}` from `/health` while loading, and users still file 503 reports against `ollama-python` with the OpenAI client (issue #164, 2024).
2. **Per-request residency control is the missing knob on both sides of the wire.** Ollama's own OpenAI-compat layer added a non-standard `keep_alive` field with the comment *"Ollama extension: without it an OpenAI-API client cannot release a model"*; LM Studio accepts a request-level `ttl`. Our kit stores `keep_alive` on the backend instance only.
3. **The reference SDK does not reconnect streams.** `openai-python` 3.31.0 documents that *stream consumption is not automatically retried*; litellm reimplemented its own retry policy rather than reusing it. A lightweight client should therefore ship a good retry policy for *unary* calls and a stall-detection/error-out policy for streams — not stream replay.

---

## 1. Gap matrix

Legend for **Worth adding?** — ✅ yes, light · ⚠️ only behind an optional extra · ❌ no (bloat).
**Est. cost** = new code + test burden, assuming stdlib-only unless stated.

### 1a. Transport, retries and resilience

| Capability | Who has it (evidence) | local-llm-kit today | Evidence | Worth adding? | Est. cost |
|---|---|---|---|---|---|
| Bounded retry policy with status allow-list (408/409/429/≥500) | `openai-python` 3.31.0 `_should_retry`; `litellm` 1.105.1 `_should_retry` (a deliberate reimplementation) | ❌ none — only tool-execution timeouts | ledger: `.../v3.31.0/src/openai/_base_client.py`; `.../litellm/v1.105.1/litellm/utils.py` | ✅ | ~60 LOC |
| Exponential backoff **with jitter**, capped | `openai-python`: `min(0.5*2**n, 8.0)` then `× (1 − 0.25·rand)`; `litellm`: `INITIAL_RETRY_DELAY*2**n` boxed + up to 0.75 s jitter | ❌ | same files (`_calculate_retry_timeout`, `_calculate_retry_after`) | ✅ | ~25 LOC |
| `Retry-After` honouring (int, `retry-after-ms`, HTTP-date) with a sanity cap | `openai-python`: `_parse_retry_after_header` + 120 s cap (`MAX_RETRY_AFTER_DELAY`); `litellm`: ≤60 s cap | ❌ | `_constants.py`, `_base_client.py` | ✅ | ~35 LOC |
| `max_retries` knob, `0` disables, validated loudly | `openai-python` (`_validate_max_retries`: 0 disables, 1000 allowed) | ❌ | README §Retries + `_base_client.py` | ✅ | ~10 LOC |
| Split connect vs read/write timeouts | `openai-python`: `Timeout(600, connect=5.0)` default, `httpx.Timeout(60, read=5, write=10, connect=2)` shown in README; LibreChat exposes `transportTimeouts{bodyTimeout,headersTimeout}` | ⚠️ one flat `timeout=` per backend (`urllib.urlopen(timeout=…)`) | `_constants.py`; `README.md §Timeouts`; `packages/api/src/types/openai.ts` | ✅ (connect vs read as **two numbers**, no httpx) | ~20 LOC |
| Default long read budget for cold starts | `openai-python` 600 s default; `llama-server -to/--timeout` default **3600 s**; `ollama-python` defaults to `timeout=None` | ⚠️ unclear default; not phase-aware | `_constants.py`; llama.cpp `tools/server/README.md`; `ollama/_client.py` | ✅ | doc + default change |
| Connection-refused tolerated during server startup (bounded accept-retry loop) | Real bug reports: vLLM #798 (2023), vLLM PR #20275 (2025, WSL2), aider PR #4977 (2026, "abort cleanly … when Ollama is unreachable"), Continue PR #12013 (2026, "improve local LM Studio connection errors") | ❌ | ledger issue/PR rows | ✅ | ~40 LOC |
| Readiness probe before the first request | llama.cpp `/health` (and `/v1/health`) returns 503 `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`; SGLang docs use `wait_for_server()`; `local-llm-kit` already exposes `GET /health` | ⚠️ server-side yes, client-side no | llama.cpp server README; SGLang `openai_api_completions.mdx`; `local_llm_kit/server.py` | ✅ | ~30 LOC + CLI flag |
| Capacity/queue signal | llama.cpp `/slots?fail_on_no_slot=1` → 503; Ollama queueing + `OLLAMA_MAX_QUEUE` | ❌ | both server docs | ⚠️ (poll `/health` only) | ~15 LOC |
| Typed, actionable errors (connection / timeout / 429 / 5xx / bad request / overloaded) | `ollama-python`: `ResponseError(text, status)` + `ConnectionError` with "check that Ollama is … running and accessible" + download link; TGI: `OverloadedError`, `ShardNotReadyError` (403/424), `ShardTimeoutError` (504); openai-python: ~20 classes | ❌ `RuntimeError`/`ValueError`/`EmbeddingError` only | `ollama/_client.py`, `text_generation/errors.py`, `openai/_exceptions.py` | ✅ (~6 classes) | ~50 LOC |
| Retry only replayable bodies; per-attempt idempotency key | `openai-python` gates retries on `request_body_replay.rewind()` and attaches `stainless-python-retry-<uuid4>` | n/a (buffered JSON bodies) | `_base_client.py` | ✅ (document the invariant) | 0 LOC |
| Streaming reconnect | **Nobody**: openai-python explicitly does not retry stream consumption; its `_streaming.py` has no reconnect path | ❌ (and no stall detection) | README §Handling errors; `openai/_streaming.py` | ⚠️ stall/idle timeout yes, replay no | ~30 LOC |
| Mid-stream error frames turned into exceptions | Ollama docs: mid-stream errors keep HTTP 200 and arrive as `{"error": …}` NDJSON; `ollama-python` raises `ResponseError` on the in-band `error` key | ⚠️ partial (kit streams but error-frame handling needs verification) | `docs/api/errors.mdx`; `ollama/_client.py` | ✅ | ~20 LOC |
| HTTP connection pooling | `openai-python` `Limits(max_connections=1000, max_keepalive_connections=100)` | ❌ (`urllib` opens a connection per request) | `_constants.py` | ❌ for a local single-user lib | — |

### 1b. Model lifecycle / cold start (the local-only problem class)

| Capability | Who has it | local-llm-kit today | Evidence | Worth adding? | Est. cost |
|---|---|---|---|---|---|
| Per-request `keep_alive` (incl. `-1` pin, `0` unload) | Ollama native API (`generate`/`chat`/`embed`/`create`); `ollama-python` exposes the kwarg on every method; `langchain_ollama` exposes `keep_alive`; `llama-index-llms-ollama` exposes `keep_alive` | ⚠️ instance-level only (default `'5m'`) | Ollama `docs/api.md`; `ollama/_client.py`; `langchain_ollama/chat_models.py`; `llama_index/…ollama/base.py` | ✅ | ~30 LOC |
| `keep_alive` through the **OpenAI-compatible** path | Ollama's Go compat layer: `KeepAlive *api.Duration json:"keep_alive,omitempty"` with comment *"Ollama extension: without it an OpenAI-API client cannot release a model"* | ❌ (kit's OpenAI-compat backend can't pin/unload) | `ollama/openai/openai.go` | ✅ | ~15 LOC |
| Explicit preload (warm the model before user work) | Ollama FAQ: empty `/api/generate` or `/api/chat` request preloads | ❌ | `docs/faq.mdx` | ✅ (2-line helper + CLI flag) | ~25 LOC |
| Idle-TTL semantics client-side | LM Studio: JIT loading default on, idle TTL default **60 min**, per-request `ttl` field, auto-evict keeps ≤1 JIT model; `lms load --ttl` | ❌ | LM Studio `ttl-and-auto-evict` doc | ✅ (send `ttl` when server is LM Studio) | ~15 LOC |
| Load/unload/ps verbs | `ollama-python`: `ps/show/pull/push/create/copy/delete`; LM Studio REST `/api/v1/models/load|unload|download`; `lms ps/ls/load/unload` | ⚠️ `pull`/`list` only | ledger rows | ✅ (`ps` + unload) | ~70 LOC CLI |
| Load-progress visibility | LM Studio native `/api/v1/chat` streams **model-load** and **prompt-processing** events (OpenAI-compat cannot) | ❌ | LM Studio REST comparison table | ⚠️ (nice-to-have; needs native API) | ~50 LOC, server-dependent |
| Context-length awareness / preflight oversize check | Ollama: VRAM-based defaults (4k/32k/256k) + `OLLAMA_CONTEXT_LENGTH`; `ollama ps` shows CONTEXT; litellm pre-call checks raise `ContextWindowExceededError`; llama-index auto-detects window via `/api/show` | ⚠️ `MessageHistory` truncates, no server-side check | `docs/context-length.mdx`; litellm reliability doc; `llama_index/…ollama/base.py` | ✅ (warn, don't block) | ~40 LOC |
| Per-model load knobs (gpu offload, context length) | `lms load` flags `ttl/gpu/contextLength/identifier` | ❌ | `lms/src/subcommands/load.ts` | ❌ (server/CLI domain) | — |

### 1c. Structured output

| Capability | Who has it | local-llm-kit today | Evidence | Worth adding? | Est. cost |
|---|---|---|---|---|---|
| `response_format` json_object/json_schema pass-through | Everywhere (Ollama, vLLM, SGLang, LM Studio, llama.cpp) | ✅ | `local_llm_kit/llm.py` `_normalize_response_format` | already have | — |
| Deterministic JSON repair of malformed model output | `json-repair` 0.64.0 (2 deps); **dspy 3.4.1 `JSONAdapter.parse()` calls `json_repair.loads()`**; llama-index ships `parse_partial_json` | ⚠️ re-prompts the backend once (`_ensure_json_output`) | `dspy/adapters/json_adapter.py`; `json_repair/README.md`; `llama_index/core/llms/utils.py` | ✅ | ~120 LOC minimal / ⚠️ optional extra |
| Schema-guided repair (drop extra keys, fill required, unwrap arrays) | `json-repair[schema]` (documented **beta**; raises `ValueError` if unrepairable) | ❌ | `json_repair/README.md` | ⚠️ optional extra | ~0 LOC (delegate) |
| Validation of the final object against the schema | `pydantic` 2.14.1 `model_validate_json`/`TypeAdapter`; `jsonschema` 4.26.0 (21 deps) | ❌ | pydantic JSON-schema docs; PyPI | ⚠️ optional extra (pydantic already in `server` extra) | ~40 LOC |
| JSON-schema derivation from Python types | pydantic `model_json_schema()` / `TypeAdapter.json_schema()`, `JsonSchemaMode` validation vs serialization, `json_schema_extra`, `SkipJsonSchema`, `GenerateJsonSchema` subclassing (strip unsupported constructs) | ⚠️ accepts raw schema only | pydantic docs | ✅ (accept a model class lazily) | ~30 LOC |
| Partial/streaming JSON extraction | llama-index `parse_partial_json` (credits open-interpreter's parser) | ❌ | `llama_index/core/llms/utils.py` | ✅ (CLI UX) | ~60 LOC |
| Grammar-constrained decoding (regex/EBNF/choice) | vLLM (`structured_outputs` object; `guided_*` **removed in v0.12.0**), SGLang (`json_schema`/`regex`/`ebnf`, backends XGrammar/Outlines/Llguidance), TGI client `grammar: Grammar` | ❌ | vLLM structured-outputs doc; SGLang docs; TGI client | ⚠️ pass-through only (`extra_body`) | ~20 LOC |
| Token-level enforcement | outlines 1.3.3 (Rust `outlines_core`, 71 requires_dist), lm-format-enforcer 0.11.3, guidance 0.3.1 | ❌ | those READMEs/pyprojects | ❌ not possible over HTTP | — |
| Structured output + reasoning interplay | vLLM flag `--structured-outputs-config.enable_in_reasoning=True`; Qwen3-Coder can silently disable structured output unless reasoning is parsed out | ❌ | vLLM doc | ⚠️ document + validate | ~15 LOC |

### 1d. Observability, accounting, persistence

| Capability | Who has it | local-llm-kit today | Evidence | Worth adding? | Est. cost |
|---|---|---|---|---|---|
| Token usage from responses (incl. streaming) | LangChain `stream_usage`/`stream_options.include_usage` (default on **only** for the default OpenAI base URL because many servers reject the field); dspy `usage_tracker` (cache hits excluded from billed usage); Ollama returns prompt/eval counts + `load_duration`; llama.cpp returns `timings` | ⚠️ `_usage()` estimates | `langchain_openai/chat_models/base.py`; `dspy/clients/base_lm.py`; Ollama `docs/api.md`; `ollama/openai/openai.go` | ✅ | ~80 LOC |
| In-process response cache (TTL, opt-in) | litellm `cache_responses`/`cache_type=local` (default 3600 s); dspy `cache=True` with `ignored_args_for_cache_key=['api_key','api_base','base_url']`; LangChain `InMemoryCache` | ⚠️ embeddings-only cache (FIFO, 4096) | ledger rows | ✅ | ~60 LOC |
| Local request logging / history | `llm` 0.36: sqlite logs with `input_tokens`, `output_tokens`, `token_details`, `duration_ms`, `conversation_id`, raw `response_json`, fragment hashing + `LogStore.verify()`, `llm logs on/off` privacy switch | ❌ | `docs/logging.md` | ⚠️ JSONL only (no sqlite-utils) | ~80 LOC |
| `ps`-style server status in the CLI | `lms ps/ls/status/log stream`; `ollama ps` (PROCESSOR, CONTEXT, UNTIL); Ollama `/api/ps` `expires_at` | ❌ | lms README; Ollama docs | ✅ | ~70 LOC |
| Model aliases | `llm` aliases (model + default options, usable anywhere a model ID is) | ❌ | `docs/aliases.md` | ✅ | ~60 LOC |
| Tracing (spans/OTLP) | `opentelemetry-instrumentation-openai` 0.62.4 (patches openai; `gen_ai.input.messages`/`gen_ai.output.messages` span attributes; manual span lifecycle for generator streams); Langfuse 5.0.2 monkey-patches `openai.resources.*` (chat/completions/beta/responses/embeddings) | ❌ | instrumentation `__init__.py`, `shared/chat_wrappers.py`, `langfuse/openai.py` | ⚠️ callback hooks + recipes, not a dependency | ~40 LOC |
| Server-side timing metrics | llama.cpp `/metrics` (Prometheus, `llamacpp:n_busy_slots_per_decode`, …); Ollama `load_duration` / `timings` | ❌ | llama.cpp README; Ollama docs + Go source | ⚠️ expose what backends already return | ~20 LOC |

### 1e. Server-side drop-in conformance (what real clients send/expect)

| Requirement | Who needs it | local-llm-kit server today | Evidence | Worth adding? | Est. cost |
|---|---|---|---|---|---|
| Accept **`max_completion_tokens`** as well as `max_tokens` | open-webui (converts `max_tokens`→`max_completion_tokens` for reasoning models, then converts back for older servers); Continue; vLLM; mlx-lm (`max_completion_tokens` with `max_tokens` fallback) | ⚠️ `max_tokens` only | `open_webui/routers/openai.py`; `continue/core/llm/llms/OpenAI.ts`; `vllm/entrypoints/openai/protocol.py`; `mlx_lm/server.py` | ✅ | ~15 LOC |
| Tolerate (not 400 on) `stream_options` | open-webui allow-list includes `stream_options` and pops it before some paths; LangChain only sends it when safe | ⚠️ unverified | same files | ✅ | ~10 LOC |
| `parallel_tool_calls`, `tool_choice` behaviour | Continue toggles `parallel_tool_calls`; vLLM documents `false` ⇒ ≤1 call; **Ollama does not support `tool_choice` at all** | ⚠️ kit supports `tool_choice` (incl. forcing) — must not *require* it | vLLM server doc; Ollama compat matrix; `continue/.../OpenAI.ts` | ✅ | ~10 LOC |
| `response_format` json_schema | LM Studio, vLLM, SGLang, Ollama | ✅ | ledger rows | already have | — |
| Return a `usage` block on every response | open-webui reads `usage` from upstream and re-emits it | ⚠️ verify completeness incl. streaming | `open_webui/routers/openai.py` | ✅ | ~20 LOC |
| OpenAI-shaped error bodies (`{"error": {...}}`) with correct status codes | TGI parses `error_type`/status; openai-python raises `APIStatusError(status_code, body)`; aider hit an uncaught `ServiceUnavailableError` | ⚠️ verify | TGI `errors.py`; `openai/_exceptions.py`; aider #2409 | ✅ | ~30 LOC |
| `GET /v1/models` shape + capability hints | Continue `FetchedModel{contextLength,maxTokens,supportsTools}`; vLLM examples discover the model via `models.list()[0].id`; llama.cpp router publishes modality metadata | ⚠️ basic list | `continue/core/llm/fetchModels.ts`; vLLM example | ✅ | ~25 LOC |
| Both `/v1/...` and bare paths; tolerate `/v1` in or out of `base_url` | `huggingface_hub` appends `/chat/completions` when a URL is passed as `model`/`base_url` | ⚠️ verify | `huggingface_hub/inference/_client.py` | ✅ | ~15 LOC |
| `/v1/completions` still needed | Continue falls back to the legacy endpoint for some models; `llm`-style tools still use it | ✅ have it | Continue `OpenAI.ts` | already have | — |
| Support **non-standard residency fields**: `keep_alive` (Ollama-compat) / `ttl` (LM Studio) | Ollama's own compat layer documents this as required for OpenAI clients to release models | ❌ | Ollama `openai/openai.go`; LM Studio TTL doc | ⚠️ accept-and-ignore, at minimum | ~10 LOC |
| Responses API / stateful chat / MCP | LM Studio native + `/v1/responses`; vLLM `/v1/responses` (+cancel); Ollama explicitly does **not** support stateful responses | ❌ | LM Studio comparison table; vLLM server doc; Ollama compat matrix | ❌ out of scope | — |
| Anthropic/Ollama-shaped endpoints | LM Studio: Anthropic-compat `/v1/messages` + Jev; SGLang: `/api/chat|generate|tags|show` for Ollama clients | ❌ | LM Studio REST doc; SGLang `ollama_api.mdx` | ⚠️ Ollama-shape `/api/tags`+`/api/chat` is the cheaper one | ~90 LOC |
| drop-param tolerance | LibreChat ships `dropParams: ['stop','user','frequency_penalty','presence_penalty']` for a provider | ✅ (kit ignores unknown fields) — document it | `librechat.example.yaml` | ✅ doc only | ~5 LOC |
| CORS for browser clients | kit already configures `CORSMiddleware` | ✅ | `local_llm_kit/server.py` | already have | — |

### 1f. Things alternatives have that a light lib should deliberately not copy

| Capability | Owner | Cost evidence | Why not |
|---|---|---|---|
| Failover router, cooldowns, weighted routing, budget enforcement | litellm `Router` (`fallbacks`, `context_window_fallbacks`, `allowed_fails`, `cooldown_time`, 5 routing strategies, `provider_budget_config`) | litellm declares **24 direct** runtime deps (openai, httpx[http2], aiohttp, tokenizers, huggingface-hub, tiktoken, boto3, jinja2, jsonschema, pydantic-settings…); PyPI lists **106** `requires_dist`; aider notes litellm costs **~1.5 s** to import | Belongs in a gateway process; users who need it can run litellm *in front of* our presets |
| 100+ provider mapping & cloud auth (Azure ad tokens, Bedrock SigV4, Vertex regions) | openai-python (Azure/Bedrock sections), litellm, huggingface_hub `provider=` policies | openai-python 14 requires_dist; hf_hub 109 | Zero user value for a server on `localhost` |
| Token-level constrained decoding | outlines (Rust `outlines_core`), lm-format-enforcer, guidance | outlines 71 requires_dist; lmfe 0.11.3 last released 2025-08-24 | Requires owning the sampler; over HTTP you get `response_format` and nothing more |
| Tracing/generation-tracking SDK | langfuse 5.0.2; opentelemetry-instrumentation-openai 0.62.4 | 9 and 5 requires_dist; OTel instrumentation is version-coupled to the `openai` client | Callback hook + documented recipes keeps the dep optional and unbroken |
| sqlite conversation database with fragment hashing and integrity verification | `llm` 0.36 (sqlite-utils) | 15 deps incl. sqlite-utils, pluggy, puremagic | Excellent, but it is a product, not a feature; opt-in JSONL is 95 % of the value |
| Plugin system (pluggy) | `llm` | +pluggy, plus a stable hook contract to maintain | Our 12 presets + a documented backend registration function cover the realistic extension need |
| Websocket/native SDK transports (model-load progress, stateful chat) | lmstudio-python (httpx-ws + msgspec) | 6 deps | HTTP + polling `/health` covers the observable need; native paths can be documented per server |
| Realtime API, webhooks, file uploads, pagination, nested-param flattening | openai-python | — | Not applicable to local text/vision/embedding servers |
| Agent framework (sessions, sandbox, approvals) | open-interpreter now wraps Codex; openai-agents (56 requires_dist) | 56 deps | Out of scope for a client library |

---

## 2. Top 8 recommended additions (ranked by user benefit ÷ cost)

### 1. Retry policy in the core HTTP layer — `max_retries`, exponential backoff + jitter, `Retry-After`, and a *separate* connect-retry loop
**Benefit:** eliminates the single most common local failure (503/queue/connection-refused at startup) without user code. **Cost:** ~85 LOC, **0 new deps** (`time.sleep`, `random`, `email.utils` for HTTP-date parsing).
**Design:** one shared `_http.py` module used by both `backends/ollama.py` and `backends/openai_compat.py`:
`should_retry(status)` → 408/409/429/≥500 (matches openai-python 3.31.0 and litellm 1.105.1 exactly); `retry_delay(n, headers)` → `min(0.5·2ⁿ, 8.0)·(1−0.25·rand)` unless a `retry-after` of 0–120 s is present (openai-python's exact rule, incl. the 120 s cap); `max_retries=2` default with `0` disabling; validate loudly (`openai-python` raises on negative/None).
**Keep out:** stream replay. `openai-python` 3.31.0 documents streams are never auto-retried; instead add an *idle/stall timeout* that aborts a stream with a typed error (see #5).
**Evidence:** https://raw.githubusercontent.com/openai/openai-python/v3.31.0/src/openai/_constants.py · https://raw.githubusercontent.com/openai/openai-python/v3.31.0/src/openai/_base_client.py · https://raw.githubusercontent.com/BerriAI/litellm/v1.105.1/litellm/utils.py

### 2. Cold-start handling: split timeouts, readiness wait, and a warm-up helper
**Benefit:** turns "it hung / it 500'd" into "it waited 40 s for the model to load". **Cost:** ~70 LOC, 0 deps.
**Design:** `connect_timeout` (default 5 s) vs `read_timeout` (default 600 s, matching openai-python's 10-minute default; `llama-server`'s own default read timeout is 3600 s); `wait_for_server(host, timeout)` polling `GET /health` (llama.cpp returns `503 {"error":{"code":503,"message":"Loading model",…}}` while loading; our own server already has `/health`); `LLM.warmup()` that sends Ollama's documented empty `/api/generate` preload request; CLI `--wait-for-server / --warmup`.
**Evidence:** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md · https://raw.githubusercontent.com/ollama/ollama/main/docs/faq.mdx · https://github.com/vllm-project/vllm/issues/798 · https://github.com/Aider-AI/aider/pull/4977

### 3. Per-request model residency: `keep_alive` / `ttl` override + `unload()`
**Benefit:** users can pin a model for a batch and release RAM afterwards; this is the top request in local-LLM issue trackers. **Cost:** ~45 LOC, 0 deps.
**Design:** accept `keep_alive` as a **per-call** kwarg on `chat/complete/embed` (merging with the instance default); accept `ttl=` for LM Studio; add `LLM.unload()` (`keep_alive=0` on Ollama, `/api/v1/models/unload` on LM Studio, `--no-keep` style for others); and add `keep_alive` (Ollama) / `ttl` (LM Studio) to the **OpenAI-compat backend** too, because Ollama's own compat layer ships that field precisely so OpenAI-shaped clients can release models.
**Evidence:** https://github.com/ollama/ollama/blob/main/openai/openai.go (comment: *"Ollama extension: without it an OpenAI-API client cannot release a model"*) · https://github.com/ollama/ollama/blob/main/docs/api.md · https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict · https://github.com/ollama/ollama/issues/10263

### 4. A 6-class error taxonomy with actionable messages
**Benefit:** makes every failure path scriptable and testable; kills "which RuntimeError was that?". **Cost:** ~50 LOC, 0 deps.
**Design:** `LocalLLMError` → `ServerNotRunningError` (message includes host:port + "is the server started? see `local-llm-kit serve`"), `RequestTimeoutError` (with the elapsed seconds), `ModelLoadingError` (detects `503` + "Loading model"/"model is loading"/"server is overloaded" bodies), `RateLimitError`, `ServerError` (status_code + body + `response_text`), `BadRequestError`. Mirrors `ollama-python`'s 2-class minimum and TGI's capacity-error naming (`OverloadedError`, `ShardNotReadyError`, `ShardTimeoutError`). Keep all existing exception types as aliases for backwards compatibility.
**Evidence:** https://github.com/ollama/ollama-python/blob/v0.6.3/ollama/_client.py · https://raw.githubusercontent.com/huggingface/text-generation-inference/main/clients/python/text_generation/errors.py · https://raw.githubusercontent.com/openai/openai-python/v3.31.0/src/openai/_exceptions.py · https://github.com/Aider-AI/aider/issues/2409

### 5. Structured-output hardening: extract → repair → validate → (optionally) repair-toward-schema
**Benefit:** this is the difference between "works with 7B local models" and "works with GPT-4o". **Cost:** ~140 LOC stdlib-only (extract JSON from prose/fences, fix trailing commas, unquoted keys, single quotes, close unbalanced braces/strings, plus llama-index-style partial-JSON closing for streaming), **plus an optional `json-repair` extra (2 deps)** for the hard cases and `[schema]` for schema-guided repair.
**Design:** replace `_ensure_json_output`'s blind re-prompt with a deterministic pipeline: `try json.loads` → `extract_json(text)` (fence/brace scanning) → `repair_json(text)` (optional `json-repair`, which **dspy 3.4.1 already uses in production** as `JSONAdapter.parse`) → `validate(schema)` (optional `pydantic`/`jsonschema`) → only then consider one re-prompt (keep it, but behind a flag). Add a `parse_partial_json` helper for streaming CLI display (algorithm is 50 lines and two independent projects converged on it).
**Evidence:** https://raw.githubusercontent.com/mangiucugna/json_repair/main/README.md · https://raw.githubusercontent.com/stanfordnlp/dspy/main/dspy/adapters/json_adapter.py · https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-core/llama_index/core/llms/utils.py · https://pydantic.dev/docs/validation/latest/concepts/json_schema/

### 6. Real token/usage accounting + optional in-memory response cache
**Benefit:** cost/latency visibility and a large speed-up on repeated prompts; makes CI runs deterministic. **Cost:** ~80 LOC accounting + ~60 LOC cache, 0 deps.
**Design:** prefer server-reported usage, fall back to backend counters — Ollama returns prompt/eval counts (and `load_duration`), llama.cpp returns `timings`, and both are exposed through the OpenAI-compat surfaces too (Ollama's `Timings` struct). Expose `response.usage = {prompt_tokens, completion_tokens, total_tokens, load_ms?}`. Cache: ordered dict keyed on `(model, messages, params)` with TTL and `--no-cache`, *ignoring* host/api-key in the key (dspy's `ignored_args_for_cache_key=['api_key','api_base','base_url']` rule) — the kit's embeddings path already does FIFO vector caching, so this is a familiar local change.
**Evidence:** https://raw.githubusercontent.com/simonw/llm/0.36/docs/logging.md · https://raw.githubusercontent.com/stanfordnlp/dspy/main/dspy/clients/lm.py · https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/partners/openai/langchain_openai/chat_models/base.py · https://raw.githubusercontent.com/BerriAI/litellm/v1.105.1/litellm/router.py

### 7. Dev-server drop-in conformance pass (accept what real clients send)
**Benefit:** makes `local-llm-kit serve` usable as the backend for Open WebUI / LibreChat / Continue / aider without per-client patching. **Cost:** ~110 LOC in `server.py`, 0 new deps.
**Checklist (from client source):** accept `max_completion_tokens` (open-webui and Continue both send it); never 400 on `stream_options` (open-webui sends it, LangChain guards it); accept `parallel_tool_calls` and `tool_choice="none"`; return a `usage` block always (open-webui re-emits it); return OpenAI-shaped `{"error": {...}}` with 400/404/503 and map "model not found" → 404; extend `/v1/models` with `created`, `owned_by` and optional `context_length`/`max_tokens`/`supports_tools` hints (Continue reads those); accept `keep_alive`/`ttl` and ignore them gracefully; accept both `/v1/chat/completions` and `/chat/completions` (hf_hub appends the suffix to whatever base URL it is given). Optionally add Ollama-shaped `/api/tags` + `/api/chat` (SGLang does this to win Ollama clients) — that is the cheapest second dialect to support.
**Evidence:** https://raw.githubusercontent.com/open-webui/open-webui/main/backend/open_webui/routers/openai.py · https://raw.githubusercontent.com/danny-avila/LibreChat/main/librechat.example.yaml · https://raw.githubusercontent.com/continuedev/continue/main/core/llm/llms/OpenAI.ts · https://raw.githubusercontent.com/huggingface/huggingface_hub/v2.2.0/src/huggingface_hub/inference/_client.py · https://raw.githubusercontent.com/sgl-project/sglang/main/docs/docs/basic_usage/ollama_api.mdx

### 8. CLI observability: `status`/`ps`, `--json`, model aliases, and an optional JSONL request log
**Benefit:** answers "is my model loaded, and why was that call slow?" without leaving the terminal, and makes our CLI feel as capable as `lms`/`llm`. **Cost:** ~200 LOC total, 0 deps (stdlib `json`, `sqlite3`-free JSONL).
**Design:** `local-llm-kit status` → server reachable?, `/health`, Ollama `/api/ps` (`expires_at`, CONTEXT, PROCESSOR via `/api/ps`+`/api/show`), LM Studio `/api/v1/models`; `local-llm-kit ps` lists loaded models; aliases in a small config file (`~/.config/local-llm-kit/aliases.json`) resolving to `{backend, model, host, options}` (the `llm` aliases design); opt-in `--log`/`LOCAL_LLM_KIT_LOG=path` writing JSONL rows with `model, backend, input_tokens, output_tokens, duration_ms, load_ms, status, prompt_hash` — the same fields `llm` stores in sqlite, without the dependency, and with a documented "logging off" switch (`llm logs off` precedent).
**Evidence:** https://github.com/lmstudio-ai/lms/blob/main/README.md · https://github.com/ollama/ollama/blob/main/docs/context-length.mdx · https://raw.githubusercontent.com/simonw/llm/0.36/docs/aliases.md · https://raw.githubusercontent.com/simonw/llm/0.36/docs/logging.md

**Runner-up (9th, cheap):** client-side `extra_body`/`extra_params` pass-through on every call, and a `thinking`/`reasoning` field on responses that strips reasoning text out of `content` (Ollama's compat layer does exactly this; Continue's converters show how 400s arise from mis-replayed reasoning fields). ~35 LOC, 0 deps.
**Runner-up (10th):** a *server-side* `/v1/context_length`-style preflight (count prompt tokens, warn when the request exceeds the loaded context) — Ollama's VRAM-based defaults (4k/32k/256k) silently cut prompts otherwise. ~40 LOC.

---

## 3. Anti-recommendations (do **not** add)

1. **A litellm-style router/failover/budget layer.** 24 direct runtime deps (PyPI lists 106 `requires_dist`), and aider carries a code comment that `import litellm` costs ~1.5 s. Users who need failover should run litellm *in front of* our presets. Evidence: `litellm/v1.105.1/pyproject.toml`; `aider/llm.py`.
2. **Constrained-decoding engines** (`outlines` 71 requires_dist + Rust core; `lm-format-enforcer`; `guidance`). Token-level enforcement needs sampler access; over HTTP the only honest lever is `response_format` plus `extra_body` for server-specific grammars. Evidence: the three pyprojects/READMEs; vLLM `docs/features/structured_outputs`.
3. **A tracing/observability SDK dependency** (langfuse 5.0.2, `opentelemetry-instrumentation-openai` 0.62.4 — the latter is version-coupled to the `openai` client it patches). Ship a callback hook and documented recipes instead. Evidence: `langfuse/openai.py`; instrumentation `__init__.py`.
4. **A plugin system à la `llm` (pluggy) or an agent framework.** A stable hook contract is a compatibility liability; a documented `register_backend()` function plus 12 presets covers real use. Evidence: `llm/0.36 docs/plugins/*`, 15 deps.
5. **A sqlite conversation store with fragment hashing and integrity verification** — beautiful, but it is a second product; opt-in JSONL logging captures most of the benefit with `json` only. Evidence: `llm/0.36 docs/logging.md`.
6. **Native websocket SDKs / Realtime API / stateful chat / MCP.** LM Studio's native path costs `httpx-ws` + `msgspec`; the observable benefit (load progress) is approximated by polling `/health`. Evidence: `lmstudio-python/1.5.0/pyproject.toml`; LM Studio REST doc.
7. **Provider routing and cloud auth** (Azure AD/Bedrock SigV4, hf_hub `provider=` policies, 100+ provider maps). Irrelevant to `localhost`; keep `base_url` + `api_key`. Evidence: openai-python README §Azure/§Bedrock; `huggingface_hub` InferenceClient.
8. **Swapping the transport to httpx/httpx2 or adding `pydantic` to core.** openai-python needed a migration guide when it moved to httpx2; httpx alone is 12 `requires_dist` entries. Our value is that core is `typing-extensions`-only.
9. **`tiktoken` for token counting.** Rust extension + runtime encodings; prefer backend counters (Ollama prompt/eval counts, llama.cpp timings, `/tokenize`).
10. **`instructor`-style validate-and-retry framework** (72 `requires_dist`). The valuable 20 % (`json-repair` + optional `pydantic`) is 2–6 deps. Evidence: PyPI metadata.
11. **Stream replay/reconnect logic.** Nobody does it — not even the reference SDK — and partial replay corrupts output. Detect stalls and fail loudly instead. Evidence: openai-python 3.31.0 README/`_streaming.py`; `litellm` mid-stream fallback notes.
12. **Auto-launching/managing local server processes, GPU offload tuning, download resume, pagination helpers, file uploads, webhook verification.** Server/CLI domain, platform-fragile, or inapplicable to local text APIs.

---

## 4. What we already match (so we can legitimately claim it)

Streaming with SSE/NDJSON; tool calling (OpenAI `tools` + legacy `functions`) and a tool-execution loop that converts failures into observations instead of exceptions (`tool_timeout`, default 10 s — several competitors raise instead); `response_format` json_object/json_schema; vision passthrough; sync+async parity (matching `ollama-python`'s Client/AsyncClient design); embeddings + RAG helper with a bounded vector cache; model download with partial-file cleanup; a dev server with `/health`, CORS and the four core OpenAI routes; 12+ server presets (which `llm` deliberately does not ship — it delegates to plugins).

---

## 5. Research honesty, counts and limitations

**Counts:** 211 ledger rows; distinct projects 37 (incl. the `local-llm-kit` baseline); evidence types: 94 source-code, 75 docs, 17 package-metadata, 8 issues, 8 PRs, 5 example-code, 2 test-code, 1 release-notes, 1 repo-metadata. Roughly 60 % of rows are dated by a PyPI release (version + upload date captured in this session) and the rest by branch (`main`/`master` fetched 2026-10-17) or GitHub issue/PR date.

**Limitations to keep in mind:**
- Branch-derived findings carry no version pin. Notable ones: Ollama `docs/*` and `openai/openai.go`, SGLang `docs/*`, vLLM `docs/*`, llama.cpp `tools/server/README.md`, `mlx_lm/server.py`, dspy/langchain/llama-index `main`/`master`, open-webui/LibreChat/Continue/aider `main`.
- **Correction recorded in the ledger:** an early search hit suggested Ollama's OpenAI-compat layer does not expose `keep_alive` (open PR #11249). Reading `ollama/openai/openai.go` showed the chat request *does* carry a `keep_alive` extension field. The ledger contains both rows; the source-code row supersedes.
- Some rows are deliberately shallow: I read READMEs/heads/signature blocks rather than entire files for `outlines`, `guidance`, `lm-format-enforcer` internals, `langfuse` wrapper internals, `otel` wrappers (44 KB, grepped), `ollama-python` tests (47 KB, grepped), LibreChat `llm.ts` (34 KB, grepped). Those rows say so.
- `open-interpreter`'s legacy `interpreter/core/llm` source was **not retrievable** — the project now resolves to `openinterpreter/openinterpreter` and is a Codex-based terminal agent. No client-side field requirements were derived from its old code; that gap is stated rather than guessed.
- No runtime testing was performed: every competitor claim is from source/docs/metadata, not from executing their clients against a live server. The "est. cost" figures are engineering estimates (LOC + deps + test burden), not measured.
- Nothing was invented: every version string (`openai 3.31.0`, `ollama 0.6.3`, `litellm 1.105.1`, `lmstudio 1.5.0`, `llm 0.36`, `vllm 0.31.0`, `sglang 0.5.22`, `dspy 3.4.1`, `langchain 1.4.6`/`langchain-core 1.6.11`/`langchain-openai 1.7.2`/`langchain-ollama 1.1.0`, `llama-index 0.14.25`, `outlines 1.3.3`, `lm-format-enforcer 0.11.3`, `guidance 0.3.1`, `json-repair 0.64.0`, `pydantic 2.14.1`, `huggingface-hub 2.2.0`, `text-generation 0.7.0`, `langfuse 5.0.2`, `opentelemetry-instrumentation-openai 0.62.4`, `instructor 1.17.0`, `jsonschema 4.26.0`, `tenacity 9.2.1`, `tiktoken 0.14.0`, `openai-agents 0.24.0`) and date came from the PyPI/GitHub APIs during this session.
