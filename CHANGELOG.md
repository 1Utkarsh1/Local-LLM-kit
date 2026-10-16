# Changelog

All notable changes to Local LLM Kit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `tool_timeout` did not actually bound tool execution: the
  `ThreadPoolExecutor` context manager joined its worker, so a slow or
  hanging tool blocked the request for its full runtime before the
  "timed out" result was produced. Timed-out calls now return promptly.
- A tool returning a non-JSON-serializable value (e.g. a `datetime`)
  raised `TypeError` out of `LLM.chat` instead of becoming tool output,
  and left a dangling tool call in memory. Results are now serialized with
  a `str` fallback, keeping the documented "failures become tool-output
  strings, never tracebacks" behaviour.
- `role: "tool"` messages were silently dropped by the ChatML, Llama-2,
  Mistral, Vicuna and plain-instruction prompt formatters (the default one
  for unknown models), so tool results never reached the model and the
  call loop repeated the same call. All formatters now render them.
- JSON mode (`response_format={"type": "json_object"}`) discarded nested
  JSON: the brace-matching regex stopped at the first closing brace and
  replaced the model's answer with a `{"error": "Failed to generate valid
  JSON", ...}` object. Extraction now uses a real JSON decoder.
- A malformed `messages` argument (a non-dict entry) was appended to the
  history before validation, leaving the instance — and every caller
  sharing a cached client — unable to serve *any* later request. The batch
  is now validated before anything is stored.
- CLI `pull`: the download-helper signature adapter did not know
  `download_model`'s `repo_or_path` parameter, so every `pull` invocation
  aborted with "unrecognised signature". `pull owner/repo:file.gguf`,
  `--output-dir` and `--revision` now reach the helper.
- CLI `--response-format`: a plain JSON Schema was mistaken for a
  `response_format` object whenever it carried a `"type"` key (e.g.
  `{"type": "object", ...}`), which failed deep inside `LLM.chat` with
  "Unsupported response_format type". Schemas are now wrapped as
  `json_schema`; only `json_object`/`json_schema`/`text` are treated as
  response_format objects.
- Server `POST /v1/embeddings`: an LLM whose `embed()` returns the
  OpenAI-shaped `{"data": [{"embedding": ...}]}` dict (i.e.
  `local_llm_kit.LLM.embed`) had its vectors discarded and replaced by the
  deterministic pseudo-embedding fallback. Real vectors are now served.
- Downloads write to `<dest>.part` and are renamed into place only after a
  complete transfer, so a failed, truncated or interrupted download can no
  longer leave a file that `list_cached_models()` / `resolve_model()` would
  report as a usable cached model.
- `SimpleVectorStore.load` raises the documented `ValueError` (with the
  offending field named) for persisted records with null or wrongly-typed
  fields instead of leaking `TypeError: object of type 'NoneType' has no
  len()`.
- `examples/simple_chat.py --backend echo` (the README's offline
  invocation) failed: `echo` was not an accepted `--backend` choice and
  `--model` was mandatory. Both now work as documented.

### Added
- Regression tests for the fixes above (`tests/test_cli.py`,
  `tests/test_models.py`, plus cases in `tests/test_rag.py` and
  `tests/test_server.py`).

## [0.2.0] - 2026-09-18

### Added
- New backends: `echo` (offline deterministic mock, zero deps), `ollama`
  (Ollama REST incl. chat/embeddings/model listing), and `openai-compat`
  (any `POST /v1/chat/completions` server: vLLM, LM Studio, Ollama,
  llama.cpp server) — all stdlib-HTTP, lazily imported
- OpenAI-compatible HTTP server (`local-llm-kit serve`,
  `local_llm_kit.server.create_app`): `GET /health`, `GET /v1/models`,
  `POST /v1/chat/completions` (incl. SSE streaming + `execute_tools` opt-in),
  `POST /v1/completions`, `POST /v1/embeddings`
- Modern tool calling: `@tool` decorator with signature-derived JSON Schema,
  `LLM.add_tool`, per-call `tools=` / `tool_choice=`, parallel calls,
  `tool`-role loop with iteration cap, timeout, scope check and arg validation
- Structured output: `response_format={"type": "json_object"}` and
  `{"type": "json_schema", ...}` (Pydantic schemas supported when installed)
- Embeddings API (`LLM.embed`, `chat.embed`, `POST /v1/embeddings`) plus
  `local_llm_kit.embeddings` (batching, cache, numpy-optional cosine,
  deterministic `hash_embed` for offline tests)
- Vision messages: `content: [{"type": "text", ...}, {"type": "image_url", ...}]`
  with text fallback on text-only backends; new prompt templates for
  Llama 3, Qwen 2/3, Gemma 2/3, Phi-3/4, DeepSeek
- Minimal RAG helpers: `local_llm_kit.rag` (`chunk_text`,
  `SimpleVectorStore` with cosine search + JSON save/load, `retrieve`,
  `build_rag_prompt`)
- Model helpers: `local_llm_kit.models` (`download_model`, `download_gguf`,
  `list_cached_models`, `resolve_model`, starter-model table) + CLI
  `pull` / `list` commands
- CLI: new `serve`, `pull`, `list`, `embed` subcommands; `--tools`,
  `--tool-choice`, `--response-format`, `--execute-tools`, `--message`,
  `--base-url`, `--version` flags; works offline with `--backend echo`
- Performance: shared client/backend caches (`get_client`,
  `clear_backend_cache`) — module-level `chat()`/`complete()` no longer
  reload the model per call; async `achat`/`acomplete` added
- Docs/DX: rewritten README (badges, 60s Ollama quickstart, backend table,
  OpenAI-SDK drop-in, migration guide); new offline-runnable examples
  (`openai_compatible`, `tools_structured`, `rag_minimal`, `vision_chat`);
  refreshed CONTRIBUTING, issue/PR templates, Code of Conduct
- Packaging: modern `pyproject.toml` (light core, 8 extras), CI matrix
  (3.9–3.12), offline test-suite (tools, prompts, RAG, server, caching)

### Changed
- `LLM.chat` accepts `tools`, `tool_choice`, `response_format`, `seed`,
  `stop`, `execute_tools`, `tool_timeout`; `complete` accepts
  `response_format`, `seed`, `stop`
- Version bumped to 0.2.0; Development Status → Beta; requires-python → >=3.9

### Deprecated
- Per-call `functions=` / `function_call=` and `LLM.add_function` in favour
  of `tools=` / `tool_choice=` / `add_tool`; `format="json"` in favour of
  `response_format`. All still work — no breaking change in 0.2.0.

### Security
- Server never executes tools unless `"execute_tools": true` is set per
  request; tool execution is scoped to advertised tools, required-arg
  validated, and time-bounded; model downloads are https/host-allowlisted
  with cache-contained paths; tool-call parsing is length-capped

## [0.1.2] - 2024-03-15

### Added
- Comprehensive documentation in PyPI package
- Additional documentation files included in distribution

### Fixed
- Documentation visibility on PyPI

## [0.1.1] - 2024-03-15

### Added
- Comprehensive documentation system
- Detailed API reference documentation
- Step-by-step tutorials with code examples
- Best practices guide
- Supported models documentation
- Contributing guidelines
- PyPI packaging improvements

### Changed
- Reorganized documentation structure
- Enhanced README with more examples

## [0.1.0] - 2023-11-25

### Added
- Initial release of Local LLM Kit
- Chat and Completion API (OpenAI-compatible)
- Function calling with automatic execution
- Multiple model backend support:
  - Hugging Face Transformers
  - llama.cpp (GGUF models)
- Streaming response support
- Memory management with auto-truncation
- JSON mode and structured output
- Prompt formatting for different model architectures:
  - Llama 2 Chat
  - Mistral Instruct
  - Vicuna
  - ChatML
  - Plain Instruct
- Command line interface
- Basic examples and documentation

### Changed
- N/A (Initial release)

### Deprecated
- N/A (Initial release)

### Removed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

### Security
- N/A (Initial release) 