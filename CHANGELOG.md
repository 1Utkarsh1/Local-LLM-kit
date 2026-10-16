# Changelog

All notable changes to Local LLM Kit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Native `apple-foundation` backend using Apple's on-device Foundation Models (`apple-fm-sdk`). Supports `chat()`/`complete()`, native function calling (OpenAI tools converted to `apple_fm_sdk.Tool` and executed on-device), and JSON `response_format` via Apple's `json_schema` guided generation. Requires macOS 26.0+ with Apple Intelligence enabled; streaming is unsupported.
- Auto-detection of the Apple backend on macOS 26+ when available (low priority behind the existing server/gguf/ollama/`transformers` resolution).
- CLI `--backend apple-foundation` / `--backend apple` and a gated `implementations`/`execute_tools` pass-through for tool-needing native backends.
- Optional `apple-fm` extra (`pip install local-llm-kit[apple-fm]`), included in the `all` extra.

### Fixed
- AirLLM `generate_stream` now yields a standardized `text`/`finish_reason` chunk (consistent with the other streaming backends) instead of the full `generate()` dict.

## [0.2.0] - 2026-10-17

### Added
- Native chat, tool-call responses, streaming argument assembly and structured-output forwarding for Ollama and OpenAI-compatible servers.
- Server presets for SGLang, vLLM, LM Studio, LocalAI, llama-server, Jan, GPT4All, MLX-LM and KoboldCpp, shared between the Python API, registry and CLI; documented limits across 12 distinct runtimes.
- An experimental optional AirLLM adapter with PyTorch/MLX contract tests, serialized generation, explicit context limits and buffered streaming.
- High-level helper options for per-call tool execution, iteration/time limits and sampling, preserving existing positional argument positions.
- `pull --filename/--file` and working `--force` cache refresh; optional custom recommended-model tables.
- Accurate quickstart/API/backend references and `llms.txt` for project discovery.
- Regression and protocol tests, including local HTTP fixtures without model downloads.

### Fixed
- Broken CLI download-helper argument mapping, JSON Schema detection and missing backend-configuration forwarding.
- Native embedding responses were discarded for pseudo-embeddings by the development server.
- Failed, truncated or interrupted downloads could replace/delete complete cached models. Unique temporary siblings are now atomically replaced only after validation.
- Cache lookup could select another repository's same-named file or an arbitrary revision. Lookups now follow repository/revision identity and Hugging Face refs.
- Invalid/null function and tool-call records raised low-level exceptions or propagated invalid arguments. Deserializers and parsers now validate consistently.
- Per-request callable tools leaked into a shared client's persistent registry.
- Tool timeout handling waited for timed-out workers, non-JSON tool results escaped as exceptions, and nested JSON extraction truncated valid output.
- Prompt templates dropped tool calls/results; Llama-2 duplicated the final user turn and Mistral dropped results before a follow-up user message.
- Malformed message batches could poison shared history; token estimates omitted tool arguments.
- Corrupt vector-store collections and values were accepted or raised unclear exceptions.
- Server generation options now return HTTP 400 for invalid values, schemas are forwarded, and runtime TypeErrors do not repeat requests.
- Ollama CLI host overrides were passed under an unused parameter name.
- Image content is preserved through the bundled server and inline image parts map to Ollama's native base64 image field.
- Remote tool execution now requires a boolean flag; truthy strings cannot enable it.
- The documented offline simple-chat example rejected its advertised arguments.

### Compatibility notes
- Requires Python 3.9 or newer; the prior PyPI release supported Python 3.8.
- Explicit nulls, JSON arrays/scalars and malformed call arguments are rejected; existing empty-string tool arguments still mean an empty object. Empty chat lists and empty completion prompts remain accepted by the Python API.
- URL caches now include the full URL's hash; older flat URL cache files are preserved but may be downloaded once again.
- Tool timeouts bound waiting; already-running Python functions continue in their worker thread.
- No new core runtime dependencies. AirLLM is optional, experimental and not GPU-certified; external-server features depend on runtime/model configuration.

### Earlier 0.2.0 development

#### Added
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

#### Changed
- `LLM.chat` accepts `tools`, `tool_choice`, `response_format`, `seed`,
  `stop`, `execute_tools`, `tool_timeout`; `complete` accepts
  `response_format`, `seed`, `stop`
- Version bumped to 0.2.0; Development Status → Beta; requires-python → >=3.9

#### Deprecated
- Per-call `functions=` / `function_call=` and `LLM.add_function` in favour
  of `tools=` / `tool_choice=` / `add_tool`; `format="json"` in favour of
  `response_format`. All still work — no breaking change in 0.2.0.

#### Security
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
