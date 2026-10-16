# Local LLM Kit

[![PyPI version](https://badge.fury.io/py/local-llm-kit.svg)](https://pypi.org/project/local-llm-kit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/1Utkarsh1/Local-LLM-kit/blob/v0.2.0/LICENSE)
[![CI](https://github.com/1Utkarsh1/Local-LLM-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/1Utkarsh1/Local-LLM-kit/actions)
[![Docs](https://readthedocs.org/projects/local-llm-kit-package/badge/?version=latest)](https://local-llm-kit-package.readthedocs.io/)
[![Offline ready](https://img.shields.io/badge/offline-echo%20backend-lightgrey.svg)](#offline-mode-no-model-needed)

**One Python API for offline models and local LLM servers.** Use the same chat and completion interface with Ollama, AirLLM, SGLang, vLLM, LM Studio, LocalAI, llama.cpp or Hugging Face Transformers; tools, schemas and embeddings follow backend capabilities. Start with the offline echo backend, then connect a real model. Core install stays light (stdlib + `typing-extensions`); inference dependencies are optional.

Version **0.2.0** includes the local runtime presets and reliability fixes below. See the [backend compatibility guide](https://github.com/1Utkarsh1/Local-LLM-kit/blob/v0.2.0/docs/source/backends.rst) for tested contracts and runtime limitations, and [llms.txt](https://github.com/1Utkarsh1/Local-LLM-kit/blob/v0.2.0/llms.txt) for a concise project reference.

> **Scope, honestly stated:** this is a zero-heavy-dependency client + adapter + offline mock — code once against the OpenAI shape, swap backends without rewriting, and test in CI without a GPU. It is not an inference engine (use vLLM/Ollama to serve traffic) and `serve` is a development shim, not production infra.

**Created by Utkarsh Rajput ([1Utkarsh1](https://github.com/1Utkarsh1))**

## 60-second quickstart (Ollama)

```bash
pip install "local-llm-kit[ollama]"
ollama pull llama3.2:3b
ollama serve &  # usually already running on http://localhost:11434
```

```python
from local_llm_kit import LLM

llm = LLM(model_path="llama3.2:3b", backend="ollama")
resp = llm.chat(messages=[{"role": "user", "content": "Say hi in one sentence."}])
print(resp["choices"][0]["message"]["content"])
```

CLI equivalent:

```bash
local-llm-kit chat --model llama3.2:3b --backend ollama
```

## Offline mode (no model needed)

Every example and the test-suite run without downloads via the `echo` backend:

```bash
python examples/simple_chat.py --backend echo
python examples/tools_structured.py --backend echo
```

## Installation

```bash
pip install local-llm-kit                                   # core only (light)
pip install "local-llm-kit[transformers]"                   # HF Transformers
pip install "local-llm-kit[airllm]"                        # optional AirLLM adapter
pip install "local-llm-kit[llamacpp]"                       # llama.cpp (GGUF)
pip install "local-llm-kit[ollama]"                         # Ollama convenience client
pip install "local-llm-kit[server]"                         # `serve` command (FastAPI/uvicorn)
pip install "local-llm-kit[embeddings]"                     # sentence-transformers, numpy
pip install "local-llm-kit[rag]"                            # numpy, faiss-cpu, pypdf
pip install "local-llm-kit[all]"                            # everything
```

Requires Python 3.9+. The generic `openai-compat` backend (vLLM, LM Studio, llama-server) needs no extra — it speaks HTTP with the standard library.

## Backend comparison

| Backend | `backend=` value | What it runs | Needs | Best for | Vision | Embeddings |
|---|---|---|---|---|---|---|
| Echo (mock) | `"echo"` | offline deterministic echo, no model | nothing | tests, CI, docs, offline dev | passes messages through | deterministic stub |
| Transformers | `"transformers"` | HF models via `transformers`+`torch` | `pip install "local-llm-kit[transformers]"` | HF text models on GPU/CPU | text adapter | separate embeddings client |
| AirLLM (experimental adapter) | `"airllm"` | AirLLM offloaded inference | `pip install "local-llm-kit[airllm]"`, prepared weights and suitable hardware | fitting larger models into limited accelerator memory | text only in this adapter | no |
| llama.cpp | `"llamacpp"` | `.gguf` via `llama-cpp-python` | `pip install "local-llm-kit[llamacpp]"` | quantized local text models | use HTTP server for vision | use HTTP server for embeddings |
| Ollama | `"ollama"` | models served by `ollama serve` | Ollama daemon | 5-min local setup | model-dependent | `nomic-embed-text` etc. |
| OpenAI-compatible | `"openai-compat"` | any `POST /v1/chat/completions` server (vLLM, LM Studio, Ollama, llama.cpp server) | server URL | pointing at an existing server | model-dependent | model-dependent |

Auto-detect: `http(s)` model path → `openai-compat`, `ollama:` prefix → `ollama`, `.gguf`/GGML path → `llamacpp`, anything else → `transformers`. Pass `backend=` explicitly for `echo`.

### Local server presets

Use a model ID that the running server actually serves. These presets use the shared OpenAI-compatible adapter; `backend_kwargs={"base_url": "http://host:port/v1"}` overrides the URL.

| Server | `backend=` | Connection preset |
|---|---|---|
| SGLang | `"sglang"` | `http://localhost:30000/v1` |
| vLLM | `"vllm"` | `http://localhost:8000/v1` |
| LM Studio | `"lmstudio"` | `http://localhost:1234/v1` |
| LocalAI | `"localai"` | `http://localhost:8080/v1` |
| llama-server | `"llama-server"` | `http://localhost:8080/v1`; launch with `--port 8080` |
| Jan | `"jan"` | `http://127.0.0.1:1337/v1`; enable its local server and supply its API key |
| GPT4All | `"gpt4all"` | `http://127.0.0.1:4891/v1`; enable Local API Server in settings |
| MLX-LM | `"mlx-lm"` | `http://127.0.0.1:8080/v1`; run `mlx_lm.server` on Apple Silicon |
| KoboldCpp | `"koboldcpp"` | `http://localhost:5001/v1` |

Together with native Ollama, Transformers and AirLLM, these cover **12 distinct local runtimes** (llama.cpp has both in-process and server paths). This is a compatibility selection, not a measured popularity ranking. GPT4All and MLX-LM presets provide text/chat; use a separate embedding backend. Native tools, schemas and vision require explicit upstream/model support. See the [capability matrix and setup references](https://github.com/1Utkarsh1/Local-LLM-kit/blob/v0.2.0/docs/source/backends.rst) before choosing a runtime.

```python
from local_llm_kit import LLM

llm = LLM(model_path="your-served-model-id", backend="sglang")
result = llm.chat([{"role": "user", "content": "Hello!"}])
print(result["choices"][0]["message"]["content"])
```

Native chat forwards messages, tools, JSON schemas, seed and stop sequences to Ollama and OpenAI-compatible servers. Tool calls returned outside the text are preserved, including fragmented streamed arguments. Tool and schema support depends on the server configuration and selected model. Ollama's native API offers automatic tool selection; named choices narrow the available tools, while `required` is not guaranteed by that API.

Compatibility tests exercise adapter contracts and a local HTTP fixture. They do not certify every server release, model, accelerator or performance level. Setup links and feature limits are in the [backend guide](https://github.com/1Utkarsh1/Local-LLM-kit/blob/v0.2.0/docs/source/backends.rst).

### AirLLM

```python
from local_llm_kit import LLM

llm = LLM(
    model_path="/path/to/trusted/local/checkpoint",
    backend="airllm",
    backend_kwargs={"device": "cuda:0", "max_seq_len": 2048},
)
response = llm.complete("Explain local inference briefly.", max_tokens=64)
print(response["choices"][0]["text"])
```

The AirLLM adapter is **experimental**: its PyTorch and macOS MLX call contracts have offline tests; real model/GPU inference has not been certified here. Generation is serialized per cached model because offloaded layers are stateful. Streaming returns one completed chunk, and per-call seeds/logprobs are unsupported. The MLX path exposes temperature sampling only. AirLLM may need significant checkpoint/shard disk space and may load model-supplied code; use trusted checkpoints and follow [AirLLM's setup instructions](https://github.com/lyogavin/airllm).

For offline inference, prepare weights and dependencies first, use local checkpoint paths, and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when using Hugging Face loaders. The kit does not install or launch external servers. A server preset connects to the endpoint you configure.

```python
from local_llm_kit import LLM

LLM(model_path="mistralai/Mistral-7B-Instruct-v0.1", backend="transformers")
LLM(model_path="./models/mistral-7b-q4.gguf", backend="llamacpp")
LLM(model_path="llama3.2:3b", backend="ollama")
LLM(model_path="any-name", backend="openai-compat",
    backend_kwargs={"base_url": "http://localhost:8080/v1"})
LLM(model_path="echo", backend="echo")  # offline
```

## OpenAI-SDK drop-in (talk to any OpenAI client)

Start the bundled server, then use the stock `openai` package unchanged:

```bash
local-llm-kit serve --model llama3.2:3b --backend ollama --port 8000
# offline smoke test: local-llm-kit serve --backend echo --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

See `examples/openai_compatible.py` (falls back to direct `LLM(backend="echo")` when no server / no `openai` package is present, so it runs offline).

Supported routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (incl. `stream=true` SSE), `POST /v1/completions`, `POST /v1/embeddings`.

> **Security note:** the server returns tool calls **without executing them** unless a request sets `"execute_tools": true`. Only enable that for callers you trust — auto-executing model-chosen functions is arbitrary-code-execution-as-a-feature if your tools are powerful.

## Server snippet

```bash
# GGUF file, OpenAI-compatible endpoints on :8000
local-llm-kit serve --model ./models/mistral-7b-q4.gguf --backend llamacpp --port 8000

# Or backed by an upstream OpenAI-compatible server
local-llm-kit serve --backend openai-compat --port 8000
```

```python
# Programmatic (requires the `server` extra)
from local_llm_kit.server import create_app, run

app = create_app(model_path="llama3.2:3b", backend="ollama")
run(app, port=8000)
```

## Modern tools snippet (`@tool` + `tools`/`tool_choice`)

Legacy `add_function` / `functions=` / `function_call=` still work. New code should use this:

```python
from local_llm_kit import LLM, tool

@tool(description="Get the current weather for a city")
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the weather for a location."""
    return {"temperature": 22, "unit": unit, "location": location}

llm = LLM(model_path="echo", backend="echo")
llm.add_tool(get_weather)  # legacy: llm.add_function(name, schema, impl) also works

resp = llm.chat(
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=[get_weather.spec],
    tool_choice="auto",
)
print(resp["choices"][0]["message"])
```

Full runnable version: `examples/tools_structured.py`.

## Structured output snippet (`response_format`, JSON Schema)

```python
from local_llm_kit import LLM

llm = LLM(model_path="echo", backend="echo")
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["name", "population"],
}
try:
    resp = llm.chat(
        messages=[{"role": "user", "content": "Describe Paris."}],
        response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
    )
except TypeError:  # older client fallback
    resp = llm.chat(
        messages=[{"role": "user", "content": "Describe Paris."}],
        format="json",
    )
print(resp["choices"][0]["message"]["content"])
```

## Minimal RAG snippet

```python
from local_llm_kit import LLM
from local_llm_kit.rag import SimpleVectorStore, build_rag_prompt

# Lexical mode: no embeddings needed, keyword search works offline.
# Pass embed_fn=... (EmbeddingClient, Ollama, OpenAI-compatible endpoint)
# for semantic retrieval instead.
store = SimpleVectorStore()
store.add(
    ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
    metadatas=[{"source": "doc-0"}, {"source": "doc-1"}],
)
hits = store.search("Where is the Eiffel Tower?", k=2)
messages = build_rag_prompt("Where is the Eiffel Tower?", hits)

llm = LLM(model_path="echo", backend="echo")
resp = llm.chat(messages=messages)
print(resp["choices"][0]["message"]["content"])
```

Full runnable version: `examples/rag_minimal.py`.

## Vision snippet (graceful degradation)

```python
from local_llm_kit import LLM

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ],
}]

for backend in ("ollama", "openai-compat", "echo"):
    try:
        llm = LLM(model_path="llava  # or model your server hosts", backend=backend)
        print(llm.chat(messages=messages)["choices"][0]["message"]["content"])
        break
    except Exception as e:
        print(f"{backend} unavailable, falling back ({e})")
```

Text-only backends receive the `text` parts and ignore images instead of crashing. Runnable demo: `examples/vision_chat.py`.

## CLI

```bash
local-llm-kit chat --model llama3.2:3b --backend ollama --system "You are helpful."
local-llm-kit complete --model ./m.gguf --backend llamacpp --prompt "Once upon a time,"
local-llm-kit serve --backend echo --port 8000
local-llm-kit pull owner/repo:file.gguf --output-dir ./models
local-llm-kit pull owner/repo --filename file.gguf --revision main --force
local-llm-kit chat --backend sglang --model your-served-model-id -q "Hello"
local-llm-kit list
local-llm-kit embed --model nomic-embed-text --backend ollama --text "hello world"
```

All `chat`/`complete`/`embed` commands accept `--backend echo|transformers|llamacpp|ollama|openai-compat`. Run any command with `--help` for flags.

Server presets and `airllm` are also accepted by the CLI. `pull` reuses repository/revision-specific and URL-specific caches; `--force` refreshes a download. Failed transfers preserve existing complete files. URL cache paths now include a URL hash, so older flat URL caches may be downloaded once again.

## Examples

| File | Shows | Offline? |
|---|---|---|
| `examples/simple_chat.py` | streaming chat loop | `--backend echo` |
| `examples/openai_compatible.py` | stock OpenAI client vs local server | yes (falls back to `echo`) |
| `examples/tools_structured.py` | `@tool` + `response_format` | yes |
| `examples/rag_minimal.py` | chunk → store → retrieve → chat | yes |
| `examples/vision_chat.py` | `image_url` message, graceful fallback | yes |

## Migration: 0.1.x → 0.2.0

Backwards compatible: `LLM` / `chat` / `complete` / `add_function` keep working.

| 0.1.x | 0.2.0 (preferred) | Notes |
|---|---|---|
| `functions=[...]` | `tools=[...]` | `functions` still accepted, normalized internally |
| `function_call="auto"` | `tool_choice="auto"` | same |
| `format="json"` | `response_format={"type": "json_object"}` or `json_schema` | `format` still accepted as shorthand |
| backends `transformers`/`llamacpp` | + `echo`, `ollama`, `openai-compat` | `echo` needs no deps |
| no server | `local-llm-kit serve` + `local_llm_kit.server` | needs `server` extra |
| no RAG | `local_llm_kit.rag` + `local_llm_kit.embeddings` | stdlib core, numpy optional |
| images unsupported | `content: [{"type":"image_url",...}]` | text fallback on text-only models |

Rename checklist: replace `functions=` with `tools=` (wrap each spec as `{"type":"function","function":spec}` — or just pass `@tool` functions, the helper does it), replace `format="json"` with `response_format`, pin new extras in your requirements.

## License / Contributing

MIT. Contributions welcome — see `CONTRIBUTING.md`.
