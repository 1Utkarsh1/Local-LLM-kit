# Local LLM Kit

[![PyPI](https://img.shields.io/pypi/v/local-llm-kit.svg)](https://pypi.org/project/local-llm-kit/)
[![Python](https://img.shields.io/pypi/pyversions/local-llm-kit.svg)](https://pypi.org/project/local-llm-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

One Python API for local language models, desktop apps, and inference servers.
Use OpenAI-shaped chat, streaming, tools and structured output while choosing
where your model runs. The core install needs only `typing-extensions`; model
engines and validation packages are optional.

**Release 0.3.0** includes native Apple FM,
MCP, discovery, Responses, agents, hybrid retrieval, logging and the hardware
advisor. Install or upgrade from PyPI:

```bash
python -m pip install --upgrade local-llm-kit
```

For the released API, use the [0.3.0 documentation](https://github.com/1Utkarsh1/Local-LLM-kit/tree/v0.3.0).
No API key is needed for local inference. Prepare your engine, dependencies and
model weights before disconnecting from the internet.

## Start with a local model

Start [Ollama](https://github.com/ollama/ollama), then download a model once:

```bash
ollama pull qwen2.5:0.5b
pip install local-llm-kit
local-llm-kit chat --backend ollama -m qwen2.5:0.5b --message 'Hello'
```

```python
from local_llm_kit import LLM

llm = LLM('qwen2.5:0.5b', backend='ollama')
response = llm.chat([{'role': 'user', 'content': 'Explain recursion briefly.'}])
print(response['choices'][0]['message']['content'])

for chunk in llm.chat([{'role': 'user', 'content': 'Write a haiku.'}], stream=True):
    print(chunk['choices'][0]['delta'].get('content') or '', end='')
```

For tests without a model or a server:

```python
llm = LLM('echo', backend='echo')
assert llm.chat([{'role': 'user', 'content': 'Hello'}])['choices'][0]['message']['content'].startswith('Echo:')
```

## Choose your runtime

Server presets connect to an engine you have already started; they do not install
or launch it. Always use that server's served model ID. You can override any URL
with `backend_kwargs={'base_url': 'http://127.0.0.1:PORT/v1'}`.

| Runtime | `backend` | Connection or extra |
|---|---|---|
| Ollama | `ollama` | Native HTTP; default port 11434 |
| llama.cpp | `llamacpp` / `llama-server` | Local GGUF with `[llamacpp]`, or server port 9931 |
| LM Studio | `lmstudio` | Server port 1234 |
| vLLM | `vllm` | Server port 8000 |
| SGLang | `sglang` | Server port 30000 |
| LocalAI | `localai` | Server port 8080 |
| Jan | `jan` | Server port 1337 |
| GPT4All | `gpt4all` | Server port 4891 |
| MLX-LM | `mlx-lm` | Server port 8080 |
| KoboldCpp | `koboldcpp` | Server port 5001 |
| Hugging Face Transformers | `transformers` / `transformers-serve` | In-process `[transformers]`, or server port 8000 |
| AirLLM | `airllm` | Experimental in-process `[airllm]`; buffered streaming |
| Apple Foundation Models | `apple-fm` | Native `[apple-fm]`, eligible Mac and SDK required |
| apfel | `apfel` | Apple model bridge; server port 11434 |
| Other OpenAI-compatible servers | `openai-compat` | Explicit `base_url` |

This is a compatibility list. Popularity and speed depend on workload and
hardware. Tools, vision, embeddings and native schema enforcement depend on the
actual engine, model, parser and configuration. GPT4All, MLX-LM and apfel presets
have no embedding endpoint in this adapter. llama-server's current binary uses
9931; older versions and its Docker image may use 8080. See the
[compatibility guide](docs/source/backends.rst) for limits and upstream references.

## Find running servers and check locality

```bash
local-llm-kit doctor
local-llm-kit doctor --json --url http://127.0.0.1:9000/v1
local-llm-kit doctor --require-local
```

```python
from local_llm_kit import discover, find_local_server

report = discover()
for server in report.live:
    print(server.provider, server.base_url, [m.to_dict() for m in server.models])

server = find_local_server(require_local=True)
```

Discovery reads available native metadata from Ollama, llama.cpp, LM Studio and
apfel. A reachable loopback URL or a local-sounding model name alone does not
prove local inference: unknown locality stays unknown. Ollama cloud routes are
marked remote. `LLM(..., require_local=True)` accepts known in-process engines
or a loopback server with verified model metadata, and rechecks before chat and
completion requests. This trusts server metadata; it is not a network sandbox
and does not prevent a model engine from downloading weights or a tool from
using the network.

HTTP adapters retry connection failures and selected transient statuses with
bounded backoff. Configure `backend_kwargs={'max_retries': 0}` to fail fast or
pass `local_llm_kit.retry.RetryPolicy` for delay controls. Streaming retries only
the connection setup, avoiding repeated tokens after output starts.

## Use Apple's built-in model

```bash
pip install 'local-llm-kit[apple-fm]'
```

```python
llm = LLM('apple-foundationmodel', backend='apple-fm',
          backend_kwargs={'timeout': 30})
print(llm.chat([{'role': 'user', 'content': 'Summarise this idea.'}]))
```

The native backend needs Python 3.10+, macOS 26+, Xcode 26+ with its SDK agreement
accepted, and an Apple Intelligence-capable Mac with Apple Intelligence enabled.
It creates a fresh SDK session per request, forwards native JSON Schema, supports
streaming stops across chunk boundaries and exposes `count_tokens_exact`.
Inline data-URL images use temporary files; actual vision requires an SDK/OS
version with image support. Unsupported sampling controls are documented.
Use `LLM.achat()` from async code.

Alternatively, run [apfel --serve](https://github.com/Arthur-Ficial/apfel) and
select `backend='apfel'`. This profile omits unsupported sampling fields, maps
prompt completion to chat, and reports no embedding support. It shares Ollama's
default port; configure separate ports to run both simultaneously.

## Python tools, MCP and agents

```python
from local_llm_kit import Agent, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm.add_tool(add)
reply = llm.chat([{'role': 'user', 'content': 'What is 12 + 30?'}])

# Or let the bounded agent own the loop with tools scoped to that agent:
agent = Agent(llm, tools=[add], max_steps=5, tool_timeout=10)
result = agent.run('What is 12 + 30?')
print(result.content, result.success)
```

Tool execution uses complete parsed arguments, names scoped to the request,
required-argument checks and a waiting timeout. Invalid or failed calls become
error feedback; truncated arguments are not repaired into executable actions.
A timed-out Python function can keep running, so tools with side effects still
need their own execution controls. A model that supports tool calling is required.
Use `execute_tools=False` to return calls for your application to handle.

Connect a trusted MCP server with the same tools interface:

```python
from local_llm_kit import MCPServer

with LLM('qwen2.5:7b', backend='ollama') as llm:
    llm.add_mcp_server('files', command='your-mcp-server', args=['/allowed/path'])
    # Tools are namespaced files__<name>. llm.close() closes the connection.
    reply = llm.chat([{'role': 'user', 'content': 'List the available files.'}])
```

```bash
local-llm-kit mcp tools --command 'your-mcp-server /allowed/path' --json
local-llm-kit mcp call --url http://127.0.0.1:9000/mcp --tool read --args '{"path":"note.txt"}'
```

The synchronous MCP tools client supports stdio and Streamable HTTP (JSON/SSE),
deadlines, pagination, namespacing and cleanup. Resources, prompts, OAuth and
server-initiated subscriptions are outside this subset. A local subprocess may
itself use the network; transport alone makes no offline guarantee.

## Validate structured output

```bash
pip install 'local-llm-kit[structured]'
```

```python
schema = {'type': 'object', 'properties': {'age': {'type': 'integer', 'minimum': 0}},
          'required': ['age'], 'additionalProperties': False}
data = llm.chat_structured([{'role': 'user', 'content': 'Return age 3 as JSON.'}],
                           schema=schema, max_retries=2)
assert data['age'] == 3
```

Dictionary schemas use `jsonschema`; Pydantic model validation is supported when
Pydantic is installed. The helper validates types and constraints and asks for
correction within its retry limit. Native constrained decoding is used where the
backend supports it. `extract_json` and `repair_json` are optional output helpers
that preserve quoted values; they are not used to repair executable tool calls.

## Retrieval with local documents

```python
from local_llm_kit import RAGPipeline, SimpleVectorStore

pipeline = RAGPipeline(llm)
pipeline.add_documents(['Local installation uses prepared weights.'],
                       metadatas=[{'topic': 'setup'}])
answer = pipeline.query('How does local installation work?', mode='lexical')
print(answer['answer'], answer['sources'])
```

`SimpleVectorStore` supports default dense/TF-IDF search, explicit BM25, metadata
filters, deletion and JSON persistence. With an `embed_fn`, `mode='hybrid'` fuses
dense/BM25 ranks; `lambda_mult` enables MMR diversity. The small in-memory store
is intended for modest document collections. Hash embeddings in examples are
deterministic fixtures, not semantic search models. Citation prompts do not
prove that an answer is grounded; returned source records let applications check.

## Save conversations and reuse prompts

```python
from local_llm_kit import ConversationLog, PromptTemplate

prompt = PromptTemplate('Explain {{topic}} in {{style|plain language}}.').render(topic='recursion')
with ConversationLog('chat.db') as log:
    llm = LLM('qwen2.5:0.5b', backend='ollama', conversation_log=log)
    response = llm.chat([{'role': 'user', 'content': prompt}])
    print(log.export_conversation(response['conversation_id'], format='openai'))
```

Logging is opt-in. SQLite writes each completed interaction atomically, including
usage, tool IDs and full message payloads; `:memory:` works across threads.
Completed streams are logged too. An interrupted stream is not saved as a complete
answer. Stored prompts, images and tool outputs are ordinary local database data.
Templates use plain text replacement, not evaluation or automatic JSON escaping;
JSON and plain-text files need no extra, while YAML files need `[yaml]`.

```bash
local-llm-kit chat --backend echo -m echo --message Hello --conversation-log ./chat.db
local-llm-kit logs list --db ./chat.db
local-llm-kit logs export CONVERSATION_ID --db ./chat.db --format openai
local-llm-kit hardware --params-b 8 --context-length 8192
```

The hardware advisor reports measured or unknown memory and an approximate
single-sequence model/KV budget. It does not combine separate GPUs into a
fictional single device or guarantee a model will fit. Supply actual KV geometry
to `estimate_model_vram` when available.

## Connect an OpenAI SDK client

```bash
pip install 'local-llm-kit[server]' openai
local-llm-kit serve --backend ollama --model qwen2.5:0.5b --host 127.0.0.1 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='unused')
response = client.responses.create(model='qwen2.5:0.5b', input='Hello', store=False)
print(response.output_text)
```

The development server exposes chat/completions, completions, embeddings, model
listing and a **stateless Responses subset**: text/image message input,
instructions, function calls/results, `text.format`, and text/tool SSE lifecycles.
Responses returns tools for the caller to execute. Stateful responses, built-in
hosted tools, reasoning controls and background jobs are rejected. This is not
full OpenAI API parity or a hardened multi-user inference gateway. Use a native
embedding backend for semantic embeddings; the development fallback is a fixture.

## Development and validation

```bash
pip install -e '.[dev,server]' httpx openai mcp
ruff check .
black --check .
pytest
LOCAL_LLM_KIT_OPENAI_INTEROP=1 pytest tests/test_responses_sdk.py
LOCAL_LLM_KIT_MCP_INTEROP=1 pytest tests/test_mcp_interop.py
```

Python 3.9+ is supported by the core. Official SDK interoperability checks use
Python 3.10+ where their dependencies require it. Offline protocol fixtures verify
routing and payload contracts; they do not certify all engines or GPU hardware.
Native Apple live tests require `LOCAL_LLM_KIT_APPLE_FM_LIVE=1` on an eligible Mac.

See [examples](examples), the [API reference](docs/source/api_reference.rst),
[contribution guide](CONTRIBUTING.md), [changelog](CHANGELOG.md), and
[PR consolidation record](docs/source/community_pr_review.rst).
Report reproducible bugs through [GitHub Issues](https://github.com/1Utkarsh1/Local-LLM-kit/issues).
Local LLM Kit is licensed under [MIT](LICENSE).
