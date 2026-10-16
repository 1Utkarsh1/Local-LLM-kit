Local backends and server compatibility
=======================================

The core client is stdlib-based. Native model packages are optional and loaded
only when selected. Server integrations connect to an existing process; they do
not install, launch or reconfigure it.

Runtime coverage and feature limits
-----------------------------------

These 14 runtimes cover desktop apps, local files, Apple Silicon, GPU servers
and offloaded inference. This is a compatibility selection, not a top-ten
ranking: no comparable active-user measurement has been established.
llama.cpp's library and server integrations count as one runtime.

.. list-table:: Capability matrix for this client
   :header-rows: 1
   :widths: 15 25 40 20

   * - Runtime
     - Integration
     - Tools and structured output
     - Embeddings
   * - Apple FM (native)
     - In-process SDK (``apple-fm``)
     - Prompted tools/JSON loop; native ``json_schema`` enforcement
     - data-URL images via temp files; no on-device embedding endpoint
   * - Apfel bridge
     - OpenAI-compatible HTTP
     - Forwarded; native tools + ``json_schema`` server-side
     - No endpoint in this preset
   * - Ollama
     - Native HTTP
     - Native tools/schema forwarding; model-dependent
     - Dedicated embedding model
   * - llama.cpp
     - In-process GGUF or llama-server
     - Prompted in-process; native server features depend on configuration
     - Server with embedding model; no in-process adapter method
   * - LM Studio
     - OpenAI-compatible HTTP
     - Forwarded; server/model-dependent
     - Load an embedding model
   * - vLLM
     - OpenAI-compatible HTTP
     - Requires suitable tool parser/chat template and schema support
     - Requires an embedding/pooling model
   * - SGLang
     - OpenAI-compatible HTTP
     - Requires suitable tool parser/chat template and schema support
     - Requires embedding configuration/model
   * - Transformers
     - In-process Python
     - Prompted tools/JSON; no schema enforcement
     - Use the separate embeddings extra/client
   * - LocalAI
     - OpenAI-compatible HTTP
     - Depends on configured backend and model
     - Configure an embedding model
   * - GPT4All
     - Desktop local API
     - Text/chat baseline; native tools/schema unverified
     - No endpoint in this preset
   * - Jan
     - Desktop local API
     - Native tools require a supporting model; schemas unverified
     - Unverified; use a separate embedding backend
   * - AirLLM
     - Experimental in-process Python
     - Prompted tools/JSON; buffered stream, no schema enforcement
     - No adapter method
   * - MLX-LM
     - Apple Silicon HTTP server
     - Text/chat baseline; additional features depend on model/server
     - No endpoint in this preset
   * - KoboldCpp
     - OpenAI-compatible HTTP
     - Forwarded; model/template-dependent
     - Requires a loaded embedding model

All server presets have client routing and payload and HTTP/SSE fixture tests.
This is protocol validation, not a real-engine or hardware certification.
Unsupported native tool/schema options may be rejected by the upstream server;
the client does not silently claim those features were enforced. Native vision
is forwarded only through the HTTP adapters. In-process adapters currently
format text; use an HTTP vision model for image inputs.

All server presets have client routing, payload and HTTP/SSE fixture tests.
This is protocol validation, not a real-engine or hardware certification.
Unsupported native tool/schema options may be rejected by the upstream server;
the client does not silently claim those features were enforced. Native vision
is forwarded only through the HTTP adapters. In-process adapters currently
format text; use an HTTP vision model for image inputs.

Apple Foundation Models (native backend + apfel bridge)
-------------------------------------------------------

Apple Silicon Macs ship an on-device model via the Foundation Models
framework. Local LLM Kit reaches it two ways:

* **Native** (``backend="apple-fm"``): Apple's official
  `python-apple-fm-sdk <https://github.com/apple/python-apple-fm-sdk>`_
  (``pip install "local-llm-kit[apple-fm]"``). Requires macOS 26+, Xcode 26+
  with the SDK agreement accepted, Python 3.10+, and Apple Intelligence
  turned on (`support article <https://support.apple.com/en-us/121115>`_).
  Sessions are created per request (backend instances are shared through
  the client cache), tool calls use the uniform client-side loop, and
  ``response_format={"type": "json_schema", ...}`` is enforced natively by
  the SDK. ``temperature`` (``0`` selects greedy sampling), ``seed``,
  ``max_tokens`` and ``stop`` are honoured; ``top_p``/``top_k``/penalties
  and logprobs have no SDK equivalent and are ignored. data-URL
  ``image_url`` parts are materialised to temporary files and passed as
  ``ImageAttachment`` objects (remote URLs are rejected, never downloaded);
  image understanding itself needs the macOS 27 SDK. ``count_tokens_exact``
  offers exact budgeting via the SDK's async ``token_count``. There is no
  on-device embedding endpoint. Everything runs on device; nothing falls
  back to Private Cloud Compute or any remote endpoint.
* **Bridge** (``backend="apfel"``): no Xcode needed. Run
  `apfel --serve <https://github.com/Arthur-Ficial/apfel>`_ (macOS 26+,
  Apple Silicon, Apple Intelligence on), which serves OpenAI-compatible
  ``POST /v1/chat/completions`` on ``http://localhost:11434/v1``. The
  preset strips the sampling fields apfel rejects (``top_k``/penalties),
  rejects prompt-completion calls (apfel answers 501 there; use ``chat()``),
  and reports no embedding support (apfel answers 501 there too).

``python examples/apple_foundation.py`` demonstrates both paths and prints
setup steps when the model or server is unavailable. Contracts are covered
by offline tests with an SDK double
(``tests/test_apple_fm.py``); the opt-in live test
(``LOCAL_LLM_KIT_APPLE_FM_LIVE=1``) needs an eligible Mac and is skipped in
CI. See also `simonw/llm-apple-foundation
<https://github.com/simonw/llm-apple-foundation>`_ for the same SDK used
through the ``llm`` CLI, which informed the sampling/schema mapping here.

Server presets
--------------

.. list-table:: OpenAI-compatible connection presets
   :header-rows: 1
   :widths: 20 20 60

   * - Runtime
     - backend
     - URL preset
   * - SGLang
     - sglang
     - http://localhost:30000/v1
   * - vLLM
     - vllm
     - http://localhost:8000/v1
   * - LM Studio
     - lmstudio
     - http://localhost:1234/v1
   * - LocalAI
     - localai
     - http://localhost:8080/v1
   * - llama-server
     - llama-server
     - http://localhost:8080/v1 (start with --port 8080)
   * - Jan
     - jan
     - http://127.0.0.1:1337/v1 (enable server and configure its API key)
   * - GPT4All
     - gpt4all
     - http://127.0.0.1:4891/v1 (enable Local API Server in settings)
   * - MLX-LM
     - mlx-lm
     - http://127.0.0.1:8080/v1 (run mlx_lm.server)
   * - KoboldCpp
     - koboldcpp
     - http://localhost:5001/v1
   * - Apfel bridge
     - apfel
     - http://localhost:11434/v1 (run ``apfel --serve``; shares Ollama's port)

Use the ID of a model already loaded by the server. A preset chooses the
connection URL and protocol parameter mapping. Override the URL with
``backend_kwargs={"base_url": "http://host:port/v1"}``; set ``api_key`` there if
needed. The generic ``openai-compat`` backend supports other servers exposing the
same endpoints. Server and model capabilities determine tool calling, schemas,
vision and embedding availability.

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM("your-served-model-id", backend="sglang")
   result = llm.chat([{"role": "user", "content": "Hello"}])

Upstream setup references:

* `SGLang OpenAI-compatible API <https://docs.sglang.io/docs/basic_usage/openai_api>`_
* `vLLM OpenAI-compatible server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_
* `LM Studio local server <https://lmstudio.ai/docs/developer/core/server>`_
* `LocalAI setup <https://localai.io/docs/>`_
* `llama-server setup <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>`_
* `Jan local API server <https://docs.jan.ai/desktop/api-server>`_
* `GPT4All local API server <https://docs.gpt4all.io/gpt4all_api_server/home.html>`_
* `MLX-LM project <https://github.com/ml-explore/mlx-lm>`_ and its
  `server endpoints/options <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py>`_
* `KoboldCpp API and setup <https://github.com/LostRuins/koboldcpp#apis-and-integrations>`_
* `apfel OpenAI API compatibility <https://github.com/Arthur-Ficial/apfel#openai-api-compatibility>`_

For example, launch MLX-LM with prepared weights using
``mlx_lm.server --model /path/to/local/mlx-model --host 127.0.0.1 --port 8080``,
then select ``backend="mlx-lm"``. Jan's ``api_key`` goes in ``backend_kwargs``.
GPT4All accepts the model's display name from its local API; other servers may
use a file path, repository name or an explicitly configured alias.

Ollama
------

Use ``backend="ollama"`` with the name of a model installed in Ollama. Override
its address with ``backend_kwargs={"host": "http://localhost:11434"}``.
The CLI's ``--base-url`` maps to this ``host`` option. Native tool arguments and
results are converted between OpenAI and Ollama message shapes. Thinking fields
needed for tool follow-ups are retained internally.
OpenAI image content parts must use inline ``data:image/...;base64,...`` URLs;
they are converted to Ollama's native ``images`` list without downloading URLs.

Ollama's native endpoint has no enforced ``tool_choice`` field. ``none`` omits
tools; a named choice narrows the offered tools. ``required`` remains
model-dependent. See `Ollama tool calling <https://docs.ollama.com/capabilities/tool-calling>`_
and `structured output <https://docs.ollama.com/capabilities/structured-outputs>`_.

AirLLM: experimental in-process adapter
---------------------------------------

.. code-block:: bash

   pip install -e ".[airllm]"

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM(
       "/path/to/trusted/local/checkpoint",
       backend="airllm",
       backend_kwargs={"device": "cuda:0", "max_seq_len": 2048},
   )
   response = llm.complete("Explain local inference.", max_tokens=64)

The adapter uses AirLLM's PyTorch generation contract or its macOS MLX contract.
Calls are serialized per backend because offloaded model layers are stateful.
Context overflow produces an explicit error rather than truncating the prompt.
Streaming is buffered and yields one completed backend chunk. Per-call seeds and
logprobs are unsupported. The MLX path provides temperature sampling; other
sampling controls depend on the PyTorch path. This is a text adapter, with no
native embedding or vision implementation.

Offline unit tests cover loading, device forwarding, output shapes, stopping,
context limits and both upstream call contracts with doubles. Real checkpoint,
CUDA and MLX execution have not been certified. AirLLM models may execute
model-supplied code and need substantial disk space for prepared layer shards.
Use trusted checkpoints, and follow `AirLLM's hardware and installation guide
<https://github.com/lyogavin/airllm>`_.

Transformers and llama.cpp
--------------------------

Use ``backend="transformers"`` with the ``transformers`` extra for Hugging Face
models, or ``backend="llamacpp"`` with the ``llamacpp`` extra for local GGUF files.
Tool calls and JSON output use the kit's prompt formatting unless the backend
implements a native capability. Prompted JSON is not a schema-enforcement guarantee.

Offline operation and validation scope
--------------------------------------

Prepare dependencies and weights while connected, then use local checkpoint
paths. Set ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` for Hugging Face
loaders. External engines have their own offline/cache configuration. A local
URL keeps client requests local, while upstream engine configuration controls
whether it downloads anything else.

Tests verify the shared HTTP adapter with a local fixture and mock native tool,
streaming and schema responses. This establishes client/protocol behaviour;
it is not a hardware benchmark or an end-to-end certification of every engine
version and model combination.
