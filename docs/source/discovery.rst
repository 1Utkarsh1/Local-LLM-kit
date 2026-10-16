Finding local servers and reading their real capabilities
=========================================================

Local AI tooling usually fails for a boring reason: nobody knows which endpoint
is actually up. Desktop apps pick their own ports, servers take minutes to load
a model, and an OpenAI-compatible listing rarely says what the model can do.

``local_llm_kit.discovery`` answers three questions with the standard library
only:

1. **What is listening?** Every well-known local endpoint is probed
   concurrently, and unreachable ports are reported instead of raised.
2. **What can it do?** Each server's *own* metadata API is read where one
   exists — Ollama's ``/api/show``, llama.cpp's ``/props``, LM Studio's
   ``/api/v1/models``, apfel's ``/v1/models``.
3. **Is it really on this machine?** Models that a server routes elsewhere (an
   Ollama ``:cloud`` model runs on ollama.com) are flagged, so "offline" means
   offline.

The command line
----------------

.. code-block:: bash

   local-llm-kit doctor
   local-llm-kit doctor --json
   local-llm-kit doctor --require-local
   local-llm-kit doctor --url localhost:9000
   local-llm-kit doctor --quiet          # exit code only

``doctor``, ``discover`` and ``probe`` are aliases.

Exit codes: ``0`` when at least one server responded (with ``--require-local``,
when at least one offers an on-device model), ``1`` otherwise. That makes it
usable as a preflight check in scripts.

The Python API
--------------

.. code-block:: python

   from local_llm_kit import discover, find_local_server, probe_endpoint

   report = discover()                     # DiscoveryReport
   for server in report.live:
       print(server.base_url, server.provider, server.latency_ms)
       for model in server.models:
           print("  ", model.id, model.context_length, model.capabilities,
                 "REMOTE" if model.remote else "on-device")

   server = find_local_server()            # first on-device-capable server, or None
   one = probe_endpoint("http://localhost:1234/v1")   # a single endpoint

``discover`` accepts ``extra_urls``, ``timeout`` (default 0.75s), ``enrich``
(capability lookup) and ``max_workers``. ``probe_endpoint`` accepts anything
:func:`local_llm_kit.discovery.normalize_base_url` understands, so
``localhost:11434``, ``http://host:1234/v1`` and a full
``.../v1/chat/completions`` URL all work.

Probing only ever issues unauthenticated ``GET`` requests to model-listing and
metadata endpoints. Nothing is downloaded, no inference runs, and no credentials
are sent.

Endpoint catalog
----------------

:data:`local_llm_kit.discovery.ENDPOINTS` covers, in probe order:

.. list-table::
   :header-rows: 1

   * - Provider
     - Default endpoint
   * - Ollama
     - ``http://127.0.0.1:11434/v1``
   * - LM Studio
     - ``http://127.0.0.1:1234/v1``
   * - llama.cpp ``llama-server``
     - ``http://127.0.0.1:9931/v1`` (default since October 2026) and ``:8080``
   * - vLLM
     - ``http://127.0.0.1:8000/v1``
   * - LocalAI
     - ``http://127.0.0.1:8080/v1``
   * - SGLang
     - ``http://127.0.0.1:30000/v1``
   * - Jan
     - ``http://127.0.0.1:1337/v1``
   * - GPT4All
     - ``http://127.0.0.1:4891/v1``
   * - MLX-LM
     - ``http://127.0.0.1:8080/v1``
   * - KoboldCpp
     - ``http://127.0.0.1:5001/v1``
   * - text-generation-webui
     - ``http://127.0.0.1:5000/v1``
   * - apfel (Apple Foundation Models)
     - ``http://127.0.0.1:11434/v1``
   * - llamafile
     - ``http://127.0.0.1:8080/v1``

Several providers share a default port, so the catalog is de-duplicated by
normalised URL before probing; the provider is then **identified from the
server's own response**, not from the port number. When a response carries no
distinguishing field (LM Studio reports a generic owner), the provider's marker
endpoint is tried.

Capability sources
------------------

.. list-table::
   :header-rows: 1

   * - Provider
     - Endpoint
     - Reported
   * - Ollama
     - ``POST /api/show``, ``GET /api/tags``
     - real ``context_length``, ``capabilities``, ``remote_host``
   * - llama.cpp
     - ``GET /props``, ``GET /v1/models``
     - live ``n_ctx``, ``modalities.vision``, model ``meta``
   * - LM Studio
     - ``GET /api/v1/models``
     - ``capabilities.vision``, ``trained_for_tool_use``, ``max_context_length``
   * - apfel
     - ``GET /v1/models``
     - ``context_window``, ``capabilities``

Adapters use the same information. Ollama's backend reads ``/api/show`` when
``auto_detect_capabilities=True`` is passed in ``backend_kwargs``, and always
reports the extra fields on ``list_models()``:

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM("qwen2.5:0.5b", backend="ollama",
             backend_kwargs={"auto_detect_capabilities": True})
   llm.context_window           # 32768 for qwen2.5:0.5b, read from the server
   llm.backend.capabilities     # ['completion', 'tools']
   llm.backend.supports_vision  # False - the server does not list vision

Detection is opt-in there so that constructing a backend never requires a
running server, and a failed lookup leaves the configured fallback in place.

Keeping inference on device
---------------------------

.. code-block:: python

   llm = LLM("qwen2.5:0.5b", backend="ollama", require_local=True)  # accepted

   LLM("glm-5.2:cloud", backend="ollama", require_local=True)
   # RuntimeError: Model 'glm-5.2:cloud' is served from https://ollama.com:443,
   #               not on this machine. ...

``require_local`` consults the backend at construction. If the backend *can*
report routing and the lookup fails, it raises rather than assuming the model is
local — an unverifiable on-device guarantee is not granted silently. Backends
with no notion of remote routing treat the flag as a no-op.

Testing scope
-------------

``tests/test_discovery.py`` and the ``doctor`` CLI tests run entirely offline
against a local HTTP fixture that reproduces the documented response shapes.
``tests/test_live_servers.py`` certifies the same behaviour against a real server
and is opt-in:

.. code-block:: bash

   LOCAL_LLM_KIT_LIVE=1 python -m pytest tests/test_live_servers.py -q

The live suite skips any check the machine cannot satisfy, never uses
provider-routed models (they leave the device, so they cannot certify local
inference), and prints what it verified. Fixture tests are protocol validation;
they are not a substitute for the live run.

Waiting for a server that is not ready yet
------------------------------------------

Discovery answers "is anything there?". The retry policy answers the follow-up:
"it is starting — wait for it". A refused connection or a transient status
(``429``, ``5xx``) is retried; a client error such as ``400`` is not. The
defaults match the official OpenAI Python client — two retries, 0.5s initial
delay doubling to an 8s ceiling, jittered, with ``Retry-After`` honoured and
clamped to 120s.

.. code-block:: python

   from local_llm_kit import LLM
   from local_llm_kit.retry import RetryPolicy

   LLM("m", backend="vllm")                                     # 2 retries (default)
   LLM("m", backend="vllm", backend_kwargs={"max_retries": 0})  # fail fast
   LLM("m", backend="vllm", backend_kwargs={"max_retries": 5})  # more patience
   LLM("m", backend="vllm", backend_kwargs={
       "max_retries": RetryPolicy(max_retries=4, initial_delay=1.0, max_delay=15.0)
   })

Measured locally, with the server starting 0.5s after the first request:
``max_retries=0`` raises ``ConnectionError`` after 0.01s, while the default
policy succeeds in 1.22s.

Streaming retries only the connection phase. Once a frame has been yielded,
retrying would replay tokens the caller already received, so the failure is
raised instead. Capability detection is deliberately single-attempt, so
``doctor`` and ``discover`` never sleep.
