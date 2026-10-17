Discover servers and model capabilities
=======================================

These APIs are available in 0.3.0. ``discover()`` probes known loopback
endpoints concurrently without generating text. ``probe_endpoint(url)`` checks
one endpoint; extra URLs can be supplied to discovery. Authentication is not
handled by discovery, so an authenticated server may appear unavailable.

.. code-block:: bash

   local-llm-kit doctor
   local-llm-kit doctor --json --url http://127.0.0.1:9000/v1
   local-llm-kit doctor --require-local

The CLI returns zero when a live server is found, or when a verified local model
is found if ``--require-local`` is set. JSON includes failures and notes.

.. code-block:: python

   from local_llm_kit import discover, find_local_server, probe_endpoint

   report = discover(timeout=0.75)
   for server in report.live:
       for model in server.models:
           locality = 'remote' if model.remote else ('local' if model.local_verified else 'unknown')
           print(server.provider, model.id, model.capabilities, locality)

   server = find_local_server(require_local=True)  # a report or None
   single = probe_endpoint('http://127.0.0.1:1234/v1')

What is measured
----------------

The OpenAI model list identifies a reachable server and served IDs. When the
provider can be recognized from its response, native metadata may enrich it:

* Ollama: ``/api/tags`` and ``/api/show`` for routing, capabilities and context.
* llama.cpp: ``/props`` for model path, configured context and modality flags.
* LM Studio: native model API for capabilities and loaded instances.
* apfel: capability fields in the model list.

A connection preset or port does not establish model capabilities. Missing
metadata remains unknown. Reports identify providers on a best-effort basis;
proxies can hide or imitate another server's response.

Local execution guard
---------------------

A loopback address alone does not establish local inference. For example, an
Ollama cloud route is accessible through a local daemon but executes elsewhere.
``ServerReport.local_models`` requires loopback plus explicit provider metadata;
``local_only()`` requires every listed model to satisfy it.

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM('qwen2.5:0.5b', backend='ollama', require_local=True)
   llm.assert_local()  # raises if the route is remote or cannot be verified

Known in-process engines are accepted. Ollama requires successful native
routing metadata. Generic OpenAI-compatible servers must expose native
metadata that verifies the requested model. Unknown routing fails closed.
Checks repeat before chat/completion, including a reused client.

This is a metadata check, not a network sandbox. It trusts the local server,
does not prohibit engine downloads, and does not constrain tools. Prepare model
weights and dependencies separately for offline use. Remote URLs, unavailable
authenticated endpoints, SSH tunnels and opaque proxies cannot be assumed local.

Capability lookup and retries
------------------------------

``backend_kwargs={'auto_detect_capabilities': True}`` enables Ollama capability
lookup when constructing a client; otherwise its configured context remains a
fallback. Discovery and the explicit ``detect_backend_capabilities()`` API read
the server's metadata. Detection errors are returned with the result.

Inference HTTP adapters use a shared retry policy for selected transient
statuses and connection failures. Discovery uses short per-probe timeouts and
no inference retry/backoff. Configure inference with ``max_retries``:

.. code-block:: python

   from local_llm_kit.retry import RetryPolicy

   llm = LLM('served-model', backend='vllm', backend_kwargs={
       'max_retries': RetryPolicy(max_retries=3, initial_delay=0.5, max_delay=8),
   })

Only connection setup is retried for streams, since replaying emitted output
would duplicate tokens. Delays must be finite; Retry-After delta seconds are
bounded. A total inference deadline is still controlled by the backend timeout.

Validation scope
-----------------

Offline HTTP fixtures verify these contracts. ``LOCAL_LLM_KIT_LIVE=1 pytest
tests/test_live_servers.py`` enables checks against existing local servers;
individual skips mean a required engine/model was not available. The tests do
not download models or establish cross-engine performance rankings.
