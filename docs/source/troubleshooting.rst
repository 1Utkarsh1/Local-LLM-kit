Troubleshooting
===============

Connection refused or HTTP 404
------------------------------

Check that the external engine is running, the model is loaded, the port matches
``backend_kwargs`` and the OpenAI-compatible base URL includes ``/v1``. Ollama's
native adapter uses ``host`` without ``/v1``. Jan can require a configured API
key. Desktop apps may need their local API server explicitly enabled.

Unsupported tools, schemas or embeddings
----------------------------------------

Check :doc:`backends` and the engine's model/template settings. HTTP compatibility
does not imply every optional feature exists. GPT4All and MLX-LM presets have no
embedding route; use a separate embedding backend. A prompted JSON response may
need application-side validation.

Invalid tool records
--------------------

Names must be non-empty strings and arguments must be objects or JSON strings
encoding objects. Explicit null, arrays, scalar JSON and malformed JSON are
rejected. Empty-string tool arguments retain the legacy empty-object behaviour.
Model-output parsers ignore malformed calls instead of executing them.

Downloads and offline loading
-----------------------------

Check repository access, exact filename, revision and available disk space.
Use ``pull --force`` to refresh an identified cache entry. Do not remove an
unrelated complete model to recover from a failed transfer. Prepare dependencies
and trusted weights before enabling Hugging Face offline mode.

For a reproducible bug report, include the package/Python/backend versions,
platform and a minimal example at the `issue tracker
<https://github.com/1Utkarsh1/Local-LLM-kit/issues>`_.
