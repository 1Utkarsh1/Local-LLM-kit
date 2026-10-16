API reference
=============

The exported Python client is ``local_llm_kit.LLM``. Responses and streaming
chunks use OpenAI-shaped dictionaries.

LLM
---

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM(model_path="echo", backend="echo", use_cache=False)
   response = llm.chat([{"role": "user", "content": "Hello"}])
   text = response["choices"][0]["message"]["content"]
   completion = llm.complete("Once upon a time", max_tokens=64)
   vectors = llm.embed("A short document")

``model_path`` identifies a checkpoint or a server's model ID. ``backend`` selects
``echo``, ``transformers``, ``airllm``, ``llamacpp``, ``ollama``,
``openai-compat`` or a named server preset. ``backend_kwargs`` configures the
selected backend; for servers this includes ``base_url``/``host``, ``api_key``
and ``timeout``. See :doc:`backends`.

``LLM.chat`` accepts ``messages``, sampling options, ``stream``, ``tools``,
``tool_choice``, ``response_format``, ``execute_tools``, ``max_iterations`` and
``tool_timeout``. Legacy ``functions``, ``function_call`` and ``format="json"``
remain available. Tools passed as callables are request-scoped. Register a
persistent callable with ``llm.add_tool(callable)`` or a legacy implementation
with ``llm.add_function(name, schema, implementation)``.

``LLM.complete`` accepts a text prompt and sampling/format options.
``LLM.embed`` accepts a string or a list of strings and requires an embedding
backend. The echo backend returns deterministic test vectors.

Set ``stream=True`` on chat or completion to receive an iterator. Chat chunks
carry ``choices[0].delta``; text-completion chunks carry ``choices[0].text``.
AirLLM streaming is buffered into one completed backend chunk.

High-level helpers
------------------

.. code-block:: python

   from local_llm_kit import chat, complete, embed, get_client

   result = chat([{"role": "user", "content": "Hello"}], model_path="echo", backend="echo")
   result = complete("Hello", model_path="echo", backend="echo")
   result = embed("Hello", model_path="echo", backend="echo")
   client = get_client(model_path="echo", backend="echo")

The helpers reuse clients. ``chat`` forwards ``execute_tools``, ``max_iterations``,
``tool_timeout``, ``top_p``, ``top_k`` and ``repetition_penalty`` to the request
rather than the backend constructor. Existing positional arguments retain their
positions. ``achat`` and ``acomplete`` provide asynchronous non-streaming wrappers.
``clear_client_cache`` and ``clear_backend_cache`` drop the corresponding caches.

Call records and validation
---------------------------

``FunctionCall.from_dict`` requires a non-empty string ``name`` and ``arguments``
as a dict or a JSON object string. ``ToolCall.from_dict`` accepts either a bare
function record or an OpenAI ``{"function": {...}, "id": ...}`` envelope. Missing
tool arguments default to an empty object. Empty argument strings also represent
an empty object. Explicit nulls, JSON arrays/scalars and malformed JSON are rejected.
Non-dict records raise ``TypeError``; invalid fields raise ``ValueError``.

This validates call-record shape. Tool execution checks required argument names;
it does not implement a complete JSON Schema validator for argument values.

Model downloads and caches
--------------------------

.. code-block:: python

   from local_llm_kit.models import download_model, resolve_model

   path = download_model("owner/repo", filename="model.gguf", revision="main", force=False)
   path_or_id, backend = resolve_model("owner/repo:model.gguf", download=False)

Downloads reuse files by repository/revision or full URL identity. ``force=True``
refreshes downloads; explicit local paths remain unchanged. Transfers use unique
temporary files and an atomic replacement after size/length checks. Named Hugging
Face revisions are resolved through their cache refs rather than arbitrary snapshots.
``list_cached_models`` discovers cached files. ``format_recommended_models``
accepts an optional list of table rows.

Memory and vector stores
------------------------

``llm.memory`` is a ``MessageHistory``. Use ``get_messages()`` and ``clear()`` to
inspect or clear it. Malformed batches are checked before they enter history, and
token estimates include tool arguments. The conversation sent to ``chat`` is the
request's supplied messages; applications should maintain the history they send.

``SimpleVectorStore`` supports dense or lexical retrieval. ``save(path)`` writes
JSON; ``load(path, embed_fn=None)`` validates collection shapes and vector dimensions.
Corrupt stores raise ``ValueError`` instead of fabricating IDs or vectors.

HTTP server
-----------

``local_llm_kit.server.create_app(llm=client)`` builds a FastAPI app. Install the
``server`` extra to serve it. Routes include ``/health``, ``/v1/models``,
``/v1/chat/completions``, ``/v1/completions`` and ``/v1/embeddings``. The server
validates generation options, forwards structured-output formats and preserves
native embedding vectors. Remote tool execution defaults to disabled and requires
an explicit ``execute_tools=true`` request.

This bundled server is a development shim. Use a dedicated inference server for
production serving requirements.
