Small end-to-end flows
======================

Offline chat and streaming
--------------------------

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM("echo", backend="echo")
   messages = [{"role": "user", "content": "Hello"}]
   response = llm.chat(messages)
   print(response["choices"][0]["message"]["content"])
   for chunk in llm.chat(messages, stream=True):
       print(chunk["choices"][0]["delta"].get("content") or "", end="")

Echo validates application plumbing. Replace it with a prepared runtime from
:doc:`backends` to generate meaningful answers.

Inspect model-selected tools
----------------------------

.. code-block:: python

   from local_llm_kit import LLM, tool

   @tool(description="Look up a local product price")
   def price(product: str) -> dict:
       return {"product": product, "price": 10}

   llm = LLM("echo", backend="echo")
   response = llm.chat(
       [{"role": "user", "content": "What does tea cost?"}],
       tools=[price],
       execute_tools=False,
   )
   calls = response["choices"][0]["message"].get("tool_calls", [])
   print(calls)

The echo backend may return no call. A tool-capable model chooses calls when
available; validate arguments and application permissions before executing
powerful tools. Set ``execute_tools=True`` only for intended local execution.

Retrieve a small local corpus
-----------------------------

.. code-block:: python

   from local_llm_kit import LLM
   from local_llm_kit.rag import SimpleVectorStore, build_rag_prompt

   store = SimpleVectorStore()
   store.add(["The Eiffel Tower is in Paris.", "Tea is sold in the local shop."])
   hits = store.search("Where is the Eiffel Tower?", k=1)
   messages = build_rag_prompt("Where is the Eiffel Tower?", hits)
   response = LLM("echo", backend="echo").chat(messages)
   assert hits[0]["text"] == "The Eiffel Tower is in Paris."

Without ``embed_fn``, the store uses lexical retrieval. Connect a real embedding
backend for semantic retrieval; hash vectors are only for tests. Save with
``store.save(path)`` and restore with ``SimpleVectorStore.load(path)``.

See :doc:`examples` for runnable scripts and :doc:`quickstart` for schemas and
the OpenAI-compatible HTTP server.
