Agents, retrieval and local history
==================================

These features are unreleased additions. See the repository README for runnable
examples and :doc:`community_pr_review` for their provenance and review fixes.

Agents and structured output
-----------------------------

``Agent(llm, tools=[callable], mcp_clients=[connected_server], max_steps=5)``
runs a bounded tool loop. It disables LLM's nested tool execution and uses a
request-local tool registry. Decorated tool names are respected. MCP tools use
``server__tool`` names; the caller manages MCP connection lifetime. Invalid JSON
arguments become error feedback and never pass through JSON repair. Python tool
timeouts limit waiting rather than terminating the worker or undoing side effects.
``AgentResult.tool_calls_executed`` counts attempts, including failed calls.

``llm.chat_structured(messages, schema=..., max_retries=2)`` validates returned
JSON. Dictionary schemas require ``pip install 'local-llm-kit[structured]'``;
Pydantic v1/v2 model classes use their own validator. Native JSON Schema is
forwarded where supported. The helper performs bounded correction turns and
returns validated JSON data, not a Pydantic instance. Remote schema retrieval
is disabled. This does not make prompted JSON a constrained-decoding guarantee.

``repair_json``/``extract_json`` repair presentation errors, preserving quoted
literals, but reject unfinished strings/values. They can close containers and
are intended for opt-in output handling rather than executable actions.

Retrieval
----------

``SimpleVectorStore.search`` retains dense/TF-IDF ranking by default. Explicit
``mode='lexical'`` uses BM25; ``mode='hybrid'`` fuses BM25/dense ranks with RRF.
An embedding function and stored embeddings are required for hybrid search.
``search_hybrid(alpha=...)`` sets the dense rank weight. ``lambda_mult`` enables
MMR diversity when dense vectors exist. Metadata filters accept an exact mapping
or a predicate. ``delete(ids)`` removes the corresponding text, metadata and
vector together; JSON save/load preserves the remaining index.

``RAGPipeline`` connects chunking, storage, retrieval and a citation prompt.
Without an embedder its default works lexically. It returns source records so
the caller can inspect grounding; citations are requested, not validated. The
store is in memory and suited to small collections. Hash embeddings in offline
examples are deterministic fixtures rather than semantic retrieval models.

Conversation storage and templates
-----------------------------------

``ConversationLog(path)`` is opt-in. One locked SQLite connection supports
threads and ``:memory:`` databases. A completed interaction commits atomically.
Messages retain their complete payload, original tool IDs, response usage and
finish reason. ``export_conversation(id, 'openai')`` returns replayable message
payloads. ``log_interaction(..., conversation_id=id)`` accepts new messages or a
full history with a previously stored prefix; the stored prefix is deduplicated.
The original prototype schema is upgraded additively without deleting rows.
Close the store or use a context manager when finished.

``LLM(..., conversation_log=log)`` saves completed chat calls and streams.
Each request creates a conversation snapshot; it does not infer a shared user
session. Interrupted streams are not saved as completed answers. Stored prompts,
images and tool outputs are ordinary local SQLite data.

.. code-block:: bash

   local-llm-kit chat --backend echo -m echo --message Hello --conversation-log ./chat.db
   local-llm-kit logs list --db ./chat.db
   local-llm-kit logs show CONVERSATION_ID --db ./chat.db
   local-llm-kit logs export CONVERSATION_ID --db ./chat.db --format openai

``PromptTemplate`` supports ``{{name}}`` and ``{{name|default}}`` text replacement.
It does not evaluate expressions or escape values for JSON. JSON and plain-text
files work with the standard library; YAML needs the optional ``[yaml]`` extra.
Directory loading reports invalid JSON/YAML instead of silently skipping it.

Hardware planning
------------------

``get_system_memory`` reports measured RAM and NVIDIA GPU memory when available,
or zero/notes for unknown values. macOS reclaimable pages approximate available
RAM. Apple unified memory is shared RAM; separate GPU capacities are not summed.

``estimate_model_vram`` estimates GiB for weights, KV cache and overhead for one
sequence. Supply ``num_layers``, ``num_kv_heads`` and ``head_dim`` together to
replace the default Llama-like architecture assumption. Weight quantization,
runtime overhead, backend allocation and batch size still affect peak usage.
``recommend_quantization`` uses 85% of available memory as a planning budget.
It does not guarantee fit or prevent an out-of-memory error.

.. code-block:: bash

   local-llm-kit hardware --params-b 8 --context-length 8192
   local-llm-kit hardware --params-b 8 --memory-gb 12

Stateless Responses API
------------------------

The development server supports ``POST /v1/responses`` for text/image message
input, instructions, flat function tool definitions, function-call/output
round trips, ``max_output_tokens`` and ``text.format``. Streaming emits the
created/item/content/delta/done/completed lifecycle for text and tool arguments,
with stable IDs and sequence numbers. Usage maps to input/output token names.
An unserved requested model returns an error. Nonstream inference runs outside
the ASGI event loop. Responses returns tool calls for the caller to execute.

Stateful responses, hosted tools, background jobs and reasoning controls are
unsupported and rejected. Native schema enforcement remains model-dependent;
use structured chat for separate returned-data validation. This is a documented
subset, not full OpenAI API parity. The official Python SDK stream manager is
exercised by ``LOCAL_LLM_KIT_OPENAI_INTEROP=1 pytest tests/test_responses_sdk.py``.
The server is a development adapter, not an authenticated multi-user gateway.
