Practical usage and limits
==========================

Choose the runtime before the model
-----------------------------------

Use :doc:`backends` to match your hardware, model format and required features.
A server preset chooses a URL and protocol; the server must already be running
with the requested model. Do not assume that an OpenAI-compatible chat endpoint
also implements tools, JSON Schema, embeddings or vision.

Keep application checks explicit
--------------------------------

Responses are dictionaries. Check the final ``finish_reason`` and inspect
``tool_calls`` before treating an answer as plain text. A length-limited answer
can be incomplete. Prompted JSON is not proof of JSON Schema conformance; validate
it against your application's schema before using it.

Pass tools as Python callables only when the application intends to execute
them. ``execute_tools=False`` returns calls for application-managed execution.
The bundled server defaults remote tool execution to off. A timeout bounds the
client's wait; it cannot kill an already-running Python function. Use process
isolation for tools that require cancellation or stronger resource limits.

Protect shared state
--------------------

Per-call tools are request-scoped. ``add_tool`` registers a persistent tool on
the client. Shared client/backend caches reduce repeated model loading; avoid
changing their model state concurrently. AirLLM generation is serialized because
its offloaded layers are stateful. Explicitly manage conversation messages in
your application; the memory helper is not a durable conversation database.

Make offline operation repeatable
---------------------------------

Prepare trusted weights, dependencies and any backend caches while connected.
Use local model paths and the runtime's offline options. Repository revisions
and filenames identify model downloads; do not substitute a same-named file
from another repository. Retain tested dependency/model versions when
reproducing an application.

Use echo and hash embeddings for plumbing tests only. Use actual semantic
embeddings and evaluate retrieval on representative questions before relying
on RAG answers. Keep performance claims tied to the exact hardware, model,
quantization, context length and workload that produced them.
