Performance and measurement
===========================

Most inference cost belongs to the selected engine and model. The kit is a
client/interface layer and does not establish throughput or memory advantages
for a particular runtime. Protocol tests use fixtures, not model benchmarks.

Recorded GPU validation
-----------------------

The `2026-10-25 A100 80 GB evidence bundle
<https://github.com/1Utkarsh1/Local-LLM-kit/tree/main/benchmarks/2026-10-25-a100>`_
contains real model checks, warm loopback HTTP measurements, exact model
revisions, software manifests, scripts, failures and passing regressions.
Each measured engine is a separate configuration; quantization and scheduling
differ. These small runs establish reproducibility and compatibility for the
listed combinations. They do not establish comparative model quality, cold-start
performance or superiority over another library.

Avoid repeated loading
----------------------

Reuse an ``LLM`` instance or the module-level cached client helpers. Model
loading can cost much more than a request. Disable reuse with ``use_cache=False``
when an application needs isolated backend instances. See :doc:`api_reference`
for cache clearing.

Budget context and output
-------------------------

Set ``max_tokens`` for each request and configure a suitable context window.
Server token estimates are heuristic; exact tokenization lives in the engine.
Measure long prompts separately from decoding and include the output length.
AirLLM rejects prompt-plus-output requests exceeding its configured
``max_seq_len`` instead of silently truncating.

Streaming and concurrency
-------------------------

HTTP chat streams are incremental when the upstream server supports them.
AirLLM's adapter emits a completed buffered chunk. These have different latency
properties. AirLLM serializes generation per backend; GPU server scheduling,
batching and concurrency belong to the external engine.

For comparisons, record engine/model versions, accelerator and RAM, quantization,
prompt/output token counts, concurrent requests, warm-up and sampling settings.
Report first-token latency and sustained generation separately. Do not infer
model performance from the deterministic echo backend.

Use upstream runtime guidance for GPU kernels, offloading, batching and
quantization; unsupported engine options should not be guessed from another
provider's configuration.
