# Community demand — what local-LLM users actually ask for

Two evidence sweeps (2026-10-16) plus the Apple-compatibility research in
`APPLE_FM_RESEARCH.md`. Ledger: `ledger.jsonl` (77 deduplicated rows).

Counts (honest, ledger-backed):
- Discovered: ~275 (repo/doc pages, Reddit/forum search hits and threads,
  GitHub search results across 15 projects, redirects and badges excluded
  from review counts).
- Screened: ~180 (title/summary examined, duplicates dropped).
- Substantively reviewed: 77 ledger rows, each with examined material,
  concrete finding, evidence pointer, relevance, and decision.
- The 1,000–2,000 target was not manufactured: duplicate pages, snippets,
  badges, and unopenable links do not count. Depth on decision-relevant
  sources was chosen over volume; every counted row carries a finding.

Method notes: Reddit bodies are not directly fetchable (network policy
blocks), so Reddit evidence rests on search-visible titles plus the
fetchable GitHub issues, docs, and forum threads that confirm the same
themes. Reaction/comment counts are quoted only where the fetch rendered
them. This repo itself has no open issues and no open PRs at time of
writing, so demand is read from competitors and forums.

## Demand themes (deduplicated, ranked by breadth × severity)

1. **Responses API (`/v1/responses`) compatibility** — broadest cross-project
   ask: ollama#9659 (Codex blocked; fixed Dec 2025), ollama#10309,
   llama.cpp#19138, vLLM#14721, LocalAI#5005, mlx-lm#1207, litellm#9146,
   transformers#40571. Shipped by LM Studio (Codex compat stated),
   Ollama ≥0.13.3 (non-stateful only), vLLM, transformers serve.
   → Implemented: stateless Responses subset on the dev server.
2. **Tool-calling correctness** — streaming chunks dropped (ollama-python#279,
   LocalAI#9722, transformers#40620), parser errors (vLLM#22519,
   koboldcpp#1651/#2516, mlx#1061), silent drops (textgen#6633),
   hallucinated calls on small models (HF forum threads).
   → Implemented: exchange-atomic history truncation (apfel#500 precedent);
   existing streamed-argument assembly retained and covered for
   message-shaped chunks.
3. **Retries / timeouts / hangs** — instructor `max_retries` docs pattern,
   llm#1504 (`--timeout`/`--max-retries` for Responses), mlx#1909
   (silent hang), apfel#522 (typed retry classifiers), vLLM#5060.
   → Implemented: opt-out bounded retries with backoff on HTTP backends.
4. **RAG quality: rerank, diversify, hybrid** — ollama#3368 rerankers
   (113c/381r, biggest RAG ask), OpenWebUI hybrid BM25+CrossEncoder recipe,
   vLLM#19229 reranker serving.
   → Implemented: MMR diversification + dense/lexical hybrid fusion
   (CrossEncoder-class reranking deferred: needs model downloads this
   environment cannot verify).
5. **Model listing / discovery** — jan#7944 (load/unload/available API),
   LM Studio `/v1/models` + lifecycle docs.
   → Implemented: `LLM.list_models()` + server aggregation (lifecycle
   control stays server-side, out of scope).
6. **Structured-output reliability** — const/enum ignored (ollama#8063),
   silently dropped constraints (#15260), order/grammar bugs
   (lmstudio#130, kobold#1449, mlx#1007).
   → Existing: native schema passthrough where engines enforce it,
   prompt/repair elsewhere; explicit rejections instead of silent drops.
7. **Vision serving** — model coverage + offload (ollama#9727,
   llama.cpp#8010, vLLM#4194, kobold#2229, apfel#510).
   → Implemented: Apple data-URL vision; `transformers serve` preset for
   VL-capable local servers.
8. **MCP support** — ollama#7865 (60c/221r, strongest single signal),
   llm#696 (17c/59r), gpt4all#3546, jan#4824.
   → Deferred with reasoning: an MCP client means spawning local servers
   (arbitrary code execution) plus a protocol dependency; needs a dedicated
   trust/security review bigger than this change. Tracked future work.
9. **Context-window honesty** — silent halving (jan#8764), VRAM-based
   defaults (OpenWebUI docs), re-processing (llama.cpp#21831).
   → Existing + Apple live `context_size`; documented, not silently worked
   around.
10. **Logging/observability** — OLLAMA_DEBUG levels (#10950), Jan verbose
    logs. Narrow, server-side; deferred for a client library.

## Backlog actually built (this change)

Responses endpoint → model listing → retries → MMR/hybrid → memory
atomicity → transformers-serve preset. Each maps to ≥2 independent demand
sources above; each is offline-testable without hardware; none adds a
runtime dependency. MCP and CrossEncoder reranking are the explicit,
reasoned deferrals.
