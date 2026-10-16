# Research summary — local server discovery, capability truth, and cold-start resilience

Agent-authored research for the Local LLM Kit repository (`1Utkarsh1/Local-LLM-kit`).
Starting commit: `6230568` (main after PR #8).

- **Ledger:** [`research/discovery_ledger.jsonl`](./discovery_ledger.jsonl) — one JSON object per
  line with `url, project, type, date_or_version, what_examined, finding, relevance, confidence, limitation`.
  Named distinctly from the other open PRs' ledgers (`research/ledger.jsonl` in #9 and #11) so the
  three do not collide on the same path.
- **Supporting reports:** [`server_probe_matrix.md`](./server_probe_matrix.md) (verified
  endpoint/port/health matrix for 16 local servers), [`competitor_gap_analysis.md`](./competitor_gap_analysis.md)
  (capability gap matrix for 37 projects), [`apple_foundation_models.md`](./apple_foundation_models.md)
  (Apple Foundation Models investigation), [`community_demand.md`](./community_demand.md)
  (ranked pain points from GitHub issues, Hacker News and Reddit).

## Honest research counts

Four streams were run in parallel: community demand (GitHub issues, Hacker News, Reddit),
server probe surfaces (ports, health and capability endpoints), competitor capabilities, and
the Apple Foundation Models investigation. Each stream kept its own funnel; the table below
reports them separately rather than summing them into one impressive number.

| Stream | Discovered | Screened | Substantively reviewed |
|---|---|---|---|
| Community demand | ~2,220 raw records (1,428 GitHub search results across 34 queries / 15 repos, 400 HN comments, 298 Reddit posts, 91 issue comments, 3 full pages) | 1,278 candidates | **353** |
| Server probe surfaces | not separately instrumented | 16 servers × all their endpoints/docs/source | **232** |
| Competitor capabilities | not separately instrumented | 37 projects | **211** |
| Apple Foundation Models | not separately instrumented | Apple, apfel, llm-apple-foundation, `fm serve`, alternatives | **49** |
| First-hand by this agent | — | — | **11** (measurements and primary-source reads) |

Measured directly from the retained artefacts:

| Stage | Count | How it was measured |
|---|---|---|
| **Discovered** | **2,985** distinct URLs | Unique URLs extracted from the 261 retained candidate files. This is a *floor*: the probe and competitor streams ran their own searches whose result pages were not kept in the same pool. |
| **Screened** | **1,278** candidates in the community stream alone, plus the probe/competitor/Apple triage | Community-stream figure as reported by that stream; the others were not instrumented separately. |
| **Substantively reviewed** | **852** distinct, deduplicated items | 856 rows were written across the streams (353 + 232 + 211 + 49 + 11); 4 collided and were merged, along with 354 byte-identical duplicate rows from the community stream's four intermediate parts. 436 high / 365 medium / 51 low confidence. |

Breakdown of the 852: 283 issues · 203 documentation pages · 146 source files · 94 source-code
items · 50 Reddit threads · 28 forum posts · 17 package-metadata records · 10 pull requests ·
6 releases · 5 example files · 2 test files · 2 discussions · 2 measurements · plus the Apple
and first-hand source reads.

Across **79 distinct projects** and **20 domains**. The largest contributors were
`ollama/ollama` (154), `ollama/ollama-python` (56), r/ollama (50), the Apple Foundation Models
stream (45), Hacker News (28), `simonw/llm` (22), `ggml-org/llama.cpp` (39 across two name
spellings), `vllm-project/vllm` (20) and `BerriAI/litellm` (18).

**Stated plainly:** the review reached **852** substantively reviewed items against the
1,000–2,000 aspiration in the brief. The count is not inflated to reach the target. The 354
merged duplicates are reported rather than counted twice, and the 2,985 discovered URLs are
not presented as reviewed. Every stream recorded its own limitations per row — most GitHub
rows are issue-body-level (comments were read for 8 flagship threads only), and Reddit rows
are post-level because Reddit rate-limits its comment APIs.

## The problem this PR solves

The strongest recurring theme across all three streams is that local AI failures are rarely
model failures. They are **environment failures**:

| # | User problem | Evidence | Frequency in the ledger |
|---|---|---|---|
| 1 | "Is my server even running, and on which port?" | `ollama/ollama#2132`, `#18390`; 50 r/ollama threads; 28 HN posts; four 2026 troubleshooting articles dedicated to `connection refused` on `localhost:11434` | ~45 items |
| 2 | "The server is up but the model is still loading" | llama.cpp `/health` returns `503 {"error":{"message":"Loading model"}}`; Ollama queues via `OLLAMA_MAX_QUEUE`; SGLang `/ready` 503; MLX-LM `/health` `{"status":"unavailable"}` | ~45 items |
| 3 | "The client reports a context window that is not the model's" | Ollama publishes the real value on `/api/show`; this library hardcoded 4096 | 47 items |
| 4 | "Is this actually running on my machine?" | Ollama `:cloud` models carry `remote_host: https://ollama.com:443` in `/api/tags`; a "local" setup can silently call a provider | ~13 items |
| 5 | "A default port in the docs no longer matches upstream" | llama.cpp moved `llama-server` 8080 → 9931 (`#30159`, merged 2026-10-09) | 3 items |

The independent community report ranks its pain points as
[`community_demand.md`](./community_demand.md) does: **#1** connection-refused / "which port" /
host-binding confusion, **#4** timeouts, hangs and retries, **#5** model discovery and health
that lies, **#6** context-window mismatches. The changes in this PR target those four directly.

## Evidence table for the selected changes

| User problem | Current repo behaviour (before) | Alternative's behaviour | Source and version | Change made | Expected benefit | Cost | Verification |
|---|---|---|---|---|---|---|---|
| Unknown which local endpoint is up | `OPENAI_SERVER_URLS` presets only; a wrong guess raises `ConnectionError` | LiteLLM/Ollama clients also require a known URL; no mainstream client probes | 50 r/ollama threads; 28 HN posts; `ollama/ollama#2132`; `simonw/llm` docs | `local_llm_kit.discovery` + `local-llm-kit doctor`: concurrent probe of 12 endpoints, provider identified from the server's own payload | One command answers "what is running and what can it do" | ~640 LOC, 0 deps | 41 offline tests; opt-in live suite; real run on Ollama 0.34.2 |
| `503` treated the same as "absent" | A 5xx was reported as a generic failure with no "it is loading" signal | llama.cpp/SGLang/MLX-LM all use 503 for "starting" | llama.cpp `tools/server/README.md` | 5xx is reported as reachable-but-not-ready, with `--keep-loading` | Users wait instead of debugging the wrong thing | ~10 LOC | Fixture test with a 500 and a 503 |
| Context window and vision are guessed | `OllamaBackend` reported `context_window=4096`, `supports_vision=true` for **every** model | Ollama publishes `context_length` and `capabilities`; LM Studio publishes both; llama.cpp publishes `n_ctx` and `modalities` | Measured on Ollama 0.34.2 + `qwen2.5:0.5b`; `ggml-org/llama.cpp` `/props` docs; LM Studio `/api/v1/models` docs | Real values read from `/api/show`, `/api/tags`, `/props`, `/api/v1/models`; adapters stop inventing | `context_window` is 32768 instead of 4096 (8× wrong) and vision is no longer claimed falsely | ~200 LOC, 0 deps | Before/after comparison on a live server; fixture tests per provider |
| A "local" model may leave the device | No notion of provider routing | Not surfaced by any mainstream Python client surveyed | Ollama `/api/tags` `remote_host` field; apfel docs | `remote`/`remote_host` on `ModelRecord`, `doctor` flags it, `LLM(require_local=True)` refuses it | "Offline" becomes enforceable rather than aspirational | ~120 LOC, 0 deps | Real refusal of `glm-5.2:cloud`; live + fixture tests |
| Request sent before the server is ready | Single attempt; instant `ConnectionError` | `openai-python` retries twice with backoff; litellm adds more | `openai-python` `_constants.py`; `ggml-org/llama.cpp` 503 semantics | `local_llm_kit.retry.RetryPolicy` wired into both HTTP backends | Cold-start race recovers transparently | ~330 LOC incl. tests, 0 deps | Measured: 0.01s failure vs 1.22s success with the server starting 0.5s late |
| `llama-server` preset pointed at a stale port | `http://localhost:8080/v1` | Upstream default is 9931 since 2026-10-09 | `ggml-org/llama.cpp#30159` (merged), `#26508` (notice) | Preset follows the binary at 9931; both ports are probed; documented migration | A default `llama-server` launch is reachable | ~5 LOC | Unit test on the preset and the probe catalog |

## Why these changes were chosen over the alternatives

The competitor analysis ranked nine candidate additions. The two highest-benefit,
lowest-cost items that also matched the strongest community signal were selected:

1. **Discovery + real capability reporting** — ranked as the top *unserved* need. Every
   alternative surveyed (litellm, ollama-python, lmstudio-python, simonw/llm, openai-python)
   requires the user to already know the URL, and none reads capability metadata across
   providers. It is also the change that makes the library's own "offline AI" claim checkable.
2. **Retry policy** — ranked #1 by the competitor gap analysis and corroborated by the
   community stream (`connection refused`, 503-while-loading). It complements discovery
   directly: find the server, then survive it not being ready yet.

Deliberately **not** done (from the anti-recommendation list in the gap analysis): a
router/failover layer (24 direct dependencies), token-level structured-output enforcement via
outlines/lm-format-enforcer, tracing SDKs, a plugin system, a conversation store, and swapping
the stdlib HTTP transport for httpx. Each would break the light-core constraint without
addressing the measured demand.

## Apple Foundation Models findings

Fully documented in [`apple_foundation_models.md`](./apple_foundation_models.md). Summary:

- **Two other open PRs already implement Apple Foundation Models support** (#9 and #10). Per
  the brief's guidance on overlap, this PR does **not** duplicate that work. The Apple
  investigation is recorded here because it is relevant to how the new discovery layer
  behaves, and because it is the evidence for the recommendation below.
- **Recommended path: bridge-first.** `apfel --serve` exposes Apple's on-device model at
  `http://127.0.0.1:11434/v1` as `apple-foundationmodel`, is MIT-licensed, needs no Xcode, and
  is actively released (v1.16.0, 2026-10-05).
- **Native `apple-fm-sdk` must stay opt-in.** Apple publishes an **sdist only** — no wheels for
  0.1.0 through 0.2.1 — so `pip install apple-fm-sdk` compiles Swift and requires macOS 26 +
  full Xcode 26 (the build backend explicitly rejects a CommandLineTools-only toolchain) and
  Python ≥ 3.10, above this library's 3.9 floor.
- **On-device is kept on-device.** No Private Cloud Compute or other remote fallback exists in
  this change, and the new `require_local` guard makes that refusal explicit.
- **Hardware not verified.** This machine has no Apple Foundation Models runtime, `apfel` is not
  installed, and the on-device model is unavailable. Apple-specific behaviour is validated
  against the `apfel` **source** (its OpenAPI request model and `ChatRequestValidator`) and
  fixture tests, **not** by real Apple-silicon inference. The third-party `fm serve` tool is
  probed on 127.0.0.1:1976 but is documented as unverified: Apple publishes no primary
  documentation for it.

## Overlap with other PRs

Seven PRs were open when this one was opened; four of them touch the same backend files. Each is
disclosed rather than glossed over.

| PR | Overlap | Handling and tradeoffs |
|---|---|---|
| **#11** `agent/local-toolkit-upgrades` | **Genuine and material: it also adds bounded retries** for `ollama`/`openai-compat` (`retries=2`, `retry_backoff=0.5s`, `retries=0` to fail fast, via `backend_kwargs`), and touches `ollama.py`, `openai_compat.py`, `backends/__init__.py`, `llm.py`. | Two independent designs for the same problem. #11 is smaller and applies to non-streaming calls. This PR adds coverage #11 does not claim: a tested `RetryPolicy` value object, `Retry-After` parsing/clamping to 120s, jitter, retrying the **streaming connection phase** (only before the first frame, so tokens are never replayed), and an explicit single-attempt path for capability detection so `doctor` never sleeps. The kwarg names differ (`max_retries` here to match the official OpenAI client; `retries` there). **Whoever merges second should converge on one policy object and one kwarg name rather than shipping both** — this is called out rather than silently duplicated. |
| #9 `agent/apple-fm-support` | Apple FM backend + apfel bridge; also creates `research/ledger.jsonl` and touches `__init__.py`, `backends/__init__.py`, `openai_compat.py`, `llm.py`, `cli.py`. | Not duplicated. This PR's ledger was renamed to `research/discovery_ledger.jsonl` so the three `research/ledger.jsonl` writers do not collide. |
| #10 `feat/apple-fm-integration` | Apple FM backend, apfel 501 handling, reasoning fields; touches the same shared backend files. | Not duplicated. Complementary: that reports apfel capabilities per request; this reports capabilities for any server via discovery. |
| #12, #13 `agent/*` (MCP, hybrid RAG, VRAM advisor, Responses API) | New modules (`mcp.py`, `agent.py`, `hardware.py`, `structured.py`) plus `llm.py`, `cli.py`, `rag.py`, `server.py`. | No feature overlap with this PR. Shared-file conflicts would be mechanical; `rag.py` and `server.py` are untouched here. |
| #7, #8 (merged) | Reliability fixes, 0.2.0 release | This PR is based on the merge of #8 and preserves those changes. |

No other agent's changes were copied, reverted or rewritten.

## Remaining limitations

- **Apple hardware unverified.** No Apple Foundation Models runtime on this machine; see above.
- **Live-server certification covers Ollama only.** `tests/test_live_servers.py` was executed
  against a real Ollama 0.34.2 on macOS. LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan,
  GPT4All, MLX-LM, KoboldCpp, apfel, Docker Model Runner and `fm serve` are covered by
  **protocol fixtures built from their documented/verified shapes**, not by running each engine.
  Their `/v1/models` and capability payloads are as documented in
  [`server_probe_matrix.md`](./server_probe_matrix.md), which also lists what could not be
  confirmed.
- **`fm serve` is unverified.** Probed defensively at 1976; its `/v1/models` shape has no
  primary source, and Apple published no documentation for it that this investigation could
  confirm.
- **Retry defaults change timing.** A failing request now takes up to ~1.5s longer before
  raising. This matches the official OpenAI client but is a behaviour change; `max_retries=0`
  restores the previous single-attempt behaviour.
- **`llama-server` port change is a preset change.** Users of an older build or the Docker
  image must pass `base_url` explicitly; documented in the changelog and backend guide.
- **Per-model capability lookup is capped.** `doctor` issues at most 12 `/api/show` calls, so a
  machine with more models reports the first 12 in detail.
- **No benchmark claims.** The only timings reported are loopback measurements of the retry
  policy and probe latency; they are not inference-engine benchmarks.
