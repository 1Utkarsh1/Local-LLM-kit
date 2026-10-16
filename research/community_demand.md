# local-llm-kit — Community Demand Research (users of local/offline LLM tooling)

Scope: what users of local/offline LLM tooling actually complain about and want, with emphasis on client-side concerns (endpoint/port discovery, model listing, health checks, timeouts/retries, streaming edge cases, tool-call parsing, context-window mismatches, embeddings, OpenAI-compat deviations).
Sources mined: GitHub issue trackers (15 repos), Hacker News (Algolia API), Reddit (r/ollama, r/LocalLLaMA, r/LMStudioAI, r/selfhosted, r/LLMDevs via RSS), plus a 2026 troubleshooting guide. Ledger: `ledger_community.jsonl` (353 rows).

## (a) Top 10 recurring pain points (ranked by frequency × intensity in the reviewed sample)

### 1. Connection refused / "which port" / host-binding confusion — by far the #1 cluster
Server not running vs localhost-only binding vs Docker/WSL2 networking vs wrong port vs IPv6 (::1) vs stale app lifecycle; users cannot tell which layer fails.
- https://github.com/ollama/ollama/issues/2132 (Errno 111 classic, 28 comments)
- https://news.ycombinator.com/item?id=46962040 (ClawKit: "ECONNREFUSED — usually just an IPv6/localhost binding issue but nobody knows that")
- https://www.reddit.com/r/ollama/comments/1pbiwfn/wsl2_ollama_localhost_access_issue/
- https://insiderllm.com/guides/ollama-api-connection-refused-fix/ (nine-cause triage guide, verified against Ollama v0.30.0, 2026)
- Related: ollama#3200, #5041, #6843, #14947, r/ollama 1mhvfm0, 1eri8a6; also port mismatch reports (plexmind: "model not found at http://localhost:11435", r/ollama 1ug26x0).

### 2. Tool-call parsing failures and silent drops
Tool calls vanish (empty content + finish_reason stop), leak into `content` as text, lose ids, get truncated/mutated (int64 clamping, newline stripping), or are rejected for edge-case JSON; repeated across ollama, llama.cpp, vLLM, MLX-LM, KoboldCpp, LiteLLM.
- https://github.com/ollama/ollama/issues/17274 ("tool-call output is silently discarded", 21 comments)
- https://github.com/ollama/ollama/issues/18390 (keys with spaces → call dropped, only a server-side warning)
- https://github.com/vllm-project/vllm/issues/61206 (Qwen tool-calling: 3 parser issues)
- https://github.com/ml-explore/mlx-lm/issues/1946 (drops tool_calls but reports finish_reason=tool_calls)
- https://www.reddit.com/r/ollama/comments/1w4nnwm/ (enumerated small-model failure modes: trailing commas, single quotes, near-miss names)

### 3. Streaming edge cases: tools+streaming, terminal events, finish_reason/usage
`stream=True` silently degrades to one block when tools are present; chunks without `choices` crash VSCode; `role` repeated in every chunk broke OpenAI SDK helpers; streams close without terminal events; usage missing in streaming; tool_calls `index` always 0; NDJSON vs SSE.
- https://github.com/ollama/ollama/issues/9092 (tools break streaming on /v1)
- https://github.com/ollama/ollama/issues/12420 (empty choices → "Response contained no choices")
- https://github.com/LostRuins/koboldcpp/issues/2480 (no usage chunk when request has tools)
- https://github.com/BerriAI/litellm/issues/46542 (fabricated finish_reason "stop" masks truncation)
- https://github.com/ollama/ollama/issues/18926 (stream never terminates; client hangs in CLOSE_WAIT)

### 4. Timeouts, hangs and retries
4-hour hangs with no timeout; tool/IDE defaults (30s) far too short for CPU inference; hardcoded caps users can't see (2 min load, 10 min, 182s cloud); "wall of timeout errors and retries"; LM Studio users can't even find the timeout setting; vLLM hardcoded 5s httpx timeout with no retry.
- https://github.com/ollama/ollama-python/issues/483 (4h15m hang; timeout feature requested)
- https://github.com/lmstudio-ai/lms/issues/617 ("Unable to modify timeout setting in LM Studio")
- https://github.com/vllm-project/vllm/issues/58543 (hardcoded 5s timeout, unconfigurable, no retry)
- https://www.reddit.com/r/ollama/comments/1f7k314/ (asks for a 30s client timeout); r/ollama 1lrfgy8 ("Ollama hangs without timeout"), 1sskvcr (retry storm)

### 5. Model discovery, availability checks and health that lies
No `exists()`/model-present check; no ping/health in official clients; empty /v1/models through proxies; "model not found" while `ollama ps` shows loaded; servers accept connections (and answer /v1/models) while generation is deadlocked/wedged.
- https://github.com/ollama/ollama-python/issues/640 (request: `client.exists()`)
- https://github.com/ollama/ollama-python/issues/121 (request: ping/health endpoint)
- https://github.com/ollama/ollama/issues/15950 (list endpoints fine; /api/generate hangs) and #18477 ("ps shows 100% GPU loaded", callers hang)
- https://www.reddit.com/r/ollama/comments/1iqv14l/ (LiteLLM /v1/models empty while ollama list works)

### 6. Context window mismatches (defaults, effective context, truncation)
Default 2048/4096 silently truncates agent prompts (even benchmarkers got fooled); `num_ctx` vs model context unclear; effective context not reported by /api/show; launchers hand coding agents wrong values (200K vs 1M; 262K labeled 1M); global GUI context setting silently applied to API calls (435% perf swing).
- https://github.com/ollama/ollama/issues/6286 (context cannot be changed; 2048-truncation, 21 comments)
- https://github.com/ollama/ollama/issues/18995 (effective ctx not honored in /api/show, /api/tags) and #18463 (agent launched with wrong window)
- https://www.reddit.com/r/ollama/comments/1wpu9cn/ (default 4k cut off prompts; invalidated benchmark)
- https://www.reddit.com/r/ollama/comments/1nax9sq/ ("This setting dramatically increases all Ollama model speeds" — global context affecting every request)

### 7. Structured output reliability, especially in combinations
`format`+`tools` mutually exclusive; schema not enforced with streaming+thinking; JSON never terminates on MLX (done_reason=length, truncated JSON); grammar builder drops optional-before-required fields; responses sometimes not JSON at all.
- https://github.com/ollama/ollama/issues/18944 (tools can't work with format=json)
- https://github.com/ollama/ollama/issues/18567 (MLX structured output never terminates)
- https://github.com/ggml-org/llama.cpp/issues/30377 (tools + json_schema → reasoning loop, no tool call)
- https://github.com/huggingface/text-generation-inference/issues/2959 (response_format unsupported on OpenAI route)

### 8. Server lifecycle, bind interface, CORS/origins, headless operation
Desktop GUI stops the API server (connection refused afterwards); tray apps show running with nothing listening; bind-interface confusion on multi-NIC Windows; browser/Electron origins rejected (no CORS button); headless start failures on Linux.
- https://github.com/ollama/ollama/issues/18507 (tray icon but no server on 11434; curl fails)
- https://news.ycombinator.com/item?id=46802252 (LM Studio: stopping GUI stops backend API server)
- https://github.com/nomic-ai/gpt4all/issues/3491 (API server binds wrong interface; unreachable) and #3483 (CORS button request)
- https://github.com/janhq/jan/issues/9286 (starting local API server crashes Flatpak app)

### 9. OpenAI-compat deviations (the "subtly incompatible" class)
Missing/invalid finish_reason (empty string broke the .NET SDK; fabricated "stop"); usage absent in streams; logprobs missing; 404-vs-400 semantics; role/content field presence deviations; Anthropic /v1/messages and Responses API event-ordering breaks for Claude Code/Codex.
- https://github.com/huggingface/text-generation-inference/issues/2211 (finish reason + logprobs missing) and #1896 (streaming differs from OpenAI; breaks Azure .NET SDK)
- https://github.com/mudler/LocalAI/issues/5102 ("Unknown ChatFinishReason value. Actual value was \"\"")
- https://github.com/ollama/ollama/issues/7626 (role repeated in every chunk broke openai-node stream helper)
- https://github.com/ollama/ollama/issues/18798 (Responses API output_index reuse/reordering)

### 10. Config silently ignored + metadata/capability drift
`keep_alive` ignored; context set in UI not applied by API server; capabilities advertise tools that don't work; strict client schemas crash on servers omitting fields (breaks llm-ollama); wrong context metadata in /api/show.
- https://github.com/ollama/ollama-python/issues/401 (keep_alive not honored)
- https://github.com/janhq/jan/issues/8668 and /8669 (context size not used/configurable by local API server)
- https://github.com/ollama/ollama-python/issues/607 (ShowResponse ValidationError breaks llm-ollama) + simonw/llm#1330
- https://github.com/ollama/ollama-python/issues/494 (capabilities absent from /show in client)

(Secondary clusters with material demand: embeddings compat — parameter drift (`input` vs `prompt`), encoding_format ignored → quarter-width vectors (litellm#42426), batch drops (ollama#7288); reasoning/thinking field loss across layers (lmstudio#476, litellm#41962, simonw/llm#1689); client lifecycle/close (ollama-python#441/#532).)

## (b) Honest counts

- **Discovered (raw records fetched and stored during research): ~2,220** — 1,428 GitHub issue/PR search results (34 search queries across 15 repos), 400 HN comments (4 Algolia comment queries), 298 Reddit posts (23 subreddit search RSS feeds; RSS returns posts only — comment threads were rate-limited), 91 GitHub issue comments (fetched for 8 flagship threads), plus 3 full web pages (1 blog guide; 2 earlier HN API pulls) and multiple web_search result sets (used for routing, not counted as items).
- **Screened (passed automated relevance filters; title/excerpt-level triage to build candidate sets): 1,278** — 403 ollama-repo candidates + 360 other-repo candidates + 227 HN candidates + 288 Reddit candidates.
- **Substantively reviewed and recorded in the ledger: 353 unique items** (0 malformed; 1 duplicate removed) — every row states a concrete finding from the item text (issue body excerpt ≥ ~250 chars, HN comment text, or Reddit post body). Breakdown: ollama/ollama 134, r/ollama 50, ollama/ollama-python 42, Hacker News 28, llama.cpp 13, vLLM 12, jan 10, LM Studio lms 9, LocalAI 9, LiteLLM 9, MLX-LM 8, TGI 8, KoboldCpp 7, simonw/llm 7, GPT4All 5, transformers 1, 1 blog.
- Coverage caveat: GitHub issue *comments* were read for 8 threads only; most rows are based on the issue body (noted per-row under `limitation`). Reddit comments were not retrievable at scale (Reddit RSS/JSON blocks: 403/429); Reddit rows are post-level, not comment-level. Dates run through the environment's current date (Oct 2026).

## (c) Complaints already solved by existing tools (don't rebuild blindly)

- **Connection troubleshooting docs/guiders** — official docs now cover OLLAMA_HOST and connection-refused FAQs; community guides (insiderllm 2026; markaicode; aimadetools) and tools like ClawKit (agent config debugging) exist. But no *generic Python client* ships a built-in "why can't I connect" doctor — a gap, not a solved problem.
- **Retries/timeouts/routing** — LiteLLM provides retries/fallbacks/routing (heavy, proxy-shaped, dependency-rich); community proxies (llmproxy, token-aware gateways) and nginx setups. Not solved for a stdlib-only, zero-dep client.
- **Tool-call repair/structured output** — agent frameworks and tools (GBNF grammar compilers, Toolhound-style diagnostics, Instructor/Outlines-style constrained decoding) solve pieces per-stack; no backend-agnostic normalization layer.
- **Unified multi-backend access** — LiteLLM/openrouter-style proxies and per-vendor SDKs cover breadth but require infrastructure/heavy deps; a lightweight adapter remains the niche (validated by HN comment 44869568 praising Ollama's API+SDK value, and 49436186 preferring zero-dependency harnesses).
- **Fixed upstream (marked closed; verify release before relying):** ollama #7626 (role in chunks), #4357 (finish_reason), #15457 (tool index, partially — #15497 still open for fallback parsers), #18567 (MLX structured output), #16212, #16575/#16578 (codex-app config), #17157/#16081 (timeouts), #14116 (tiered context); LocalAI #5102; lmstudio lms #80 (port reporting); TGI #2211/#2369 (finish_reason/logprobs, /v1/models); ollama-python #484 (tolerant tool args), #463 (streaming+tools example behavior). Partially fixed/in-flight: ollama-python nested tool properties (issues #429→#724; fix PR #725 open, not yet merged) — still live for released versions; ollama #17274 corrected but open; tool_choice enforcement (#17921/#14967/#11171) open across the board.

## Files
- Ledger: `ledger_community.jsonl` (353 rows; keys: url, project, type, date_or_version, what_examined, finding, relevance, confidence, limitation)
- Raw mining artifacts: `raw/gh/*.json`, `raw/hn/*.json`, `raw/rd/*.xml`, condensed candidate sets in `items/`, comment extracts in `items/comments1.jsonl`
