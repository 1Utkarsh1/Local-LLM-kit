# MCP host support — research summary

Agent: buffy-agent (independent workstream)
Repository: `1Utkarsh1/Local-LLM-kit` (branch off `main` @ `6230568`)
Date: 2026-10-17

## What was built and why

A **stdlib-only Model Context Protocol (MCP) host**. Local LLM Kit can now
connect to MCP servers over `stdio` or Streamable HTTP, discover their tools,
convert the schemas to OpenAI `tools` specs, and execute them inside the
existing chat/tool loop.

The decision came out of comparing three options after reviewing the ecosystem:

| Option | Evidence | Verdict |
|---|---|---|
| Depend on the official `mcp` Python SDK | PyPI resolver: **every** `mcp` release requires Python >= 3.10. Repo floor is 3.9. | Rejected — breaks the light core and the 3.9 floor. |
| OpenAI-bridge-only MCP (shell out to some proxy) | llama.cpp treats MCP as experimental (server README); proxy setups commonly misconfigured (llama.cpp discussion #20698). | Rejected — extra moving parts for no schema benefit. |
| **Stdlib JSON-RPC client (chosen)** | MCP stdio is newline-delimited JSON-RPC over pipes; HTTP is POST + JSON/SSE. Both are implementable with `subprocess` + `urllib`. | **Selected** — zero new dependency, 3.9-compatible, testable offline. |

## Honest research counts

Methodology: discovery came from the two focused web-research passes (MCP spec +
SDK, competitor/user-demand, and Apple FM facts) plus direct inspection of the
repository and local experiments. Each ledger row records material I (or the
research sub-agent, whose summary I read) actually examined, with a concrete
finding and a decision — not a search snippet.

- **Discovered** (distinct URLs returned/opened): ~90
- **Screened** (opened, title/metadata read, relevance judged): ~70
- **Substantively reviewed** (content examined + finding + decision recorded in
  `research/mcp_ledger.jsonl`): **49**

The 49 rows are the deduplicated ledger (`research/mcp_ledger.jsonl`). I did
**not** reach the 1,000–2,000 target and did not manufacture volume to appear to:
this was a single bounded session that also had to implement, test, and ship the
change. Counts above are the real numbers. Limitations: several ledger rows are
vendor documentation or single GitHub issues — strong evidence of a *specific*
request or behaviour, not a representative survey. Reddit pages could not be
fetched as readable text, so no Reddit thread is counted as reviewed.

## User-demand evidence (why MCP, not something else)

- Ollama MCP host request — github.com/ollama/ollama/issues/7865 (2024-11-27) and
  UI request #11603 (2025-07-31).
- GPT4All MCP + user-defined tools request — github.com/nomic-ai/gpt4all/issues/3546
  (2025-03-27, open, comments to 2025-12).
- Tool-calling reliability pain: ollama#8717 (template/model mismatches),
  llama.cpp#14697 (calls emitted as JSON text), LM Studio bug #810 (call shown
  but not executed).
- Jan docs explicitly warn not every local model tool-calls well and that tool
  definitions consume context — so the kit documents this rather than promising it.

## Competitive state (primary sources)

| Project | MCP status | Source |
|---|---|---|
| LM Studio | Full local+remote MCP host | lmstudio.ai/blog/lmstudio-v0.3.17, docs/developer/core/mcp |
| LocalAI | Remote HTTP + local stdio, discovery/execution | localai.io/docs/features/mcp/ |
| Jan | MCP stable since v0.6.9 | jan.ai/changelog |
| llama.cpp server | `--mcp-servers-config` **experimental** | tools/server/README.md |
| vLLM | MCP via Responses API, initially narrow | docs.vllm.ai MCP example; vllm#30115 |
| Ollama | Tool calling + one web-search MCP server, not a general host | ollama.com/blog/web-search |

**Read:** leading *apps* have MCP; Python *libraries/kits* largely do not, and the
reference C++ server marks it experimental. A stable, dependency-free host in a
Python kit is a real gap and a clear "modern + broadly compatible" win.

## Apple Foundation Models (separate milestone — status)

Investigated per the brief; the implementation already lives in the open Apple
PR (#9, `agent/apple-fm-support`) and a competing agent PR (#10). Findings that
matter for accuracy:

- Official SDK `apple-fm-sdk` 0.2.1 requires **macOS 26+, Xcode 26+, Python 3.10+**
  and an Apple-Intelligence Mac (M1+). It does not work on Intel or macOS < 26.
- apfel v1.16.0 is a no-Xcode bridge: `/v1/chat/completions` supported,
  `/v1/completions` and `/v1/embeddings` return **501**; model id
  `apple-foundationmodel`.
- The macOS 27 `contextSize == 0` cold-start bug was **fixed in apfel v1.8.4**
  (2026-07-22) — it is not an ongoing defect, contrary to one competing PR.
- Context size is machine-dependent (4096 on macOS 26 / M1–M2; 8192 on macOS 27
  with M3+ and >=12 GB).

**Hardware validation not performed here:** this environment is Python 3.9.6 with
no Apple-Intelligence-eligible runtime, so no real on-device inference was run.
Apple support remains contract-tested (mocked) plus an opt-in live test.

## Verification performed for this change

- Offline contract suite `tests/test_mcp.py` (26 tests): stdio + HTTP transports,
  JSON and SSE bodies, session-id echo, pagination, timeouts, dead servers,
  tool errors, JSON-RPC errors, name sanitization, config validation, LLM
  registration/namespacing/include/exclude, refresh, cleanup, and the CLI.
- **Real interop** `tests/test_mcp_interop.py` (opt-in): our client against a
  server built with the **official `mcp` SDK 2.3.0** on Python 3.11.15 — passed.
- Full suite: **207 passed, 17 skipped** on Python 3.9.6 and 207 passed on 3.11.15
  (skips are pre-existing optional-server tests + the opt-in interop test).
- `ruff check` clean; `black --check` clean.

## Limitations / follow-ups

- HTTP transport assumes one POST → JSON or request-scoped SSE; long-lived
  server→client streams and OAuth are not implemented.
- MCP *resources* and *prompts* are not surfaced (tools only).
- No real third-party MCP server binary (npx/uvx) was exercised here; the stdio
  path is covered by contract tests plus the SDK interop fixture.
