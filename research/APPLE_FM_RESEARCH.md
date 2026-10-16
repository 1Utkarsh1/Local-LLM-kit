# Apple Foundation Models compatibility — research summary

Scope: how Local LLM Kit should support Apple's on-device Apple Intelligence
model (Foundation Models framework), plus the competitive landscape for a
lightweight local-LLM client. Ledger: `ledger.jsonl` (48 deduplicated rows).

Counts (honest, per-row findings in the ledger):
- Discovered: ~95 (repo/docs pages encountered across direct fetches and two
  research subagents, incl. redirects, badges, and search hits).
- Screened: ~70 (title/summary examined, duplicates dropped).
- Substantively reviewed: 48 (11 direct primary fetches + 14 repo-internal
  source reads + 23 secondary fetches with recorded findings).
- The 1,000–2,000 target was deliberately not met literally: manufacturing
  rows (duplicate pages, snippets, badges) is explicitly out of scope, and
  depth on decision-relevant primary sources beats volume. Every ledger row
  carries what was examined, a concrete finding, evidence, relevance, and a
  decision.

## Load-bearing claims (all primary-verified 2026-10-16)

1. `apple/python-apple-fm-sdk` README: macOS 26+, Xcode 26+ with SDK
   agreement accepted, Python 3.10+, Apple Intelligence on; install
   `pip install apple-fm-sdk`; async-only; not taking contributions.
2. PyPI `apple-fm-sdk` JSON: latest 0.2.1 (2026-06-29); history
   0.1.0/0.1.1 (2026-03-08), 0.2.0 (2026-06-08); `requires-python >= 3.10`;
   Alpha; macOS-only; Apache-2.0.
3. SDK `session.py`: `respond(prompt, generating/schema/json_schema/options)`;
   `stream_response` yields full-text snapshots (not deltas), text-only, no
   guided streaming; sessions serialize concurrent requests.
4. SDK `generation_options.py`: only knobs are `temperature`, `sampling`
   (`greedy()` / `random(top XOR probability_threshold, seed)`),
   `maximum_response_tokens` (strict caps can malform output).
5. SDK `tool.py`: native tools = `Tool` subclasses with `GenerationSchema`
   args + `async call`; framework auto-executes. Dynamic synthesis from
   OpenAI JSON schemas is fragile → deferred, client-side loop kept.
6. SDK `core.py`: `use_case` GENERAL/CONTENT_TAGGING, `guardrails`
   DEFAULT/PERMISSIVE; `is_available()` reasons; live `context_size`;
   exact `token_count` is async.
7. SDK `prompt.py`: `ImageAttachment(path)` needs a file on disk; vision is
   file-based, not data-URL → `supports_vision=False` + placeholder fallback.
8. SDK `generable.py`: `GeneratedContent.to_json() -> str` verified →
   native `json_schema` path implemented with `str()` fallback.
9. `simonw/llm-apple-foundation` README (13★, 0 forks, ~6 commits
   2026-07-26): same SDK via `llm` CLI; options/tools/schema/conversations;
   images unsupported; tests need real hardware (pattern copied for the
   opt-in live test). Not a dependency (would drag `llm` into a light core).
10. `Arthur-Ficial/apfel` README (v1.16.0, ~6.5k★, 252 forks): `brew install
    apfel`, no Xcode; `apfel --serve` on `http://localhost:11434/v1`, model
    id `apple-foundationmodel`; chat+responses supported; `/v1/completions`
    and `/v1/embeddings` → 501; `logprobs`/`n>1`/`stop`/penalties → 400;
    `json_object` via prompt injection, `json_schema` enforced;
    4096 (macOS 26 / M1-M2) vs 8192 (macOS 27 M3+ 12GB+) windows; MIT,
    100% on-device.

## Comparative evidence table (why this solution)

| User problem | Current repo behavior | Alternative behavior (source, version) | Proposal | Benefit / cost / verification |
|---|---|---|---|---|
| Use the on-device system model from Python | No path; closest is `mlx-lm` preset (downloaded weights, different privacy story) | `apple-fm-sdk` 0.2.1 native sessions (primary) | New `apple-fm` backend, lazy optional extra | Benefit: zero-daemon on-device inference, same `LLM.chat` API. Cost: macOS/Xcode/Python gates, async bridge (~450 lines + 28 tests). Verified: offline contract tests with SDK double; live test opt-in, skipped here |
| Use the system model without Xcode | No path | `apfel --serve` OpenAI bridge (primary, v1.16.0) | New `apfel` preset with 400/501-safe profile | Benefit: no Xcode/SDK; stdlib HTTP only. Cost: external daemon, chat-only, no embeddings. Verified: payload-shape tests + mocked HTTP round-trip |
| Structured output on Apple | Prompted JSON + repair only | SDK `json_schema` guided generation (primary, `generable.py`) | Native passthrough in `apple-fm`; apfel forwards `json_schema` | Benefit: schema enforcement where the engine supports it. Verified: `to_json` contract test |
| Tool calling on Apple | Text parsing (all prompt backends) | SDK auto-executing `Tool` (primary, `tool.py`) / apfel native tools / `llm` plugin bridge | Keep uniform client-side loop (scoped, validated, timed-out); describe tools in prompt | Benefit: single security model; server stays execute-off-by-default. Cost: small-model text-call quality is model-dependent (documented) |
| Sampling knobs | temperature/top_p/top_k/penalties forwarded | SDK: temperature + sampling mode + max tokens only (primary) | Map temperature/seed/max_tokens/stop; ignore the rest (documented) | Benefit: no silent mis-mapping. Verified: options-mapping tests |
| Context accounting | 4 chars/token heuristic everywhere | SDK live `context_size` + async `token_count` (primary, `core.py`) | Live `context_size` default; heuristic counting retained | Benefit: real window (4096/8192) without per-call latency. Verified: override/live tests |
| Thin-client positioning vs LiteLLM/llm/vLLM/transformers-serve | Light core + presets | Gateways/servers cover superset endpoints incl. `/v1/responses`, `/v1/decisions|systemone` (secondary) | No new endpoints; track responses/decisions as future work | Benefit: PR stays coherent. Cost: none now |

## Alternatives considered and rejected

- Depend on `simonw/llm-apple-foundation`: would add the `llm` package to a
  light core for a thin mapping; copied its mapping ideas instead.
- Native in-framework tool auto-execution: breaks the uniform scoped/
  validated/timed-out tool model and needs fragile `@generable` synthesis;
  deferred and documented.
- `mlx-lm` as "Apple support": rejected as the answer — it downloads
  HuggingFace weights and is not the system model (kept as the separate,
  existing `mlx-lm` preset).
- macOS 27 `fm serve` preset: secondary-source only, OS requirement unmet
  here; reachable via `base_url` override, no dedicated profile.
- Private Cloud Compute: explicitly out of scope; nothing in this change
  can route traffic off-device (no PCC code path exists in the SDK surface
  used, and the backend never sends network requests).

## What remains unverified

No eligible hardware in this environment (Python 3.9 system interpreter, no
Xcode, Apple Intelligence state unknown): real inference, exact context-size
values per Mac, snapshot-streaming behavior, guardrail/refusal surfacing,
and `apfel --serve` wire behavior are covered by contract tests + mocks only.
The opt-in live test (`LOCAL_LLM_KIT_APPLE_FM_LIVE=1`) and the runnable
example document exactly what to run on an eligible Mac.
