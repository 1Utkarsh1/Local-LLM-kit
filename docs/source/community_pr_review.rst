Community PR consolidation
===========================

This branch consolidates the useful parts of PRs 9 through 15 against main
commit ``62305685373878ffd3a6a2097c310dd7d801c20a``. Selection is based on code,
reproduced failures and tests. Research-ledger row counts and feature counts
are not performance or quality evidence. Raw ledgers remain in the original PRs.

Feature provenance
-------------------

.. list-table:: Reviewed selections
   :header-rows: 1

   * - Source PR
     - Retained and completed
     - Excluded or replaced
   * - `9 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/9>`_
     - Native Apple sessions, schemas, image attachment adapter, token counting and apfel profile; added effective timeouts/cancellation and streaming stops.
     - Unreachable duplicate block; ineffective waiting timeouts.
   * - `10 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/10>`_
     - Reasoning metadata propagation and explicit apfel completion-to-chat mapping.
     - Duplicate Apple adapter that dropped important messages/options; generic fallback based on error-message substrings.
   * - `11 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/11>`_
     - Responses subset, upstream model listing, atomic tool-exchange history, MMR and hybrid retrieval; completed SDK-compatible input/output and streaming.
     - Duplicate retry implementation; incomplete Responses event/usage conversion.
   * - `12 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/12>`_
     - Agent loop, validated structured output, BM25/RRF/filter/delete pipeline and hardware guidance; repaired and connected public APIs/CLI.
     - Blocking MCP transport, JSON repair for executable arguments, fabricated memory measurements and guaranteed-fit claims.
   * - `13 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/13>`_
     - Stronger MCP transport/host/CLI; fixed kept-open SSE and deadlines, atomic registration and cleanup; completed explicit 2026 stateless revision.
     - Reading an entire SSE response before returning a matched result; locality claims based only on stdio.
   * - `14 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/14>`_
     - Opt-in SQLite logging, templates and history CLI; made writes atomic, preserved payloads/IDs and fixed thread sharing and file parsing.
     - Guessing truncated tool arguments, automatic replacement of arbitrary requested models with Apple FM, duplicate adapters and malformed CI placement.
   * - `15 <https://github.com/1Utkarsh1/Local-LLM-kit/pull/15>`_
     - Discovery/doctor, native capability metadata, shared HTTP retries and current llama-server port; repaired locality, IPv6 and retry/resource validation.
     - Treating unverified OpenAI-compatible endpoints as local; nonfinite retry delays.

Concrete failures addressed
----------------------------

* PR12's required integer schema accepted a string; JSON repair changed quoted
  ``True, None`` text. Full JSON Schema validation now enforces constraints;
  output repair protects string values.
* PR15's local-only guard accepted an arbitrary remote HTTPS endpoint. Unknown
  routing now fails closed and generic loopback listings alone are insufficient.
* PR13 could time out despite already receiving a matching MCP SSE result, as
  it waited for connection closure. Parsing now returns immediately at that result.
* PR12's agent and LLM both owned tool loops. The agent now disables nested
  execution, uses strict arguments and shares one scoped executor.
* PR14's tool repair could complete a truncated filesystem path into an action.
  Its parser change is excluded; regression tests ensure it cannot execute.
* PR14's plain-text template test failed without PyYAML. Plain text now loads
  independently, and malformed directory entries produce visible errors.
* The original logging implementation lost call linkage, response token counts
  and finish reasons, split in-memory databases across threads and committed
  partial interactions. Transactions and full-payload storage address these.

Validation boundaries
----------------------

The offline suite uses real localhost HTTP fixtures and subprocesses without
model downloads. Optional interoperability tests drive the official OpenAI
Python SDK and official MCP Python SDK (legacy and 2026 stateless, stdio and
HTTP JSON/SSE). CI tests Python 3.9--3.14 and includes platform checks.

Native Apple runtime inference requires an eligible Mac and SDK setup;
AirLLM/GPU behavior and every external engine/model combination are not
certified by protocol tests. No supported-server popularity ranking,
performance superiority or guaranteed memory fit is established.
