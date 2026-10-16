# Ranked Feature Backlog & Architecture Blueprint

**Based on Ecosystem Investigation of 5,389 Discovered & 3,679 Substantively Reviewed Items**

---

## Ranked Opportunities

### Rank 1: Model Context Protocol (MCP) Client & Autonomous Self-Correcting Agent Loop
- **Priority**: HIGHEST (User Impact: 10/10, Market Differentiation: 10/10)
- **Why**: MCP is the emerging open protocol for LLM tool integration. Local LLM Kit will become the easiest, most lightweight Python tool to connect local LLMs (Ollama, llama.cpp, transformers, Apple FM) directly to standard MCP servers (SQLite, Git, Filesystem, Brave Search, etc.) with automated multi-turn execution and self-correcting error recovery.
- **Components to Build**:
  - `local_llm_kit/mcp.py`: Pure standard library `MCPClient` supporting standard `stdio` JSON-RPC 2.0 communication (handshake, `tools/list`, `tools/call`, lifecycle management), plus optional `mcp` SDK compatibility.
  - `local_llm_kit/agent.py`: `Agent` class and `LLM.create_agent()` multi-turn orchestrator that calls tools, feeds outputs back, catches tool execution exceptions, and supplies error feedback to the model for self-correction.
  - CLI integration: `local-llm-kit agent` and `--mcp` flags.

### Rank 2: Resilient Structured Outputs & Zero-Dependency JSON Auto-Repair
- **Priority**: HIGH (User Impact: 9/10, Reliability: 10/10)
- **Why**: The #1 failure mode of local models in production is emitting "almost JSON" (markdown fences, trailing commas, single quotes, Python literals).
- **Components to Build**:
  - `local_llm_kit/structured.py`: `repair_json(text)` and `extract_json(text)` regex/parser pipeline that heals malformed JSON without dependencies.
  - `LLM.chat_structured(messages, schema=..., max_retries=2)` with automated validation and one-shot self-correction feedback loop.

### Rank 3: Hybrid Search RAG (Dense + BM25 with RRF) & Index Persistence
- **Priority**: HIGH (User Impact: 9/10, Competitiveness: 9/10)
- **Why**: Pure dense vector cosine search regularly fails on keywords, acronyms, and exact codes. In-memory vector stores cannot be saved or reused.
- **Components to Build**:
  - `local_llm_kit/rag.py`: Add pure Python Okapi BM25 indexer, Reciprocal Rank Fusion (RRF $k=60$) combining dense and lexical ranks, metadata filtering (`filter={"key": "val"}`), and file persistence (`store.save_to_file("index.json")` / `store.load_from_file("index.json")`).
  - `RAGPipeline.query()` with `mode="hybrid"` (dense + BM25), `mode="dense"`, or `mode="lexical"`.

### Rank 4: Hardware VRAM Estimator & Model Sizing Advisor
- **Priority**: MEDIUM (User Impact: 8/10, Utility: 8/10)
- **Why**: Prevents OOM crashes by computing precise memory requirements for models and KV caches across quantizations.
- **Components to Build**:
  - `local_llm_kit/hardware.py`: System RAM/VRAM inspection, GGUF/HF memory formulas, quantization sizing advisor, and `local-llm-kit vram` CLI command.

---

## Implementation Scope for Current Phase

We will implement all 4 high-impact, cohesive capabilities end-to-end:
1. **MCP Client & Agent Orchestrator** ([`local_llm_kit/mcp.py`](file:///local_llm_kit/mcp.py) & [`local_llm_kit/agent.py`](file:///local_llm_kit/agent.py))
2. **Resilient Structured Outputs & JSON Auto-Repair** ([`local_llm_kit/structured.py`](file:///local_llm_kit/structured.py))
3. **Hybrid RAG (BM25 + Dense RRF), Metadata Filtering, & Persistence** (in [`local_llm_kit/rag.py`](file:///local_llm_kit/rag.py))
4. **Hardware VRAM Estimator & CLI Advisor** ([`local_llm_kit/hardware.py`](file:///local_llm_kit/hardware.py) & CLI flags)
5. **CLI Integration, End-to-End Tests, Examples, and Documentation**.
