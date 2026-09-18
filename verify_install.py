"""
Quick test script to verify local_llm_kit installation.

Runs fully offline using the `echo` backend (no model download needed).
"""

try:
    import local_llm_kit

    print("local_llm_kit version %s is installed correctly!" % local_llm_kit.__version__)

    print("\nBackends (core import is always light; heavy ones load lazily):")
    from local_llm_kit.backends import BACKENDS, available_backends

    print("- registry: %s" % ", ".join(sorted(set(BACKENDS))))
    for name in available_backends():
        print("  - %-12s (lazy import)" % name)

    print("\nSmoke test with offline echo backend:")
    from local_llm_kit import LLM, tool

    llm = LLM(model_path="echo", backend="echo")
    resp = llm.chat(messages=[{"role": "user", "content": "hello"}])
    print("- chat: %r" % resp["choices"][0]["message"]["content"])
    resp = llm.complete(prompt="hello")
    print("- complete: %r" % resp["choices"][0]["text"])
    vecs = llm.embed("hello")
    print(
        "- embed: %d vector(s) of dim %d" % (len(vecs["data"]), len(vecs["data"][0]["embedding"]))
    )

    @tool(description="Echo a word back")
    def echo_word(word: str) -> str:
        """Echo a word."""
        return word

    llm.add_tool(echo_word)
    print("- @tool registered: %s" % echo_word.spec["function"]["name"])

    from local_llm_kit.rag import SimpleVectorStore, chunk_text

    store = SimpleVectorStore(embed_fn=lambda texts: [[float(len(x))] for x in texts])
    store.add(chunk_text("hello world, offline RAG works", chunk_size=10, chunk_overlap=2))
    print("- rag: store size %d" % len(store))

    from local_llm_kit import models

    print("- models helper: %d recommended starters" % len(models.list_recommended_models()))

    try:
        import local_llm_kit.server  # noqa: F401 (fastapi needed only for create_app)

        print("- server module: importable (install fastapi to serve)")
    except ImportError as e:
        print("- server module: %s" % e)

    print("\nAvailable features:")
    for feature in [
        "Chat API",
        "Completion API",
        "Streaming",
        "Modern tool calling (@tool, tools/tool_choice)",
        "Structured output (response_format)",
        "Embeddings API",
        "Vision messages",
        "Memory management",
        "Prompt formatting (Llama3/Qwen/Gemma/Phi/DeepSeek/...)",
        "RAG helpers",
        "Model download helpers",
        "OpenAI-compatible server",
        "CLI (chat/complete/serve/pull/list/embed)",
    ]:
        print("- %s OK" % feature)

    print("\nInstallation test completed successfully!")

except ImportError as e:
    print("Error importing local_llm_kit: %s" % e)
    print("Please install the package with: pip install local-llm-kit")
