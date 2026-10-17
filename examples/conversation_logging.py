"""
Example: SQLite conversation logging + prompt templates with local_llm_kit.

Demonstrates the Phase 2 features:
  * PromptTemplate / TemplateRegistry for reusable prompts.
  * ConversationLog for persisting conversations, messages, and tool calls.
  * LLM(conversation_log=...) for automatic interaction logging.

Runs fully offline with ``--backend echo``.

Usage:
    python examples/conversation_logging.py --backend echo --prompt "Hello"
"""

import os
import tempfile

from local_llm_kit import (
    LLM,
    ConversationLog,
    register_template,
    render_template,
)


def main():
    import argparse

    from local_llm_kit.backends import BACKENDS

    parser = argparse.ArgumentParser(description="Conversation logging demo")
    parser.add_argument("--backend", "-b", choices=sorted(BACKENDS), default="echo")
    parser.add_argument("--model", "-m", default="echo")
    parser.add_argument("--prompt", "-q", default="Hello, how are you?")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite log path (default: a temp file for this demo).",
    )
    args = parser.parse_args()

    # --- Register a reusable prompt template -------------------------------
    register_template("greeting", "You are {{model}}, a helpful assistant.")

    # --- Set up the SQLite conversation log (stdlib sqlite3 only) ----------
    db_path = args.db or os.path.join(tempfile.mkdtemp(), "demo_chat.db")
    log = ConversationLog(db_path)
    print("Logging conversations to: {0}".format(db_path))

    # --- Create an LLM wired to the log -----------------------------------
    system_prompt = render_template("greeting", model=args.model)
    llm = LLM(
        model_path=args.model,
        backend=args.backend,
        conversation_log=log,
    )

    # --- Send a message (automatically logged) -----------------------------
    response = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        tools=[],
    )
    reply = response["choices"][0]["message"]
    print("Assistant: {0}".format(reply.get("content", reply)))

    # --- List conversations from the DB -------------------------------------
    convs = log.get_conversations(10)
    print("\nLogged {0} conversation(s).".format(len(convs)))
    if convs:
        cid = convs[0]["id"]
        print("\n--- Exported transcript ---")
        print(log.export_conversation(cid, format="text")[:500])

    log.close()


if __name__ == "__main__":
    main()
