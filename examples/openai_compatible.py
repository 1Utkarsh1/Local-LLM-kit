"""Use the stock OpenAI client against a local-llm-kit server.

Offline-first: if the `openai` package or a server at --base-url is not
reachable, falls back to a direct LLM(backend="echo") call so the script
always exits 0 without downloads.

Usage:
    # 1) in one terminal (offline smoke test):
    local-llm-kit serve --backend echo --port 8000
    # 2) in another:
    python examples/openai_compatible.py
    python examples/openai_compatible.py --model llama3.2:3b --base-url http://localhost:8000/v1
    python examples/openai_compatible.py --backend echo   # force offline path
"""
import argparse
from typing import Dict, List


def _direct_echo(messages: List[Dict], model: str) -> str:
    from local_llm_kit import LLM
    llm = LLM(model_path=model, backend="echo")
    resp = llm.chat(messages=messages)
    return resp["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI client vs local server demo")
    ap.add_argument("--model", "-m", default="echo",
                    help="Model name to request (default: echo)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="Server base URL (default: http://localhost:8000/v1)")
    ap.add_argument("--backend", "-b", default="echo",
                    help="Fallback backend for offline mode (default: echo)")
    ap.add_argument("--prompt", "-p", default="Say hi in one sentence.",
                    help="User message to send")
    args = ap.parse_args()

    messages = [{"role": "user", "content": args.prompt}]

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("openai package not installed; using direct backend fallback.")
        print(_direct_echo(messages, args.model))
        return

    try:
        client = OpenAI(base_url=args.base_url, api_key="not-needed")
        resp = client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=256,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print("Server at %s unreachable (%s); falling back to backend=%r."
              % (args.base_url, e, args.backend))
        print(_direct_echo(messages, args.model))


if __name__ == "__main__":
    main()
