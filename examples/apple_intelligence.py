"""Apple Intelligence on-device Foundation Models example.

Demonstrates:
  1. Native on-device inference via `apple-fm` backend (apple-fm-sdk).
  2. Local HTTP bridge via `apfel` preset (http://127.0.0.1:11434/v1).
  3. Graceful offline fallback on non-Mac platforms or when Apple Intelligence is disabled.

Prerequisites for Native Inference:
  - macOS 26+ running on Apple Silicon (M-series / A17 Pro+ / A18 Pro+).
  - Apple Intelligence enabled in System Settings.
  - Python 3.10+ and `pip install "local-llm-kit[apple-fm]"`.

Usage:
  # Native on-device (Apple Silicon Mac):
  python examples/apple_intelligence.py --backend apple-fm

  # Via Apfel bridge server (after running `apfel --serve` or `brew services start apfel`):
  python examples/apple_intelligence.py --backend apfel

  # Offline fallback smoke test (runs on any OS without Apple hardware):
  python examples/apple_intelligence.py --backend echo
"""

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Apple Intelligence on-device Foundation Models demo")
    ap.add_argument(
        "--backend",
        "-b",
        default="auto",
        choices=["auto", "apple-fm", "apfel", "echo"],
        help="Backend to use: 'apple-fm' (native SDK), 'apfel' (local bridge), 'auto', or 'echo' (offline fallback).",
    )
    ap.add_argument(
        "--prompt",
        "-p",
        default="Explain on-device foundation models in two sentences.",
        help="Prompt to complete.",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream response tokens.",
    )
    args = ap.parse_args()

    from local_llm_kit import LLM

    # 1. Determine candidate backends
    if args.backend == "auto":
        candidates = ["apple-fm", "apfel", "echo"]
    else:
        candidates = [args.backend, "echo"]

    messages = [{"role": "user", "content": args.prompt}]
    llm = None
    chosen_backend = None

    for b in candidates:
        try:
            model_name = "default" if b == "apple-fm" else ("apfel" if b == "apfel" else "echo")
            llm = LLM(model_path=model_name, backend=b)
            # Try a quick test completion
            if not args.stream:
                resp = llm.chat(messages=messages, max_tokens=128)
                chosen_backend = b
                print("Connected using backend: %s" % chosen_backend)
                print("Assistant: %s" % resp["choices"][0]["message"]["content"])
                if resp["choices"][0]["message"].get("reasoning_content"):
                    print("[Thinking: %s]" % resp["choices"][0]["message"]["reasoning_content"])
                return
            else:
                chosen_backend = b
                print("Streaming using backend: %s" % chosen_backend)
                for chunk in llm.chat_stream(messages=messages, max_tokens=128):
                    delta = chunk["choices"][0]["delta"]
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        print(piece, end="", flush=True)
                print()
                return
        except Exception as exc:
            if b == "echo":
                print("Error on fallback echo backend: %s" % (exc,))
                sys.exit(1)
            print("Notice: '%s' backend unavailable (%s). Trying next candidate..." % (b, exc))

    if llm is None:
        print("Falling back to offline echo backend.")
        llm = LLM(model_path="echo", backend="echo")
        resp = llm.chat(messages=messages)
        print("Assistant (echo): %s" % resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
