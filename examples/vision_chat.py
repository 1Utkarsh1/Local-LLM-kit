"""Chat with an image_url message; degrades gracefully to text-only.

On text-only backends/models the image part is ignored and the text part
is still answered, so this script always runs offline with --backend echo.

Usage:
    python examples/vision_chat.py
    python examples/vision_chat.py --image https://example.com/cat.jpg --prompt "What is in this image?"
    python examples/vision_chat.py --backend openai-compat --model llava --base-url http://localhost:8080/v1
"""
import argparse

TEXT_FALLBACK_NOTE = ("(vision not supported by this backend/model; "
                      "answered from text part only)")


def build_messages(prompt: str, image_url: str):
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}]


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision chat demo (offline-safe)")
    ap.add_argument("--model", "-m", default="echo", help="Model path/name")
    ap.add_argument("--backend", "-b", default="echo", help="Backend (default: echo)")
    ap.add_argument("--base-url", default=None, help="Base URL for openai-compat/ollama")
    ap.add_argument("--image", default="https://example.com/cat.jpg",
                    help="Image URL (or data: URI)")
    ap.add_argument("--prompt", "-p", default="What is in this image?",
                    help="Text accompanying the image")
    args = ap.parse_args()

    from local_llm_kit import LLM

    kwargs = {"model_path": args.model, "backend": args.backend}
    if args.base_url:
        kwargs["backend_kwargs"] = {"base_url": args.base_url}

    messages = build_messages(args.prompt, args.image)
    try:
        llm = LLM(**kwargs)
        resp = llm.chat(messages=messages)
        print(resp["choices"][0]["message"]["content"])
    except Exception as e:
        # Ultimate fallback: text-only question via echo backend
        print("%s (%s)" % (TEXT_FALLBACK_NOTE, e))
        llm = LLM(model_path="echo", backend="echo")
        resp = llm.chat(messages=[{"role": "user", "content": args.prompt}])
        print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
