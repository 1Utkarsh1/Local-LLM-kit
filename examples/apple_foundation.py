"""
Example: using the Apple Foundation Models (Apple Intelligence) backend.

Requires macOS 26.0+ with Apple Intelligence enabled and the optional SDK::

    pip install "local-llm-kit[apple-fm]"

The backend talks to the on-device model directly (no network, no downloaded
weights). Run on an Apple Silicon Mac with Apple Intelligence turned on::

    python examples/apple_foundation.py
    python examples/apple_foundation.py --tool           # function calling
    python examples/apple_foundation.py --json           # structured JSON output
    python examples/apple_foundation.py --list --backend echo   # inspect (offline)
"""

import argparse

from local_llm_kit import LLM


def basic_chat(backend: str) -> None:
    """One-shot completion against the on-device model."""
    client = LLM(model_path="system", backend=backend, use_cache=False)
    answer = client.complete("Why is the sky blue?", max_tokens=64)
    print("Assistant:", answer["choices"][0]["text"])


def chat_with_function_calling(backend: str) -> None:
    """Register a tool and let the on-device model invoke it natively.

    ``add_tool`` stores the implementation on the LLM; for backends that opt in
    via ``_accepts_tool_impls`` (like Apple Foundation Models) the implementations
    are forwarded to the on-device session, which executes them automatically.
    """

    def get_current_weather(location: str, unit: str = "celsius") -> dict:
        print("(tool called: get_current_weather(location=%r, unit=%r))" % (location, unit))
        return {"location": location, "unit": unit, "temperature": 22, "condition": "sunny"}

    client = LLM(model_path="system", backend=backend, use_cache=False)
    client.add_tool(get_current_weather, name="get_current_weather")

    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    answer = client.chat(messages, max_tokens=64)
    print("Assistant:", answer["choices"][0]["message"]["content"])


def chat_json_response(backend: str) -> None:
    """Request structured JSON via response_format (Apple json_schema guidance)."""
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["capital", "population"],
    }
    client = LLM(
        model_path="system",
        backend=backend,
        instructions="You are a concise fact extractor.",
        use_cache=False,
    )
    messages = [{"role": "user", "content": "Tell me about France in JSON."}]
    answer = client.chat(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "country", "schema": schema},
        },
        max_tokens=128,
    )
    print("Assistant:", answer["choices"][0]["message"]["content"])


def list_registered_models(backend: str) -> None:
    """Print backend/model metadata (use --backend echo to run offline)."""
    client = LLM(model_path="system", backend=backend, use_cache=False)
    print("Backend:", client.backend.backend_name)
    print("Models:", client.backend.list_models())
    print("Model info:", client.backend.model_info())


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple Foundation Models example")
    parser.add_argument(
        "--backend", default="apple-foundation", help="Backend (default: apple-foundation)"
    )
    parser.add_argument("--tool", action="store_true", help="Demonstrate function calling")
    parser.add_argument("--json", action="store_true", help="Request structured JSON output")
    parser.add_argument("--list", action="store_true", help="Print backend/model info only")
    args = parser.parse_args()

    if args.list:
        list_registered_models(args.backend)
    elif args.tool:
        chat_with_function_calling(args.backend)
    elif args.json:
        chat_json_response(args.backend)
    else:
        basic_chat(args.backend)


if __name__ == "__main__":
    main()
