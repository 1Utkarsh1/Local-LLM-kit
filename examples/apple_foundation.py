"""
Apple Foundation Models example: on-device inference with zero setup.

Two paths, same OpenAI-shaped API:

1. Native (recommended on eligible Macs)::

       pip install "local-llm-kit[apple-fm]"   # macOS 26+, Xcode 26+, Python 3.10+

   from local_llm_kit import LLM

   llm = LLM(model_path="apple-foundationmodel", backend="apple-fm")
   print(llm.chat([{"role": "user", "content": "Say hi in one sentence."}]))

2. Bridge (no Xcode needed)::

       brew install apfel        # macOS 26+, Apple Silicon, Apple Intelligence on
       apfel --serve             # serves http://localhost:11434/v1

   The ``apfel`` preset speaks to that server with the shared
   OpenAI-compatible adapter (same code path as vLLM/LM Studio presets).

Run:  python examples/apple_foundation.py [--bridge] [--model MODEL]
"""

import argparse
import sys

PREREQUISITES = """\
Apple Foundation Models prerequisites (https://support.apple.com/en-us/121115):

  native: macOS 26+, Apple Intelligence-capable Mac with Apple Intelligence
          on, Xcode 26+ (SDK agreement accepted), Python 3.10+,
          pip install "local-llm-kit[apple-fm]"
  bridge: macOS 26+, Apple Silicon, Apple Intelligence on, `brew install apfel`
          then `apfel --serve`
"""


def _preflight_native():
    from local_llm_kit.backends.apple_fm import AppleFMBackend

    ok, reason = AppleFMBackend.is_available()
    if not ok:
        print("Apple Foundation Models unavailable: %s" % reason)
        print(PREREQUISITES)
        return False
    return True


def run_native(model, message):
    if not _preflight_native():
        return 2
    from local_llm_kit import LLM

    llm = LLM(model_path=model, backend="apple-fm")
    response = llm.chat([{"role": "user", "content": message}])
    print(response["choices"][0]["message"]["content"])
    return 0


def run_bridge(model, message, base_url):
    from local_llm_kit import LLM

    llm = LLM(
        model_path=model,
        backend="apfel",
        backend_kwargs={"base_url": base_url},
    )
    try:
        response = llm.chat([{"role": "user", "content": message}])
    except ConnectionError as exc:
        print("Could not reach an OpenAI-compatible server: %s" % exc)
        print("Start one first, e.g. `apfel --serve`, then retry.")
        return 2
    print(response["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Apple's on-device model.")
    parser.add_argument("--model", "-m", default="apple-foundationmodel")
    parser.add_argument(
        "--message",
        "-q",
        default="Say hi in one sentence.",
        help="Single message to send (default: a greeting prompt).",
    )
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="Use the apfel bridge server instead of the native SDK backend.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Bridge server URL (default: apfel --serve default).",
    )
    args = parser.parse_args()
    if args.bridge:
        sys.exit(run_bridge(args.model, args.message, args.base_url))
    sys.exit(run_native(args.model, args.message))
