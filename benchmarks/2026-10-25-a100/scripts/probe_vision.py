"""One reproducible inline-image check: identify a generated solid-red PNG."""

import argparse
import base64
import json
import pathlib
import struct
import time
import zlib

from local_llm_kit import LLM

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--backend", choices=["ollama", "sglang"], required=True)
args = p.parse_args()


def chunk(kind, data):
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))


image = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress((b"\0" + bytes([255, 0, 0]) * 64) * 64))
    + chunk(b"IEND", b"")
)
root = pathlib.Path("/workspace/artifacts")
(root / "vision-input-red.png").write_bytes(image)
model = "qwen3.8:27b" if args.backend == "ollama" else "Qwen/Qwen3.8-27B"
settings = (
    {"think": False, "num_ctx": 8192}
    if args.backend == "ollama"
    else {"chat_template_kwargs": {"enable_thinking": False}}
)
llm = LLM(model, backend=args.backend, use_cache=False, backend_kwargs={"timeout": 120})
started = time.perf_counter()
result = llm.chat(
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the dominant color in this image? Reply with only its name.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image).decode()
                    },
                },
            ],
        }
    ],
    temperature=0,
    max_tokens=64,
    **settings,
)
text = result["choices"][0]["message"]["content"]
record = {
    "backend": args.backend,
    "model": model,
    "input": "vision-input-red.png",
    "passed": isinstance(text, str) and "red" in text.lower(),
    "seconds": time.perf_counter() - started,
    "response": result,
    "limit": "One solid-color synthetic image; not a general vision benchmark.",
}
(root / f"qwen-{args.backend}-vision.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
assert record["passed"], text
