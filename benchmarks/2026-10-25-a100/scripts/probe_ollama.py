"""Exercise the native Ollama adapter against a real, locally served Qwen model."""

import json
import pathlib
import sys
import time
import traceback

from local_llm_kit import LLM

MODEL = "qwen3.8:27b"
REPORT = pathlib.Path("/workspace/artifacts/qwen-ollama-probes.json")
records = []


def client():
    return LLM(
        MODEL,
        backend="ollama",
        require_local=True,
        use_cache=False,
        backend_kwargs={
            "host": "http://127.0.0.1:11434",
            "timeout": 120,
            "max_retries": 0,
            "keep_alive": "10m",
        },
    )


def ask(prompt, **kwargs):
    options = {"temperature": 0, "max_tokens": 128, "seed": 7, "think": False, "num_ctx": 8192}
    options.update(kwargs)
    return client().chat([{"role": "user", "content": prompt}], **options)


def text(result):
    value = result["choices"][0]["message"]["content"]
    assert isinstance(value, str) and value.strip(), result
    return value


def run(name, operation):
    started = time.perf_counter()
    try:
        data = operation()
        item = {
            "name": name,
            "passed": True,
            "seconds": time.perf_counter() - started,
            "data": data,
        }
    except Exception as exc:
        item = {
            "name": name,
            "passed": False,
            "seconds": time.perf_counter() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    records.append(item)
    REPORT.write_text(
        json.dumps({"model": MODEL, "backend": "ollama", "results": records}, indent=2)
    )
    print(name, item["passed"], round(item["seconds"], 3), flush=True)


def locality():
    llm = client()
    assert llm.assert_local()
    result = llm.detect_backend_capabilities()
    assert result["detected"]
    return result


def chat():
    result = ask("What is 2 plus 2? Give only the number.")
    assert "4" in text(result)
    assert result["usage"]["completion_tokens"] > 0
    return result


def stream():
    prompt = "Write a Python function add(a, b) returning their sum."
    normal = ask(prompt, max_tokens=64)
    pieces = list(ask(prompt, max_tokens=64, stream=True))
    actual = "".join(c["choices"][0]["delta"].get("content") or "" for c in pieces)
    assert actual == text(normal), {"normal": normal, "streamed": actual}
    return {"response": normal, "streamed_text": actual, "chunks": len(pieces)}


def schema():
    result = ask(
        "Return the city Paris and the country France in JSON.",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
            },
        },
    )
    assert set(json.loads(text(result))) == {"city", "country"}
    return result


def tools():
    result = ask(
        "Call lookup_weather for Sydney.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        execute_tools=False,
    )
    call = result["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"])["city"]
    return result


def reasoning():
    result = ask("What is 2 plus 2?", think=True, max_tokens=256)
    assert result["choices"][0]["message"].get("thinking"), result
    assert text(result)
    return result


def completion():
    result = client().complete(
        "The capital of France is", max_tokens=32, temperature=0, think=False, num_ctx=8192
    )
    assert result["choices"][0]["text"].strip()
    return result


for name, fn in [
    ("locality_and_capabilities", locality),
    ("chat", chat),
    ("stream_matches_nonstream", stream),
    ("structured_json_schema", schema),
    ("tool_call_without_execution", tools),
    ("reasoning_metadata", reasoning),
    ("text_completion", completion),
]:
    run(name, fn)
sys.exit(0 if all(item["passed"] for item in records) else 1)
