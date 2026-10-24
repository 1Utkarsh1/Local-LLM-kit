"""Run actual Local-LLM-kit requests against a loopback inference server."""

import argparse
import concurrent.futures
import json
import pathlib
import sys
import time
import traceback
import urllib.request

from local_llm_kit import LLM

p = argparse.ArgumentParser()
p.add_argument("--backend", required=True)
p.add_argument("--url", required=True)
p.add_argument("--model", required=True)
p.add_argument("--report", required=True)
args = p.parse_args()
records = []


def client():
    return LLM(
        args.model,
        backend=args.backend,
        use_cache=False,
        backend_kwargs={
            "base_url": args.url,
            "timeout": 120,
            "context_window": 8192,
            "max_retries": 0,
        },
    )


def ask(prompt, **options):
    settings = {
        "max_tokens": 128,
        "temperature": 0,
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    settings.update(options)
    return client().chat([{"role": "user", "content": prompt}], **settings)


def content(response):
    value = response["choices"][0]["message"]["content"]
    assert isinstance(value, str) and value.strip(), response
    return value


def case(name, fn):
    started = time.monotonic()
    try:
        data = fn()
        record = {"name": name, "passed": True, "seconds": time.monotonic() - started, "data": data}
    except Exception as exc:
        record = {
            "name": name,
            "passed": False,
            "seconds": time.monotonic() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    records.append(record)
    pathlib.Path(args.report).write_text(
        json.dumps(
            {"backend": args.backend, "model": args.model, "url": args.url, "results": records},
            indent=2,
        )
    )
    print(name, "PASS" if record["passed"] else "FAIL", round(record["seconds"], 3), flush=True)


def listing():
    with urllib.request.urlopen(args.url + "/models", timeout=10) as r:
        result = json.load(r)
    assert args.model in [m["id"] for m in result["data"]]
    return result


def chat():
    result = ask("What is 2 plus 2? Give only the number.")
    assert "4" in content(result)
    return result


def stream():
    prompt = "Write a Python function named add that returns the sum of a and b."
    normal = ask(prompt, max_tokens=64)
    chunks = list(ask(prompt, max_tokens=64, stream=True))
    text = "".join(
        (chunk["choices"][0].get("delta") or {}).get("content") or "" for chunk in chunks
    )
    assert text == content(normal), {"normal": normal, "stream": text}
    return {"response": normal, "stream_text": text, "chunks": len(chunks)}


def schema():
    result = ask(
        "Return a JSON object for Paris, France.",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
            },
        },
    )
    parsed = json.loads(content(result))
    assert set(parsed) == {"city", "country"}
    assert all(isinstance(v, str) for v in parsed.values())
    return result


def tools():
    result = ask(
        "Look up the weather in Sydney.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        execute_tools=False,
    )
    choice = result["choices"][0]
    calls = choice["message"]["tool_calls"]
    assert choice["finish_reason"] == "tool_calls"
    assert calls[0]["function"]["name"] == "lookup_weather"
    assert isinstance(json.loads(calls[0]["function"]["arguments"])["city"], str)
    return result


def reasoning():
    result = ask(
        "What is 2 plus 2?",
        max_tokens=256,
        chat_template_kwargs={"enable_thinking": True},
        reasoning_effort="low",
    )
    message = result["choices"][0]["message"]
    assert message.get("reasoning_content") or message.get("reasoning"), result
    assert content(result)
    return result


def completion():
    result = client().complete("The capital of France is", max_tokens=32, temperature=0)
    assert result["choices"][0]["text"].strip(), result
    return result


def overflow(streaming):
    try:
        result = ask(" hello" * 9000, max_tokens=8, stream=streaming)
        if streaming:
            list(result)
    except (ValueError, RuntimeError) as exc:
        assert "context" in str(exc).lower() or "400" in str(exc), str(exc)
        return {"rejected_with": type(exc).__name__, "message": str(exc)}
    raise AssertionError("Oversized context was not rejected")


def concurrent_requests():
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda n: ask(f"What is {n} plus 10? Give only the number.", max_tokens=32),
                range(4),
            )
        )
    for result in results:
        content(result)
    return {"requests": 4, "wall_seconds": time.monotonic() - started, "responses": results}


case("model_listing", listing)
case("chat", chat)
case("stream_matches_nonstream", stream)
case("structured_json_schema", schema)
case("forced_tool_call_without_execution", tools)
case("reasoning_metadata", reasoning)
case("text_completion", completion)
case("context_overflow_nonstream", lambda: overflow(False))
case("context_overflow_stream", lambda: overflow(True))
case("four_concurrent_requests", concurrent_requests)
sys.exit(0 if all(r["passed"] for r in records) else 1)
