"""Warm loopback latency measurements; requires an already running local engine."""

import argparse
import concurrent.futures
import importlib.metadata
import json
import pathlib
import statistics
import time
from datetime import datetime, timezone

from local_llm_kit import LLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    messages = [
        {
            "role": "user",
            "content": "List the integers from 1 to 200, separated by commas. Do not explain or stop early.",
        }
    ]
    settings = dict(temperature=0, max_tokens=64, top_p=1, top_k=40, repetition_penalty=1, seed=7)
    settings.update(
        {"think": False, "num_ctx": 8192}
        if args.backend == "ollama"
        else {"chat_template_kwargs": {"enable_thinking": False}}
    )

    def request(stream=False):
        backend_kwargs = {"timeout": 120, "max_retries": 0, "context_window": 8192}
        backend_kwargs["host" if args.backend == "ollama" else "base_url"] = args.url
        llm = LLM(args.model, backend=args.backend, use_cache=False, backend_kwargs=backend_kwargs)
        started = time.perf_counter()
        response = llm.chat(messages, stream=stream, **settings)
        if stream:
            first = None
            pieces = []
            for chunk in response:
                text = chunk["choices"][0].get("delta", {}).get("content") or ""
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    pieces.append(text)
            assert first is not None, "No streamed content"
            return {
                "seconds": time.perf_counter() - started,
                "ttft_seconds": first,
                "text": "".join(pieces),
            }
        seconds = time.perf_counter() - started
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        tokens = usage.get("completion_tokens")
        assert isinstance(text, str) and text.strip(), response
        assert isinstance(tokens, int) and tokens > 0, usage
        assert response["choices"][0]["finish_reason"] != "length" or tokens == 64, usage
        return {
            "seconds": seconds,
            "usage": usage,
            "text": text,
            "output_tokens_per_second": tokens / seconds,
            "finish_reason": response["choices"][0]["finish_reason"],
        }

    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model": args.model,
        "kit_version": importlib.metadata.version("local-llm-kit"),
        "endpoint": args.url,
        "messages": messages,
        "settings": settings,
        "method": "2 warmups, 10 sequential repeats, 5 streaming repeats, then 4 concurrent requests; same prompt, engine prefix caching enabled; localhost HTTP; no network-isolation claim",
    }
    path = pathlib.Path(args.report)

    def save():
        path.write_text(json.dumps(report, indent=2))

    report["warmups"] = [request() for _ in range(2)]
    report["sequential"] = []
    for _ in range(10):
        report["sequential"].append(request())
        save()
    report["streaming"] = [request(stream=True) for _ in range(5)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        report["concurrent"] = list(pool.map(lambda _: request(), range(4)))
    report["concurrent_wall_seconds"] = time.perf_counter() - started
    report["summary"] = {
        "sequential_n": 10,
        "median_latency_seconds": statistics.median(r["seconds"] for r in report["sequential"]),
        "min_latency_seconds": min(r["seconds"] for r in report["sequential"]),
        "max_latency_seconds": max(r["seconds"] for r in report["sequential"]),
        "median_output_tokens_per_second": statistics.median(
            r["output_tokens_per_second"] for r in report["sequential"]
        ),
        "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in report["streaming"]),
        "concurrent_output_tokens_per_second": sum(
            r["usage"]["completion_tokens"] for r in report["concurrent"]
        )
        / report["concurrent_wall_seconds"],
        "completion_tokens": [r["usage"]["completion_tokens"] for r in report["sequential"]],
    }
    assert len(report["sequential"]) == 10 and len(report["concurrent"]) == 4
    assert all(r["seconds"] > 0 for r in report["sequential"])
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
