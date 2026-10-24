"""Actual AirLLM smoke; no fake model objects or execution-locality assumptions."""

import importlib.metadata
import json
import pathlib
import time

import pytest

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ARTIFACTS = pathlib.Path("/workspace/artifacts")


@pytest.fixture(scope="module")
def backend():
    from local_llm_kit.backends.airllm import AirLLMBackend

    started = time.monotonic()
    result = AirLLMBackend(MODEL, device="cuda:0", max_seq_len=256, prefetching=False)
    (ARTIFACTS / "airllm-environment.json").write_text(
        json.dumps(
            {
                "airllm": importlib.metadata.version("airllm"),
                "transformers": importlib.metadata.version("transformers"),
                "model": MODEL,
                "max_seq_len": 256,
                "load_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )
    return result


def test_real_completion(backend):
    response = backend.generate("The capital of France is", temperature=0, max_new_tokens=8)
    assert response["text"].strip()
    (ARTIFACTS / "airllm-response.json").write_text(json.dumps(response, indent=2))


def test_buffered_stream(backend):
    response = list(backend.generate_stream("Hello", temperature=0, max_new_tokens=4))
    assert len(response) == 1 and response[0]["text"].strip()


def test_context_bound(backend):
    with pytest.raises(ValueError, match="max_seq_len"):
        backend.generate(" hello" * 300, max_new_tokens=8)
