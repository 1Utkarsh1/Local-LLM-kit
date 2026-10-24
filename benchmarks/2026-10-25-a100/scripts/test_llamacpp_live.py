"""Actual CUDA GGUF smoke and boundary checks, using the Ollama-downloaded file."""

import importlib.metadata
import json
import pathlib
import time

import pytest

MODEL = "/workspace/ollama-models/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"
ARTIFACTS = pathlib.Path("/workspace/artifacts")


@pytest.fixture(scope="module")
def backend():
    import llama_cpp

    from local_llm_kit.backends.llamacpp import LlamaCppBackend

    assert llama_cpp.llama_supports_gpu_offload()
    with open(MODEL, "rb") as model:
        assert model.read(4) == b"GGUF"
    started = time.monotonic()
    result = LlamaCppBackend(MODEL, n_gpu_layers=-1, n_ctx=8192, verbose=True)
    (ARTIFACTS / "llamacpp-environment.json").write_text(
        json.dumps(
            {
                "llama_cpp_python": importlib.metadata.version("llama-cpp-python"),
                "model_sha256": MODEL.rsplit("sha256-", 1)[1],
                "n_gpu_layers": -1,
                "n_ctx": 8192,
                "load_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )
    return result


def test_real_native_locality_and_completion(backend):
    from local_llm_kit import LLM

    llm = LLM(MODEL, backend_instance=backend, require_local=True, use_cache=False)
    assert llm.assert_local()
    response = llm.complete("The capital of France is", temperature=0, max_tokens=32)
    assert response["choices"][0]["text"].strip()
    (ARTIFACTS / "llamacpp-response.json").write_text(json.dumps(response, indent=2))


def test_real_stream_matches(backend):
    prompt = "Write a Python function to add two numbers:\n"
    normal = backend.generate(prompt, temperature=0, max_new_tokens=32)
    streamed = "".join(
        item["text"] for item in backend.generate_stream(prompt, temperature=0, max_new_tokens=32)
    )
    assert streamed == normal["text"]


@pytest.mark.parametrize("streaming", [False, True])
def test_real_context_overflow(backend, streaming):
    with pytest.raises(ValueError, match="context|tokens"):
        response = (backend.generate_stream if streaming else backend.generate)(
            " hello" * 9000, max_new_tokens=8
        )
        if streaming:
            list(response)
