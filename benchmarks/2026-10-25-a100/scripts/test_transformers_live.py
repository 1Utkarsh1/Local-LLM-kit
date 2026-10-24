"""Real CUDA inference checks; run only on the disposable test pod."""

import asyncio
import importlib.metadata
import json
import os
import pathlib
import time

import pytest

MODEL = os.environ.get("LLK_LIVE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ARTIFACTS = pathlib.Path("/workspace/artifacts")


@pytest.fixture(scope="module")
def backend():
    import torch

    from local_llm_kit.backends.transformers import TransformersBackend

    assert torch.cuda.is_available(), "Live CUDA tests must actually use a GPU"
    started = time.monotonic()
    instance = TransformersBackend(MODEL, device="cuda", torch_dtype="bfloat16")
    assert next(instance.model.parameters()).is_cuda
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "transformers-environment.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "kit": importlib.metadata.version("local-llm-kit"),
                "context_window": instance.get_context_window(),
                "load_seconds": time.monotonic() - started,
                "gpu": torch.cuda.get_device_name(),
            },
            indent=2,
        )
    )
    return instance


@pytest.mark.parametrize("budget", [8, 32])
def test_streamed_deltas_equal_real_greedy_completion(backend, budget):
    prompt = "Write a short Python function to add two numbers:\n"
    normal = backend.generate(prompt, temperature=0, max_new_tokens=budget)
    chunks = list(backend.generate_stream(prompt, temperature=0, max_new_tokens=budget))
    assert chunks
    assert all(isinstance(c["text"], str) for c in chunks)
    assert "".join(c["text"] for c in chunks) == normal["text"]
    assert normal["text"].strip()


@pytest.mark.parametrize("stream", [False, True])
def test_actual_oversized_prompt_rejected(backend, stream):
    prompt = " hello" * (backend.get_context_window() + 16)
    assert backend.count_tokens(prompt) > backend.get_context_window()
    with pytest.raises(ValueError, match="context window"):
        result = (backend.generate_stream if stream else backend.generate)(prompt, max_new_tokens=8)
        if stream:
            list(result)


@pytest.mark.parametrize("stream", [False, True])
def test_output_reservation_rejected(backend, stream):
    with pytest.raises(ValueError, match="context window"):
        result = (backend.generate_stream if stream else backend.generate)(
            "Hello", max_new_tokens=backend.get_context_window()
        )
        if stream:
            list(result)


@pytest.mark.parametrize("stream", [False, True])
def test_real_generation_error_reaches_caller(backend, stream):
    started = time.monotonic()
    with pytest.raises(ValueError, match="max_new_tokens"):
        result = (backend.generate_stream if stream else backend.generate)(
            "Hello", max_new_tokens=-1
        )
        if stream:
            list(result)
    assert time.monotonic() - started < 10


def test_native_exact_context_fit(backend):
    prompt = " hello" * (backend.get_context_window() - 16)
    remaining = backend.get_context_window() - backend.count_tokens(prompt)
    assert 0 < remaining <= 32
    response = backend.generate(prompt, temperature=0, max_new_tokens=remaining)
    assert isinstance(response["text"], str)


@pytest.mark.parametrize(
    "prompt",
    ["Say hello.", "Explain café in one sentence.", "Return a JSON object with a city field."],
)
def test_public_llm_chat(backend, prompt):
    from local_llm_kit import LLM

    llm = LLM(MODEL, backend_instance=backend, use_cache=False, require_local=True)
    response = llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=32)
    json.dumps(response)
    assert response["choices"][0]["message"]["content"].strip()


def test_http_routes_with_real_inference(backend):
    import httpx

    from local_llm_kit import LLM
    from local_llm_kit.server import create_app

    app = create_app(llm=LLM(MODEL, backend_instance=backend, use_cache=False, require_local=True))

    async def exercise():
        samples = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=120
        ) as client:
            cases = [
                (
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "Write a greeting."}]},
                ),
                ("/v1/completions", {"prompt": "Write a greeting:"}),
                ("/v1/responses", {"input": "Write a greeting."}),
            ]
            for route, body in cases:
                body.update(model=MODEL, max_tokens=32, temperature=0)
                if route.endswith("responses"):
                    body.pop("max_tokens")
                    body["max_output_tokens"] = 32
                result = await client.post(route, json=body)
                assert result.status_code == 200, result.text
                assert result.json()
                samples.append({"route": route, "response": result.json()})
            task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Write a long story about a mountain expedition.",
                            }
                        ],
                        "max_tokens": 128,
                        "temperature": 0,
                    },
                )
            )
            await asyncio.sleep(0.05)
            started = time.monotonic()
            health = await client.get("/health")
            health_latency = time.monotonic() - started
            unfinished = not task.done()
            response = await task
            assert health.status_code == 200
            assert response.status_code == 200, response.text
            (ARTIFACTS / "native-http.json").write_text(
                json.dumps(
                    {
                        "routes": samples,
                        "health_seconds": health_latency,
                        "health_served_while_generation_unfinished": unfinished,
                    },
                    indent=2,
                )
            )

    asyncio.run(exercise())
