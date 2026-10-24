"""Blocking inference and model loading must leave the ASGI loop responsive."""

import asyncio
import threading
import unittest
from unittest import mock

try:
    import httpx
    from fastapi import FastAPI  # noqa: F401

    from local_llm_kit.server import create_app

    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


class Gate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.threads = []

    def wait(self):
        self.threads.append(threading.get_ident())
        self.entered.set()
        self.release.wait(timeout=3)


class GatedLLM:
    model_path = "fixture"

    def __init__(self, gate=None):
        self.gate = gate

    def wait(self):
        if self.gate is not None:
            self.gate.wait()

    def chat(self, messages, **kwargs):
        self.wait()
        if kwargs.get("stream"):
            return iter([{"choices": [{"delta": {"content": "works"}}]}])
        return {"choices": [{"message": {"role": "assistant", "content": "works"}}]}

    def complete(self, prompt, **kwargs):
        self.wait()
        result = {"choices": [{"text": "works"}]}
        return iter([result]) if kwargs.get("stream") else result

    def embed(self, texts):
        self.wait()
        return [[1.0, 0.0] for _ in texts]


REQUESTS = (
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
    ("/v1/embeddings", {"input": "hi"}),
    ("/v1/responses", {"input": "hi"}),
)


@unittest.skipUnless(HAS_SERVER, "requires optional fastapi and httpx packages")
class TestServerConcurrency(unittest.IsolatedAsyncioTestCase):
    async def assert_health_while_blocked(self, app, gate, requests):
        loop_thread = threading.get_ident()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            tasks = [asyncio.create_task(client.post(path, json=body)) for path, body in requests]
            try:
                self.assertTrue(await asyncio.to_thread(gate.entered.wait, 5))
                health = await asyncio.wait_for(client.get("/health"), timeout=2)
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {"status": "ok"})
                self.assertTrue(all(worker != loop_thread for worker in gate.threads))
                self.assertTrue(all(not task.done() for task in tasks))
            finally:
                gate.release.set()
                results = await asyncio.gather(*tasks)
            for response in results:
                self.assertEqual(response.status_code, 200, response.text)
            return results

    async def test_chat_inference_does_not_block_health(self):
        gate = Gate()
        await self.assert_health_while_blocked(create_app(llm=GatedLLM(gate)), gate, [REQUESTS[0]])

    async def test_completion_inference_does_not_block_health(self):
        gate = Gate()
        await self.assert_health_while_blocked(create_app(llm=GatedLLM(gate)), gate, [REQUESTS[1]])

    async def test_native_embeddings_do_not_block_health(self):
        gate = Gate()
        await self.assert_health_while_blocked(create_app(llm=GatedLLM(gate)), gate, [REQUESTS[2]])

    async def test_responses_inference_remains_nonblocking(self):
        gate = Gate()
        await self.assert_health_while_blocked(create_app(llm=GatedLLM(gate)), gate, [REQUESTS[3]])

    async def test_streaming_routes_remain_nonblocking(self):
        for path, body in (REQUESTS[0], REQUESTS[1], REQUESTS[3]):
            with self.subTest(path=path):
                gate = Gate()
                responses = await self.assert_health_while_blocked(
                    create_app(llm=GatedLLM(gate)), gate, [(path, dict(body, stream=True))]
                )
                self.assertIn("text/event-stream", responses[0].headers["content-type"])
                self.assertIn("works", responses[0].text)
                if path != "/v1/responses":
                    self.assertIn("data: [DONE]", responses[0].text)

    async def test_lazy_loading_does_not_block_any_inference_route(self):
        for request in REQUESTS:
            with self.subTest(path=request[0]):
                gate = Gate()

                def load(*args, gate=gate, **kwargs):
                    gate.wait()
                    return GatedLLM()

                with mock.patch("local_llm_kit.llm.LLM", side_effect=load) as factory:
                    await self.assert_health_while_blocked(
                        create_app(model_path="fixture"), gate, [request]
                    )
                    self.assertEqual(factory.call_count, 1)

    async def test_concurrent_first_requests_construct_only_one_model(self):
        gate = Gate()

        def load(*args, **kwargs):
            gate.wait()
            return GatedLLM()

        with mock.patch("local_llm_kit.llm.LLM", side_effect=load) as factory:
            await self.assert_health_while_blocked(
                create_app(model_path="fixture"), gate, [REQUESTS[0], REQUESTS[1]]
            )
            self.assertEqual(factory.call_count, 1)

    async def test_fallback_embedding_work_does_not_block_health(self):
        gate = Gate()

        def embed(text):
            gate.wait()
            return [1.0, 0.0]

        with mock.patch("local_llm_kit.server._pseudo_embedding", side_effect=embed):
            await self.assert_health_while_blocked(create_app(llm=object()), gate, [REQUESTS[2]])
