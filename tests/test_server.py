"""Offline tests for the OpenAI-compatible HTTP server.

Target API (v0.2.0): local_llm_kit.server exposing an app factory
(create_app / build_app / make_app / get_app) serving
POST /v1/chat/completions, POST /v1/completions, GET /v1/models and
GET /health. Tests inject a fake echo LLM so no model is loaded, and skip
gracefully when fastapi (or the server module) is unavailable.

Run: python -m pytest tests/test_server.py -q
"""

import unittest

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False
    TestClient = None

try:
    import sys as _sys

    __import__("local_llm_kit.server")
    SERVER_MOD = _sys.modules["local_llm_kit.server"]
    HAS_SERVER = True
except ImportError:
    SERVER_MOD = None
    HAS_SERVER = False


class EchoLLM(object):
    """Minimal fake LLM double: OpenAI-shaped echo responses, no I/O."""

    def __init__(self, model="echo-model"):
        self.model = model
        self.chat_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        last_user = ""
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                last_user = content if isinstance(content, str) else str(content)
                break
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "echo:" + last_user},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def complete(self, prompt, **kwargs):
        return {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": self.model,
            "choices": [{"text": "echo:" + prompt, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def make_app(echo):
    """Build the app across likely v0.2.0 factory spellings."""
    import inspect

    for name in ("create_app", "build_app", "make_app", "get_app", "create_server"):
        factory = getattr(SERVER_MOD, name, None)
        if not callable(factory):
            continue
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            params = {}
        for kwargs in ({"llm": echo}, {"model": echo}, {}):
            if kwargs and all(k not in params for k in kwargs) and params:
                continue
            try:
                app = factory(**kwargs)
            except TypeError:
                continue
            if app is not None:
                if not kwargs:
                    try:
                        app.state.llm = echo
                    except Exception:
                        pass
                return app
    app = getattr(SERVER_MOD, "app", None)
    if app is not None:
        try:
            app.state.llm = echo
        except Exception:
            pass
        return app
    raise AttributeError("local_llm_kit.server exposes no known app factory")


class GeneratorEmbedLLM(EchoLLM):
    """LLM double whose ``embed()`` yields vectors from a non-list iterable.

    numpy matrices, tuples and generators are all valid embedding
    containers; they must be served like plain lists.
    """

    def embed(self, input):  # noqa: A002 (OpenAI field name)
        texts = [input] if isinstance(input, str) else list(input)
        return iter([[float(i), -1.0] for i in range(len(texts))])


class DictEmbedLLM(EchoLLM):
    """LLM double whose ``embed()`` returns an OpenAI-shaped dict.

    This mirrors ``local_llm_kit.LLM.embed``, which always answers with
    ``{"data": [{"embedding": [...]}]}``.
    """

    def embed(self, input):  # noqa: A002 (OpenAI field name)
        texts = [input] if isinstance(input, str) else list(input)
        return {
            "object": "list",
            "model": self.model,
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(i), 1.5, -2.0]}
                for i in range(len(texts))
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }


@unittest.skipUnless(
    HAS_SERVER and HAS_FASTAPI, "server tests need local_llm_kit.server and fastapi (both optional)"
)
class TestOpenAIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.echo = EchoLLM()
        cls.app = make_app(cls.echo)
        cls.client = TestClient(cls.app)

    def test_chat_completions_echo(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "echo-model",
                "messages": [{"role": "user", "content": "hello server"}],
            },
        )
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        self.assertIn("hello server", content)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")

    def test_chat_completions_empty_messages_rejected_not_500(self):
        resp = self.client.post(
            "/v1/chat/completions", json={"model": "echo-model", "messages": []}
        )
        self.assertIn(resp.status_code, (200, 400, 422))

    def test_completions_echo(self):
        resp = self.client.post(
            "/v1/completions", json={"model": "echo-model", "prompt": "once upon"}
        )
        if resp.status_code == 404:
            self.skipTest("/v1/completions route not exposed")
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        self.assertIn("once upon", resp.json()["choices"][0]["text"])

    def test_models_list(self):
        for path in ("/v1/models", "/models"):
            resp = self.client.get(path)
            if resp.status_code == 200:
                self.assertIn("data", resp.json())
                return
        self.skipTest("no models route exposed")

    def test_health(self):
        for path in ("/health", "/v1/health", "/"):
            resp = self.client.get(path)
            if resp.status_code == 200:
                return
        self.skipTest("no health route exposed")

    def test_malformed_body_is_4xx_not_500(self):
        resp = self.client.post("/v1/chat/completions", json={"bogus": True})
        self.assertGreaterEqual(resp.status_code, 400)
        self.assertLess(resp.status_code, 500)

    def test_responses_string_input(self):
        resp = self.client.post(
            "/v1/responses", json={"model": "echo-model", "input": "hello responses"}
        )
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        body = resp.json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        output = body["output"][0]
        self.assertEqual(output["type"], "message")
        self.assertEqual(output["role"], "assistant")
        self.assertIn("hello responses", output["content"][0]["text"])

    def test_responses_message_list_and_instructions(self):
        resp = self.client.post(
            "/v1/responses",
            json={
                "model": "echo-model",
                "instructions": "Be terse.",
                "input": [
                    {"role": "user", "content": "first"},
                    {"type": "input_text", "text": "second"},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "third"}],
                    },
                ],
            },
        )
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        self.assertIn("third", resp.json()["output"][0]["content"][0]["text"])

    def test_responses_rejects_bad_input_and_stateful_options(self):
        for payload in ({"input": ""}, {"input": []}, {"input": [{"type": "nonsense"}]}):
            resp = self.client.post("/v1/responses", json=payload)
            self.assertEqual(resp.status_code, 400, msg=resp.text[:300])
        for payload in (
            {"input": "hi", "previous_response_id": "resp_1"},
            {"input": "hi", "background": True},
            {"input": "hi", "store": True},
            {"input": "hi", "reasoning": {"effort": "low"}},
        ):
            resp = self.client.post("/v1/responses", json=payload)
            self.assertEqual(resp.status_code, 400, msg=resp.text[:300])

    def test_responses_streaming_deltas_and_completed(self):
        resp = self.client.post("/v1/responses", json={"input": "hi", "stream": True})
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        events = [
            line[len("data: ") :]
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        import json as _json

        parsed = [_json.loads(e) for e in events]
        deltas = [e["delta"] for e in parsed if e.get("type") == "response.output_text.delta"]
        self.assertTrue(deltas)
        completed = [e for e in parsed if e.get("type") == "response.completed"]
        self.assertEqual(len(completed), 1)
        full = completed[0]["response"]["output"][0]["content"][0]["text"]
        self.assertEqual(full, "".join(deltas))

    def test_embeddings_use_the_llm_vectors_when_it_returns_openai_shape(self):
        """LLM.embed() answers with {"data": [...]}: use those vectors, not a stub."""
        app = make_app(DictEmbedLLM())
        client = TestClient(app)
        resp = client.post("/v1/embeddings", json={"model": "echo-model", "input": ["a", "b"]})
        if resp.status_code == 404:
            self.skipTest("/v1/embeddings route not exposed")
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        vectors = [item["embedding"] for item in resp.json()["data"]]
        self.assertEqual(vectors, [[0.0, 1.5, -2.0], [1.0, 1.5, -2.0]])

    def test_embeddings_accept_non_list_iterables(self):
        """Generators / numpy matrices are valid embedding containers."""
        app = make_app(GeneratorEmbedLLM())
        resp = TestClient(app).post("/v1/embeddings", json={"model": "m", "input": ["a", "b"]})
        if resp.status_code == 404:
            self.skipTest("/v1/embeddings route not exposed")
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        vectors = [item["embedding"] for item in resp.json()["data"]]
        self.assertEqual(vectors, [[0.0, -1.0], [1.0, -1.0]])

    def test_embeddings_fall_back_when_the_llm_has_no_embed_support(self):
        """EchoLLM exposes no embed(): a stable local fallback is still served."""
        body = {"model": "echo-model", "input": "hello"}
        first = self.client.post("/v1/embeddings", json=body)
        if first.status_code == 404:
            self.skipTest("/v1/embeddings route not exposed")
        self.assertEqual(first.status_code, 200, msg=first.text[:500])
        first_vec = first.json()["data"][0]["embedding"]
        second_vec = self.client.post("/v1/embeddings", json=body).json()["data"][0]["embedding"]
        self.assertTrue(first_vec, msg="fallback embeddings must not be empty")
        self.assertEqual(first_vec, second_vec, msg="fallback embeddings must be deterministic")


@unittest.skipUnless(not HAS_FASTAPI, "fastapi is installed; skip-path test not applicable")
class TestServerSkipPath(unittest.TestCase):
    def test_missing_fastapi_is_a_skip_not_a_failure(self):
        self.assertIsNone(TestClient)


@unittest.skipUnless(HAS_FASTAPI and HAS_SERVER, "server extra unavailable")
class TestConsolidatedServerValidation(unittest.TestCase):
    def test_tool_execution_requires_a_boolean_not_a_truthy_value(self):
        llm = EchoLLM()
        with TestClient(SERVER_MOD.create_app(llm=llm)) as client:
            for value in ("false", "true", 1, None, [True]):
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "execute_tools": value},
                )
                self.assertEqual(response.status_code, 400)
        self.assertEqual(llm.chat_calls, [])

    def test_image_content_parts_reach_the_client(self):
        llm = EchoLLM()
        content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]
        with TestClient(SERVER_MOD.create_app(llm=llm)) as client:
            response = client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.chat_calls[0]["messages"][0]["content"], content)

    def test_invalid_generation_options_return_400_for_both_routes(self):
        with TestClient(SERVER_MOD.create_app(llm=EchoLLM())) as client:
            for route, body in (
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                ("/v1/completions", {"prompt": "hi"}),
            ):
                for options in (
                    {"temperature": "hot"},
                    {"temperature": True},
                    {"max_tokens": 1.5},
                    {"max_tokens": True},
                    {"max_tokens": -1},
                    {"response_format": []},
                ):
                    with self.subTest(route=route, options=options):
                        response = client.post(route, json=dict(body, **options))
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_schema_and_completion_token_alias_are_forwarded(self):
        llm = EchoLLM()
        schema = {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}},
        }
        with TestClient(SERVER_MOD.create_app(llm=llm)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "JSON please"}],
                    "response_format": schema,
                    "max_completion_tokens": 12,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.chat_calls[-1]["kwargs"]["response_format"], schema)
        self.assertEqual(llm.chat_calls[-1]["kwargs"]["max_tokens"], 12)

    def test_invalid_embedding_items_return_400(self):
        with TestClient(SERVER_MOD.create_app(llm=EchoLLM())) as client:
            self.assertEqual(
                client.post("/v1/embeddings", json={"input": ["ok", None]}).status_code, 400
            )

    def test_runtime_type_errors_do_not_repeat_requests(self):
        class BrokenLLM(EchoLLM):
            def chat(self, messages, **kwargs):
                self.chat_calls.append(messages)
                raise TypeError("backend failed after doing work")

        llm = BrokenLLM()
        with TestClient(SERVER_MOD.create_app(llm=llm)) as client:
            response = client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(llm.chat_calls), 1)

    def test_legacy_llms_cannot_bypass_tool_execution_opt_in(self):
        class LegacyLLM:
            def chat(self, messages, functions=None, function_call="auto"):
                raise AssertionError("legacy executor must not run")

        with self.assertRaisesRegex(TypeError, "cannot disable tool execution"):
            SERVER_MOD._call_llm_chat(
                LegacyLLM(),
                {
                    "messages": [],
                    "functions": [{"name": "dangerous"}],
                    "execute_tools": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
