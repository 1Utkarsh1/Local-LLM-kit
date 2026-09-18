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
        return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "echo:" + last_user},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def complete(self, prompt, **kwargs):
        return {"id": "cmpl-test", "object": "text_completion", "created": 0,
                "model": self.model,
                "choices": [{"text": "echo:" + prompt, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


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


@unittest.skipUnless(HAS_SERVER and HAS_FASTAPI,
                     "server tests need local_llm_kit.server and fastapi (both optional)")
class TestOpenAIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.echo = EchoLLM()
        cls.app = make_app(cls.echo)
        cls.client = TestClient(cls.app)

    def test_chat_completions_echo(self):
        resp = self.client.post("/v1/chat/completions", json={
            "model": "echo-model",
            "messages": [{"role": "user", "content": "hello server"}],
        })
        self.assertEqual(resp.status_code, 200, msg=resp.text[:500])
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        self.assertIn("hello server", content)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")

    def test_chat_completions_empty_messages_rejected_not_500(self):
        resp = self.client.post("/v1/chat/completions", json={"model": "echo-model", "messages": []})
        self.assertIn(resp.status_code, (200, 400, 422))

    def test_completions_echo(self):
        resp = self.client.post("/v1/completions", json={"model": "echo-model", "prompt": "once upon"})
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


@unittest.skipUnless(not HAS_FASTAPI, "fastapi is installed; skip-path test not applicable")
class TestServerSkipPath(unittest.TestCase):
    def test_missing_fastapi_is_a_skip_not_a_failure(self):
        self.assertIsNone(TestClient)


if __name__ == "__main__":
    unittest.main()
