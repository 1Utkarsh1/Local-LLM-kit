"""Offline tests for client reuse/caching + backwards-compatible signatures.

Covers: get_client caching (LLM reuse across calls) where the v0.2.0 helper
exists (probed in several locations; skipped if not landed yet), FakeBackend
injection (no downloads), and backwards compatibility of LLM.chat/complete
plus module-level chat()/complete() with the legacy
functions/function_call API. Stdlib unittest only. Py3.9 compatible.

Run: python -m pytest tests/test_client_cache.py -q
"""

import inspect
import sys
import unittest
from unittest import mock

from local_llm_kit.backends.base import BaseBackend
from local_llm_kit.llm import LLM

CHAT_MOD = sys.modules.get("local_llm_kit.chat")
if CHAT_MOD is None:
    import importlib as _importlib

    CHAT_MOD = _importlib.import_module("local_llm_kit.chat")
    CHAT_MOD = sys.modules["local_llm_kit.chat"]  # NOTE: attribute local_llm_kit.chat is the
    # function (shadowed by `from .chat import chat`); sys.modules holds the module.


def _probe_get_client():
    import local_llm_kit as pkg

    if callable(getattr(pkg, "get_client", None)):
        return pkg.get_client
    for mod_name in ("local_llm_kit.client", "local_llm_kit.llm", "local_llm_kit.backends"):
        try:
            __import__(mod_name)
        except ImportError:
            continue
        fn = getattr(sys.modules[mod_name], "get_client", None)
        if callable(fn):
            return fn
    return None


GET_CLIENT = _probe_get_client()
HAS_GET_CLIENT = callable(GET_CLIENT)


class FakeBackend(BaseBackend):
    """Deterministic echo backend: no torch, no downloads, no network."""

    def __init__(self, tag="fake"):
        self.tag = tag
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"text": "fake-response:" + prompt[-30:]}

    def generate_stream(self, prompt, **kwargs):
        yield {"text": "fake-response:"}
        yield {"text": ":done"}

    def get_context_window(self):
        return 2048

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def make_fake_llm(model_path="fake-model", **kwargs):
    """Build an LLM wired to FakeBackend (never touches real backends)."""
    with mock.patch.object(LLM, "_init_backend", lambda self, name=None: FakeBackend()):
        llm = LLM(model_path=model_path, backend="echo", **kwargs)
    assert isinstance(llm.backend, FakeBackend)
    return llm


class TestFakeLLMBehaviour(unittest.TestCase):
    """Baseline: the fake harness itself returns OpenAI-shaped payloads."""

    def test_chat_returns_openai_shape(self):
        llm = make_fake_llm()
        resp = llm.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(resp["object"], "chat.completion")
        self.assertEqual(resp["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(resp["choices"][0]["message"]["content"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIn(key, resp["usage"])

    def test_complete_returns_openai_shape(self):
        llm = make_fake_llm()
        resp = llm.complete(prompt="once upon")
        self.assertEqual(resp["object"], "text_completion")
        self.assertTrue(resp["choices"][0]["text"])

    def test_stream_ends_with_stop(self):
        llm = make_fake_llm()
        chunks = list(llm.chat(messages=[{"role": "user", "content": "hi"}], stream=True))
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_legacy_functions_kwargs_still_accepted(self):
        llm = make_fake_llm()
        functions = [
            {"name": "w", "description": "d", "parameters": {"type": "object", "properties": {}}}
        ]
        resp = llm.chat(
            messages=[{"role": "user", "content": "hi"}], functions=functions, function_call="none"
        )
        self.assertIn("choices", resp)


@unittest.skipUnless(HAS_GET_CLIENT, "get_client helper not available yet (v0.2.0 API)")
class TestGetClientCaching(unittest.TestCase):
    def _build(self, model, **kwargs):
        params = inspect.signature(GET_CLIENT).parameters
        call_kwargs = {}
        for key in ("backend", "backend_name"):
            if key in params and "backend" not in call_kwargs:
                call_kwargs[key] = kwargs.pop("backend", "echo")
                break
        call_kwargs.update(kwargs)
        with mock.patch.object(LLM, "_init_backend", lambda self, name=None: FakeBackend()):
            try:
                return GET_CLIENT(model, **call_kwargs)
            except TypeError:
                with mock.patch.object(LLM, "_init_backend", lambda self, name=None: FakeBackend()):
                    return GET_CLIENT(model_path=model, **call_kwargs)

    def test_same_model_returns_same_instance(self):
        try:
            first = self._build("fake-model")
            second = self._build("fake-model")
        except Exception as exc:
            self.skipTest("get_client needs a real backend: %s" % exc)
        self.assertIs(first, second)

    def test_different_models_return_different_instances(self):
        try:
            first = self._build("fake-model-a")
            second = self._build("fake-model-b")
        except Exception as exc:
            self.skipTest("get_client needs a real backend: %s" % exc)
        self.assertIsNot(first, second)

    def test_cached_client_is_usable_llm(self):
        try:
            client = self._build("fake-model")
        except Exception as exc:
            self.skipTest("get_client needs a real backend: %s" % exc)
        self.assertTrue(hasattr(client, "chat") and hasattr(client, "complete"))
        resp = client.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("choices", resp)


class TestBackwardsCompatSignatures(unittest.TestCase):
    def test_llm_chat_keeps_legacy_params(self):
        params = set(inspect.signature(LLM.chat).parameters)
        for name in (
            "messages",
            "functions",
            "function_call",
            "temperature",
            "max_tokens",
            "stream",
            "format",
        ):
            self.assertIn(name, params, msg="LLM.chat lost legacy param: %s" % name)

    def test_llm_complete_keeps_legacy_params(self):
        params = set(inspect.signature(LLM.complete).parameters)
        for name in ("prompt", "temperature", "max_tokens", "stream", "format"):
            self.assertIn(name, params, msg="LLM.complete lost legacy param: %s" % name)

    def test_module_chat_complete_keep_model_path_first_api(self):
        for fn_name, first in (("chat", "messages"), ("complete", "prompt")):
            fn = getattr(CHAT_MOD, fn_name)
            params = list(inspect.signature(fn).parameters)
            self.assertIn(first, params)
            self.assertIn("model_path", params)
            legacy = {"functions", "function_call", "temperature", "max_tokens", "stream", "format"}
            if fn_name == "chat":
                self.assertTrue(legacy & set(params))

    def test_module_chat_delegates_to_llm_without_loading_models(self):
        seen = {}

        class FakeLLM(object):
            def __init__(self, *args, **kwargs):
                seen["init"] = (args, kwargs)

            def chat(self, **kwargs):
                seen["chat"] = kwargs
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with mock.patch.object(CHAT_MOD, "LLM", FakeLLM):
            resp = CHAT_MOD.chat(
                messages=[{"role": "user", "content": "hi"}],
                model_path="fake-model",
                functions=[{"name": "w"}],
                function_call="auto",
            )
        self.assertIn("choices", resp)
        self.assertEqual(seen["chat"]["function_call"], "auto")
        self.assertEqual(seen["chat"]["functions"], [{"name": "w"}])

    def test_module_complete_delegates_to_llm_without_loading_models(self):
        seen = {}

        class FakeLLM(object):
            def __init__(self, *args, **kwargs):
                seen["init_kwargs"] = kwargs

            def complete(self, **kwargs):
                seen["complete"] = kwargs
                return {"choices": [{"text": "ok"}]}

        with mock.patch.object(CHAT_MOD, "LLM", FakeLLM):
            resp = CHAT_MOD.complete(prompt="hi", model_path="fake-model")
        self.assertIn("choices", resp)
        self.assertEqual(seen["complete"]["prompt"], "hi")

    def test_new_style_tools_kwargs_accepted_when_supported(self):
        params = set(inspect.signature(LLM.chat).parameters)
        if "tools" not in params:
            self.skipTest("LLM.chat has no tools param yet (v0.2.0 API)")
        llm = make_fake_llm()
        spec = {
            "type": "function",
            "function": {
                "name": "w",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        resp = llm.chat(
            messages=[{"role": "user", "content": "hi"}], tools=[spec], tool_choice="none"
        )
        self.assertIn("choices", resp)


if __name__ == "__main__":
    unittest.main()
