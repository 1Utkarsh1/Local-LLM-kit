"""Apple Foundation Models backend + apfel bridge contracts.

All tests run offline with a fake ``apple_fm_sdk`` module (no macOS, Xcode,
or on-device model needed). They verify the adapter contract against the
real SDK surface (``apple/apple-fm-sdk``, PyPI ``apple-fm-sdk`` 0.2.1):

* platform / interpreter / SDK / availability guards with actionable errors
* generate / chat / streaming delta conversion / stop truncation
* native ``json_schema`` guided generation via ``GeneratedContent.to_json()``
* sampling mapping (temperature 0 -> greedy, seed -> random)
* per-request session isolation (backends are shared via the LLM cache)
* tool specs travel in the prompt; no automatic in-framework execution
* apfel bridge profile: no top_k/penalty fields, chat-only, no embeddings

A live on-device test runs only with
``LOCAL_LLM_KIT_APPLE_FM_LIVE=1`` on an eligible Mac; it is skipped
everywhere else (including CI).
"""

import asyncio
import enum
import os
import sys
import types
import unittest
from unittest import mock

from local_llm_kit import LLM
from local_llm_kit.backends import BACKEND_ALIASES, BACKENDS, OPENAI_SERVER_URLS, get_backend
from local_llm_kit.backends.apple_fm import APPLE_MODEL_ID, AppleFMBackend
from local_llm_kit.backends.openai_compat import OpenAICompatBackend

# ---------------------------------------------------------------------------
# Fake apple_fm_sdk mirroring the real package surface
# (SystemLanguageModel / LanguageModelSession / GenerationOptions /
# SamplingMode / GeneratedContent.to_json / enums).
# ---------------------------------------------------------------------------


class FakeUseCase(enum.Enum):
    GENERAL = 0
    CONTENT_TAGGING = 1


class FakeGuardrails(enum.Enum):
    DEFAULT = 0
    PERMISSIVE_CONTENT_TRANSFORMATIONS = 1


class FakeReason(enum.Enum):
    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2
    UNKNOWN = 255


class FakeSampling:
    def __init__(self, mode, top=None, probability_threshold=None, seed=None):
        self.mode = mode
        self.top = top
        self.probability_threshold = probability_threshold
        self.seed = seed


class FakeSamplingMode:
    @classmethod
    def greedy(cls):
        return FakeSampling("greedy")

    @classmethod
    def random(cls, top=None, probability_threshold=None, seed=None):
        return FakeSampling("random", top, probability_threshold, seed)


class FakeOptions:
    def __init__(self, temperature=None, sampling=None, maximum_response_tokens=None):
        self.temperature = temperature
        self.sampling = sampling
        self.maximum_response_tokens = maximum_response_tokens


class FakeGeneratedContent:
    def __init__(self, payload):
        self.payload = payload

    def to_json(self):
        import json

        return json.dumps(self.payload)


class FakeSystemModel:
    available = True
    reason = None
    window = 4096
    variant_name = None
    token_count_calls = []

    def __init__(self, use_case=None, guardrails=None):
        self.use_case = use_case
        self.guardrails = guardrails

    def is_available(self):
        return FakeSystemModel.available, FakeSystemModel.reason

    @property
    def context_size(self):
        return FakeSystemModel.window

    @property
    def variant(self):
        if FakeSystemModel.variant_name is None:
            return None
        return types.SimpleNamespace(display_name=FakeSystemModel.variant_name)

    async def token_count(self, value=None, *, instructions=None):
        FakeSystemModel.token_count_calls.append((value, instructions))
        text = instructions if instructions is not None else value
        return max(1, len(str(text)) // 4)


class FakeImageAttachment:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as fh:
            self.data = fh.read()


class FakeSession:
    instances = []
    snapshots = ["Hello", "Hello, world", "Hello, world!"]
    fail_with = None

    def __init__(self, instructions=None, model=None, tools=None, _ptr=None):
        self.instructions = instructions
        self.model = model
        self.tools = tools
        self.calls = []
        FakeSession.instances.append(self)

    async def respond(
        self, prompt, generating=None, *, schema=None, json_schema=None, options=None
    ):
        self.calls.append(
            {"prompt": prompt, "schema": schema, "json_schema": json_schema, "options": options}
        )
        if FakeSession.fail_with is not None:
            raise FakeSession.fail_with
        if json_schema is not None:
            return FakeGeneratedContent({"ok": True, "echo": str(prompt)[:24]})
        if isinstance(prompt, list):
            texts = [p for p in prompt if isinstance(p, str)]
            images = [p for p in prompt if not isinstance(p, str)]
            return "Echo: %s [%d image(s)]" % (" ".join(texts), len(images))
        return "Echo: %s" % prompt

    async def stream_response(self, prompt, options=None):
        self.calls.append({"prompt": prompt, "options": options, "stream": True})
        for snapshot in FakeSession.snapshots:
            yield snapshot


def make_fake_sdk():
    module = types.ModuleType("apple_fm_sdk")
    module.SystemLanguageModel = FakeSystemModel
    module.SystemLanguageModelUseCase = FakeUseCase
    module.SystemLanguageModelGuardrails = FakeGuardrails
    module.LanguageModelSession = FakeSession
    module.ImageAttachment = FakeImageAttachment
    module.GenerationOptions = FakeOptions
    module.SamplingMode = FakeSamplingMode
    module.GeneratedContent = FakeGeneratedContent
    return module


def install_fake_sdk():
    return mock.patch.dict(sys.modules, {"apple_fm_sdk": make_fake_sdk()})


def mac_only():
    return mock.patch.object(sys, "platform", "darwin"), mock.patch.object(
        sys, "version_info", (3, 10, 0, "final", 0)
    )


class AppleFMBackendTest(unittest.TestCase):
    def setUp(self):
        FakeSystemModel.available = True
        FakeSystemModel.reason = None
        FakeSystemModel.window = 4096
        FakeSystemModel.variant_name = None
        FakeSystemModel.token_count_calls = []
        FakeSession.instances = []
        FakeSession.fail_with = None
        self._patches = []

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()

    def _mac(self):
        for patcher in mac_only():
            patcher.start()
            self._patches.append(patcher)
        sdk = install_fake_sdk()
        sdk.start()
        self._patches.append(sdk)

    def test_registry_and_aliases(self):
        self.assertEqual(BACKEND_ALIASES["apple"], "apple-fm")
        self.assertEqual(BACKEND_ALIASES["apfel"], "openai-compat")
        for alias in ("apple-foundation", "foundation-models", "foundationmodels"):
            self.assertEqual(BACKEND_ALIASES[alias], "apple-fm")
        self.assertIn("apple-fm", BACKENDS)
        self.assertIn("apfel", OPENAI_SERVER_URLS)
        # Every preset (incl. apfel) is accepted by the registry + LLM routing.
        backend = get_backend("apfel", model="apple-foundationmodel")
        self.assertEqual(backend.base_url, OPENAI_SERVER_URLS["apfel"])
        self.assertEqual(backend.provider, "apfel")

    def test_platform_guard_fires_before_sdk_import(self):
        with mock.patch.object(sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "only on macOS"):
                AppleFMBackend("apple-foundationmodel")

    def test_missing_sdk_gives_install_hint(self):
        self._mac()
        with mock.patch.dict(sys.modules, {"apple_fm_sdk": None}):
            with self.assertRaisesRegex(ImportError, "local-llm-kit\\[apple-fm\\]"):
                AppleFMBackend("apple-foundationmodel")

    def test_llm_wraps_missing_sdk_as_runtime_error(self):
        self._mac()
        with mock.patch.dict(sys.modules, {"apple_fm_sdk": None}):
            with self.assertRaisesRegex(RuntimeError, "apple-fm backend needs"):
                LLM("apple-foundationmodel", backend="apple-fm", use_cache=False)

    def test_unavailable_model_reports_actionable_reason(self):
        self._mac()
        FakeSystemModel.available = False
        FakeSystemModel.reason = FakeReason.MODEL_NOT_READY
        with self.assertRaisesRegex(RuntimeError, "still downloading"):
            AppleFMBackend("apple-foundationmodel")

    def test_is_available_preflight_never_raises(self):
        self._mac()
        ok, reason = AppleFMBackend.is_available()
        self.assertTrue(ok)
        self.assertIsNone(reason)
        FakeSystemModel.available = False
        FakeSystemModel.reason = FakeReason.DEVICE_NOT_ELIGIBLE
        ok, reason = AppleFMBackend.is_available()
        self.assertFalse(ok)
        self.assertIn("not eligible", reason)

    def test_generate_roundtrip_and_model_id(self):
        self._mac()
        backend = AppleFMBackend("apple-foundationmodel")
        result = backend.generate("hi there", max_new_tokens=32)
        self.assertEqual(result["text"], "Echo: hi there")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["model"], APPLE_MODEL_ID)

    def test_invalid_options_rejected(self):
        self._mac()
        with self.assertRaisesRegex(ValueError, "use_case"):
            AppleFMBackend("m", use_case="summarize-everything")
        with self.assertRaisesRegex(ValueError, "guardrails"):
            AppleFMBackend("m", guardrails="none")

    def test_sampling_mapping(self):
        self._mac()
        backend = AppleFMBackend("m")
        greedy = backend._options(0, 64)
        self.assertEqual(greedy.sampling.mode, "greedy")
        self.assertEqual(greedy.maximum_response_tokens, 64)
        seeded = backend._options(0.7, 64, seed=42)
        self.assertEqual(seeded.sampling.mode, "random")
        self.assertEqual(seeded.sampling.seed, 42)
        plain = backend._options(0.7, 64)
        self.assertEqual(plain.temperature, 0.7)
        self.assertIsNone(plain.sampling)

    def test_chat_splits_system_into_instructions(self):
        self._mac()
        backend = AppleFMBackend("m")
        out = backend.chat(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Say hi."},
            ]
        )
        self.assertIn("Say hi.", out["text"])
        session = FakeSession.instances[-1]
        self.assertEqual(session.instructions, "Be terse.")
        self.assertIn("Say hi.", session.calls[0]["prompt"])

    def test_each_request_gets_a_fresh_session(self):
        self._mac()
        backend = AppleFMBackend("m")
        backend.generate("one")
        backend.generate("two")
        self.assertEqual(len(FakeSession.instances), 2)
        prompts = [s.calls[0]["prompt"] for s in FakeSession.instances]
        self.assertEqual(prompts, ["one", "two"])

    def test_native_json_schema_guided_generation(self):
        self._mac()
        backend = AppleFMBackend("m")
        out = backend.chat(
            [{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reply",
                    "schema": {"type": "object", "properties": {}},
                },
            },
        )
        import json

        self.assertEqual(json.loads(out["text"])["ok"], True)
        session = FakeSession.instances[-1]
        self.assertEqual(session.calls[0]["json_schema"], {"type": "object", "properties": {}})

    def test_streaming_yields_deltas_not_snapshots(self):
        self._mac()
        backend = AppleFMBackend("m")
        chunks = list(backend.generate_stream("hi"))
        self.assertEqual([c["text"] for c in chunks], ["Hello", ", world", "!"])
        self.assertEqual("".join(c["text"] for c in chunks), "Hello, world!")

    def test_stop_sequences_truncate(self):
        self._mac()
        backend = AppleFMBackend("m")
        out = backend.generate("say world hello", stop=["world"])
        self.assertEqual(out["text"], "Echo: say ")

    def test_tools_documented_in_prompt_without_auto_execution(self):
        self._mac()
        backend = AppleFMBackend("m")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "City weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        out = backend.chat([{"role": "user", "content": "weather?"}], tools=tools)
        session = FakeSession.instances[-1]
        self.assertIsNone(session.tools)
        self.assertIn("get_weather", session.calls[0]["prompt"])
        self.assertIn("weather?", out["text"])

    def test_full_llm_tool_loop_parses_text_calls(self):
        self._mac()

        class Scripted(AppleFMBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.turn = 0

            def chat(self, *args, **kwargs):
                self.turn += 1
                if self.turn == 1:
                    return {
                        "text": 'Let me check. <tool_call>{"name": "add", '
                        '"arguments": {"a": 2, "b": 3}}</tool_call>',
                        "finish_reason": "stop",
                        "model": self.model,
                    }
                return {"text": "5", "finish_reason": "stop", "model": self.model}

        llm = LLM("m", backend_instance=Scripted("m"), use_cache=False)
        llm.add_tool(lambda a, b: a + b, name="add")
        out = llm.chat(
            [{"role": "user", "content": "add 2 and 3"}],
            tools=[{"type": "function", "function": {"name": "add"}}],
        )
        self.assertIn("5", out["choices"][0]["message"]["content"])

    def test_sync_bridge_inside_a_running_loop(self):
        self._mac()
        backend = AppleFMBackend("m")

        async def _main():
            return backend.generate("from-loop")

        self.assertEqual(asyncio.run(_main())["text"], "Echo: from-loop")

    def test_embeddings_unsupported(self):
        self._mac()
        backend = AppleFMBackend("m")
        self.assertFalse(backend.supports_embeddings)
        with self.assertRaises(NotImplementedError):
            backend.embed(["hello"])

    def test_model_info_and_list_models(self):
        self._mac()
        FakeSystemModel.variant_name = "AFM 3 Core"
        backend = AppleFMBackend("m")
        info = backend.model_info()
        self.assertEqual(info["backend"], "apple-fm")
        self.assertTrue(info["on_device"])
        self.assertFalse(info["requires_network"])
        self.assertEqual(info["variant"], "AFM 3 Core")
        self.assertEqual(info["context_window"], 4096)
        models = backend.list_models()
        self.assertEqual(models[0]["owned_by"], "apple")

    def test_context_window_override(self):
        self._mac()
        FakeSystemModel.window = 8192
        self.assertEqual(AppleFMBackend("m").get_context_window(), 8192)
        self.assertEqual(AppleFMBackend("m", context_window=2048).get_context_window(), 2048)

    def test_close_is_safe(self):
        self._mac()
        backend = AppleFMBackend("m")
        backend.generate("hi")
        backend.close()
        self.assertTrue(backend._closed)

    def test_text_only_prompt_stays_a_plain_string(self):
        self._mac()
        backend = AppleFMBackend("m")
        backend.chat([{"role": "user", "content": "hello"}])
        prompt = FakeSession.instances[-1].calls[0]["prompt"]
        self.assertIsInstance(prompt, str)
        self.assertIn("User: hello", prompt)

    def test_vision_data_url_materialized_and_cleaned_up(self):
        import base64 as _b64

        self._mac()
        backend = AppleFMBackend("m")
        self.assertTrue(backend.supports_vision)
        raw = _b64.b64encode(b"\x89PNG-fake-bytes").decode()
        out = backend.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + raw},
                        },
                    ],
                }
            ]
        )
        self.assertIn("[1 image(s)]", out["text"])
        session = FakeSession.instances[-1]
        prompt = session.calls[0]["prompt"]
        self.assertIsInstance(prompt, list)
        attachments = [p for p in prompt if isinstance(p, FakeImageAttachment)]
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].data, b"\x89PNG-fake-bytes")
        # Temp files are removed after the request completes.
        self.assertFalse(os.path.exists(attachments[0].path))

    def test_vision_remote_url_rejected_without_download(self):
        self._mac()
        backend = AppleFMBackend("m")
        with self.assertRaisesRegex(ValueError, "never downloaded"):
            backend.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/a.png"},
                            }
                        ],
                    }
                ]
            )
        self.assertEqual(FakeSession.instances, [])

    def test_vision_oversize_image_rejected(self):
        import base64 as _b64

        self._mac()
        backend = AppleFMBackend("m")
        with mock.patch.object(backend._fm, "ImageAttachment", FakeImageAttachment):
            import local_llm_kit.backends.apple_fm as apple_mod

            with mock.patch.object(apple_mod, "_MAX_IMAGE_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "limit"):
                    backend.chat(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + _b64.b64encode(b"12345").decode()
                                        },
                                    }
                                ],
                            }
                        ]
                    )

    def test_exact_token_count_uses_sdk(self):
        self._mac()
        backend = AppleFMBackend("m")
        self.assertEqual(backend.count_tokens_exact("hello world, this is a test"), 6)
        self.assertEqual(
            FakeSystemModel.token_count_calls[-1], ("hello world, this is a test", None)
        )
        self.assertEqual(backend.count_tokens_exact(instructions="be terse"), 2)
        self.assertEqual(FakeSystemModel.token_count_calls[-1], (None, "be terse"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            backend.count_tokens_exact()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            backend.count_tokens_exact("a", instructions="b")

    def test_serve_end_to_end_with_apple_backend(self):
        try:
            from fastapi.testclient import TestClient

            from local_llm_kit.server import create_app
        except ImportError:
            self.skipTest("fastapi not installed")
        sdk = install_fake_sdk()
        sdk.start()
        self._patches.append(sdk)
        # Platform/interpreter patches apply only to backend construction:
        # FastAPI inspects the real interpreter version at route setup.
        with (
            mock.patch.object(sys, "platform", "darwin"),
            mock.patch.object(sys, "version_info", (3, 10, 0, "final", 0)),
        ):
            llm = LLM("apple-foundationmodel", backend="apple-fm", use_cache=False)
        client = TestClient(create_app(llm=llm))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "apple-foundationmodel",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Echo:", resp.json()["choices"][0]["message"]["content"])


class ApfelBridgeTest(unittest.TestCase):
    def test_chat_payload_omits_rejected_fields(self):
        backend = OpenAICompatBackend("apple-foundationmodel", provider="apfel")
        payload = backend._chat_payload(
            [{"role": "user", "content": "hi"}],
            0.7,
            128,
            0.95,
            40,
            1.1,
            False,
            None,
            None,
            None,
            None,
            None,
        )
        for banned in ("top_k", "repeat_penalty", "repetition_penalty"):
            self.assertNotIn(banned, payload)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 128)

    def test_completion_payload_omits_rejected_fields(self):
        backend = OpenAICompatBackend("apple-foundationmodel", provider="apfel")
        payload = backend._completion_payload("hi", 0.7, 64, 0.95, 40, 1.1, False)
        for banned in ("top_k", "repeat_penalty", "repetition_penalty"):
            self.assertNotIn(banned, payload)

    def test_other_providers_keep_sampling_fields(self):
        backend = OpenAICompatBackend("m", provider="vllm")
        payload = backend._chat_payload(
            [{"role": "user", "content": "hi"}],
            0.7,
            128,
            0.95,
            40,
            1.1,
            False,
            None,
            None,
            None,
            None,
            None,
        )
        self.assertEqual(payload["repetition_penalty"], 1.1)

    def test_prompt_completion_rejected_with_chat_pointer(self):
        backend = OpenAICompatBackend("apple-foundationmodel", provider="apfel")
        with self.assertRaisesRegex(RuntimeError, "chat\\(\\)"):
            backend.generate("hi")
        with self.assertRaisesRegex(RuntimeError, "chat_stream"):
            list(backend.generate_stream("hi"))

    def test_apfel_has_no_embeddings_endpoint(self):
        backend = OpenAICompatBackend("apple-foundationmodel", provider="apfel")
        self.assertFalse(backend.supports_embeddings)
        with self.assertRaisesRegex(NotImplementedError, "embeddings"):
            backend.embed(["hello"])

    def test_apfel_preset_routes_through_llm(self):
        llm = LLM("apple-foundationmodel", backend="apfel", use_cache=False)
        self.assertIsInstance(llm.backend, OpenAICompatBackend)
        self.assertEqual(llm.backend.base_url, OPENAI_SERVER_URLS["apfel"])
        with mock.patch.object(
            llm.backend,
            "_post_json",
            return_value={"choices": [{"message": {"content": "served"}}]},
        ) as post:
            out = llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["choices"][0]["message"]["content"], "served")
        sent = post.call_args.args[1]
        for banned in ("top_k", "repeat_penalty", "repetition_penalty"):
            self.assertNotIn(banned, sent)

    def test_apfel_complete_rejected_end_to_end(self):
        llm = LLM("apple-foundationmodel", backend="apfel", use_cache=False)
        with self.assertRaisesRegex(RuntimeError, "chat\\(\\)"):
            llm.complete("hi")


@unittest.skipUnless(
    os.environ.get("LOCAL_LLM_KIT_APPLE_FM_LIVE") == "1",
    "opt-in live test: needs an eligible Mac (macOS 26+, Xcode 26+, "
    "Apple Intelligence on). Not run in CI.",
)
class AppleFMLiveTest(unittest.TestCase):
    def test_live_generate(self):
        backend = AppleFMBackend("apple-foundationmodel")
        result = backend.generate("Reply with the single word: ready", max_new_tokens=16)
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["text"].strip())


if __name__ == "__main__":
    unittest.main()
