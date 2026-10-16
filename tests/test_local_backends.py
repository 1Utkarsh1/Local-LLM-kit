"""Local server presets and AirLLM adapter contracts; no models or GPU needed."""

import subprocess
import sys
import types
import unittest
from unittest import mock

from local_llm_kit import LLM
from local_llm_kit.backends import OPENAI_SERVER_URLS, get_backend
from local_llm_kit.backends.airllm import AirLLMBackend
from local_llm_kit.cli import _build_backend_kwargs, _normalize_backend, build_parser


class TestServerPresets(unittest.TestCase):
    def test_every_preset_is_accepted_by_cli_library_and_registry(self):
        for name, url in OPENAI_SERVER_URLS.items():
            with self.subTest(name=name):
                args = build_parser().parse_args(
                    ["chat", "--backend", name, "--model", "local-model"]
                )
                self.assertEqual(_normalize_backend(args.backend), name)
                client = LLM("local-model", backend=name, use_cache=False)
                self.assertEqual(client.backend.base_url, url)
                self.assertEqual(get_backend(name, model="local-model").base_url, url)
                override = LLM(
                    "local-model",
                    backend=name,
                    use_cache=False,
                    backend_kwargs={"base_url": "http://127.0.0.1:9000/v1"},
                )
                self.assertEqual(override.backend.base_url, "http://127.0.0.1:9000/v1")

    def test_cli_server_configuration_and_ollama_host_are_forwarded(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            for name in OPENAI_SERVER_URLS:
                args = build_parser().parse_args(
                    [
                        "chat",
                        "--backend",
                        name,
                        "--model",
                        "m",
                        "--base-url",
                        "http://localhost:9000/v1",
                        "--api-key",
                        "test-key",
                    ]
                )
                self.assertEqual(
                    _build_backend_kwargs(args),
                    {"base_url": "http://localhost:9000/v1", "api_key": "test-key"},
                )
            args = build_parser().parse_args(
                [
                    "chat",
                    "--backend",
                    "ollama",
                    "--model",
                    "m",
                    "--base-url",
                    "http://localhost:11500",
                ]
            )
            self.assertEqual(_build_backend_kwargs(args), {"host": "http://localhost:11500"})

    def test_server_presets_use_their_sampling_field_names(self):
        for name in ("vllm", "sglang", "mlx-lm"):
            backend = get_backend(name, model="local-model")
            with mock.patch.object(
                backend, "_post_json", return_value={"choices": [{"message": {"content": "ok"}}]}
            ) as post:
                make = LLM("local-model", backend_instance=backend, use_cache=False)
                make.chat([{"role": "user", "content": "hi"}], repetition_penalty=1.2)
            self.assertEqual(post.call_args.args[0], "/chat/completions")
            self.assertEqual(post.call_args.args[1]["repetition_penalty"], 1.2)
            self.assertNotIn("repeat_penalty", post.call_args.args[1])

    def test_chat_only_servers_reject_embeddings_without_a_request(self):
        for name in ("gpt4all", "mlx-lm"):
            client = LLM("local-model", backend=name, use_cache=False)
            self.assertFalse(client.backend.supports_embeddings)
            with mock.patch.object(client.backend, "_post_json") as post:
                with self.assertRaisesRegex(NotImplementedError, "separate embedding backend"):
                    client.embed("hello")
            post.assert_not_called()

    def test_extra_backend_modules_do_not_load_heavy_dependencies(self):
        code = "import sys; import local_llm_kit.backends.airllm; from local_llm_kit import LLM; LLM('m', backend='sglang', use_cache=False); assert not set(('torch', 'transformers', 'airllm', 'mlx')).intersection(sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class Tensor(list):
    def to(self, device):
        self.device = device
        return self


class Tokenizer:
    def __call__(self, text, **kwargs):
        self.options = kwargs
        return {"input_ids": Tensor([[1, 2]])}

    def decode(self, tokens, **kwargs):
        return "".join({3: "answer", 4: " END", 5: " ignored"}[token] for token in tokens)

    def encode(self, text):
        return list(range(len(text.split())))


class FakeAirModel:
    device = "cpu"

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.calls = []

    def generate(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return types.SimpleNamespace(sequences=[[1, 2, 3, 4, 5]])


class TestAirLLMAdapter(unittest.TestCase):
    def setUp(self):
        self.model = FakeAirModel()
        self.auto = mock.Mock()
        self.auto.from_pretrained.return_value = self.model
        patcher = mock.patch.dict(
            sys.modules, {"airllm": types.SimpleNamespace(AutoModel=self.auto)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_initialization_generation_and_stop_preserve_expected_contract(self):
        backend = AirLLMBackend("/trusted/model", device="cpu", max_seq_len=32, compression=None)
        self.auto.from_pretrained.assert_called_once_with(
            "/trusted/model", device="cpu", max_seq_len=32, compression=None
        )
        output = backend.generate("question", max_new_tokens=5, stop=" END", temperature=0)
        self.assertEqual(output, {"text": "answer", "finish_reason": "stop"})
        inputs, options = self.model.calls[0]
        self.assertEqual(inputs.device, "cpu")
        self.assertFalse(options["do_sample"])
        self.assertTrue(options["return_dict_in_generate"])
        self.assertFalse(self.model.tokenizer.options["truncation"])
        self.assertEqual(backend.count_tokens("one two"), 2)

    def test_library_registry_cli_and_buffered_stream_reach_airllm(self):
        client = LLM(
            "/trusted/model", backend="airllm", backend_kwargs={"device": "cpu"}, use_cache=False
        )
        response = client.complete("question", max_tokens=5)
        self.assertIn("answer", response["choices"][0]["text"])
        backend = get_backend("airllm", model_path="/trusted/model", device="cpu")
        chunks = list(backend.generate_stream("question", max_new_tokens=5))
        self.assertEqual(len(chunks), 1)
        args = build_parser().parse_args(
            ["complete", "--backend", "airllm", "--model", "/trusted/model", "--device", "cpu"]
        )
        self.assertEqual(_build_backend_kwargs(args)["device"], "cpu")

    def test_context_overflow_and_unsupported_options_are_explicit(self):
        backend = AirLLMBackend("/trusted/model", max_seq_len=4)
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            backend.generate("question", max_new_tokens=5)
        for options in ({"logprobs": True}, {"seed": 42}):
            with self.assertRaises(NotImplementedError):
                backend.generate("question", **options)
        self.assertEqual(backend.generate("question", max_new_tokens=0)["text"], "")
        self.assertEqual(self.model.calls, [])

    def test_mlx_api_accepts_array_inputs_and_returns_text(self):
        model_type = type("AirLLMLlamaMlx", (), {"__module__": "airllm.airllm_llama_mlx"})
        model = model_type()
        model.tokenizer = Tokenizer()
        model.generate = mock.Mock(return_value="answer")
        self.auto.from_pretrained.return_value = model
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        core = types.ModuleType("mlx.core")
        core.array = lambda value: value
        mlx.core = core
        with mock.patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core}):
            backend = AirLLMBackend("/trusted/model")
            self.assertEqual(backend.generate("question", max_new_tokens=5)["text"], "answer")
        self.assertEqual(model.tokenizer.options["return_tensors"], "np")

    def test_missing_extra_has_an_install_hint(self):
        with mock.patch.dict(sys.modules, {"airllm": None}):
            with self.assertRaisesRegex(ImportError, r"local-llm-kit\[airllm\]"):
                AirLLMBackend("/trusted/model")


if __name__ == "__main__":
    unittest.main()
