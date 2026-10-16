"""Regression cases retained and strengthened while consolidating PRs #2--#6."""

import concurrent.futures
import importlib
import inspect
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_llm_kit import LLM, models
from local_llm_kit.cli import _adapt_download_call, _instantiate_app, _parse_choice, build_parser
from local_llm_kit.embeddings import EmbeddingClient, hash_embed
from local_llm_kit.function_calling import FunctionCall, parse_function_calls
from local_llm_kit.memory import MessageHistory
from local_llm_kit.prompt_formatting import (
    ChatMLPromptFormatter,
    Llama2ChatPromptFormatter,
    MistralInstructPromptFormatter,
    PlainInstructPromptFormatter,
    VicunaPromptFormatter,
)
from local_llm_kit.rag import SimpleVectorStore
from local_llm_kit.tools import ToolCall, parse_tool_calls
from tests.test_models import FakeResponse


class TestCallValidation(unittest.TestCase):
    def test_invalid_record_types_have_clear_errors(self):
        for cls in (FunctionCall, ToolCall):
            for record in (None, [], "bad", 1):
                with self.subTest(cls=cls, record=record), self.assertRaises(TypeError):
                    cls.from_dict(record)
        for payload in (None, [], "bad"):
            with self.subTest(payload=payload), self.assertRaises(TypeError):
                ToolCall.from_dict({"function": payload})

    def test_invalid_names_and_arguments_cannot_become_empty_calls(self):
        for cls in (FunctionCall, ToolCall):
            for name in (None, "", "   ", 1):
                with self.subTest(cls=cls, name=name), self.assertRaises(ValueError):
                    cls.from_dict({"name": name, "arguments": "{}"})
            for arguments in (None, [], 1, "null", "[]", "1", "{broken"):
                with self.subTest(cls=cls, arguments=arguments), self.assertRaises(ValueError):
                    cls.from_dict({"name": "f", "arguments": arguments})

    def test_valid_argument_forms_roundtrip(self):
        for cls in (FunctionCall, ToolCall):
            for value, expected in (({}, {}), ("", {}), ('{"x": 1}', {"x": 1})):
                with self.subTest(cls=cls, value=value):
                    call = cls.from_dict({"name": "f", "arguments": value})
                    self.assertEqual(call.arguments, expected)
                    self.assertEqual(cls.from_dict(call.to_dict()).arguments, expected)
        self.assertEqual(ToolCall.from_dict({"name": "f"}).arguments, {})
        with self.assertRaises(ValueError):
            FunctionCall.from_dict({"name": "f"})

    def test_parsers_reject_invalid_calls_in_every_envelope(self):
        for arguments in (None, [], 4):
            call = {"name": "f", "arguments": arguments}
            for text in (json.dumps(call), json.dumps({"function_call": call})):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    parse_function_calls(text)
            for text in (
                json.dumps({"tool_calls": [{"function": call}]}),
                "<tool_call>" + json.dumps(call) + "</tool_call>",
            ):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    parse_tool_calls(text)


class TestModelCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.object(models, "_hf_hub_cache_dir", return_value=str(self.root / "hf"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repo_filename_collision_does_not_return_another_model(self):
        other = self.root / "other" / "repo" / "model.gguf"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"OTHER MODEL")
        resolved, _ = models.resolve_model("owner/repo", filename="model.gguf", cache_dir=self.root)
        self.assertEqual(resolved, "owner/repo/model.gguf")

    def test_repo_revisions_cache_reuse_and_force_refresh(self):
        with (
            mock.patch.object(models, "_try_hf_hub_download", return_value=None),
            mock.patch(
                "urllib.request.urlopen", side_effect=lambda *a, **kw: FakeResponse(b"MODEL")
            ) as request,
        ):
            main = models.download_model("owner/repo", "m.gguf", self.root)
            self.assertEqual(models.download_model("owner/repo", "m.gguf", self.root), main)
            self.assertEqual(request.call_count, 1)
            branch = models.download_model("owner/repo", "m.gguf", self.root, "feature/tools")
            self.assertNotEqual(main, branch)
            self.assertIn("feature%2Ftools", branch)
            self.assertIn("feature%2Ftools", request.call_args.args[0].full_url)
            models.download_model("owner/repo", "m.gguf", self.root, force=True)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(request.call_args.kwargs["timeout"], 30)

    def test_hub_cache_resolves_requested_ref_or_commit(self):
        repo = self.root / "models--owner--repo"
        refs = repo / "refs"
        refs.mkdir(parents=True)
        for label, sha in (("main", "a" * 40), ("v1", "b" * 40)):
            (refs / label).write_text(sha)
            snapshot = repo / "snapshots" / sha
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "m.gguf").write_bytes(label.encode())
        for revision, sha in (("main", "a" * 40), ("v1", "b" * 40), ("b" * 40, "b" * 40)):
            path, _ = models.resolve_model(
                "owner/repo", filename="m.gguf", cache_dir=self.root, revision=revision
            )
            self.assertEqual(path, str(repo / "snapshots" / sha / "m.gguf"))
        path, _ = models.resolve_model("owner/repo", cache_dir=self.root, revision="v1")
        self.assertEqual(path, str(repo / "snapshots" / ("b" * 40)))

    def test_urls_with_the_same_filename_have_separate_caches(self):
        with mock.patch(
            "urllib.request.urlopen", side_effect=lambda *a, **kw: FakeResponse(b"OK")
        ) as request:
            first = "https://huggingface.co/owner/one/resolve/main/m.gguf"
            second = "https://huggingface.co/owner/two/resolve/main/m.gguf"
            one = models.download_model(first, cache_dir=self.root)
            two = models.download_model(second, cache_dir=self.root)
            self.assertNotEqual(one, two)
            self.assertEqual(models.download_model(first, cache_dir=self.root), one)
            self.assertEqual(request.call_count, 2)
            models.download_model(first, cache_dir=self.root, force=True)
            self.assertEqual(request.call_count, 3)

    def test_traversal_and_symlink_escapes_are_rejected(self):
        for repo in ("../outside", "owner/../outside"):
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                models._dest_for_repo_file(repo, "m.gguf", self.root)
        outside = self.root / "outside"
        outside.mkdir()
        cache = self.root / "cache"
        cache.mkdir()
        try:
            (cache / "owner").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks are unavailable")
        with self.assertRaises(ValueError):
            models._dest_for_repo_file("owner/repo", "m.gguf", cache)

    def test_concurrent_downloads_use_distinct_temporary_files(self):
        barrier = threading.Barrier(2)

        class Response(FakeResponse):
            def read(self, size=-1):
                if self.reads == 0:
                    barrier.wait(timeout=5)
                return super().read(size)

        destination = str(self.root / "m.gguf")
        replace = os.replace
        with (
            mock.patch(
                "urllib.request.urlopen", side_effect=lambda *a, **kw: Response(b"COMPLETE")
            ),
            mock.patch.object(models.os, "replace", wraps=replace) as renamed,
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(
                pool.map(
                    lambda _: models._download_url("https://huggingface.co/m", destination),
                    range(2),
                )
            )
        self.assertEqual(results, [destination, destination])
        self.assertEqual(len({call.args[0] for call in renamed.call_args_list}), 2)
        self.assertEqual(Path(destination).read_bytes(), b"COMPLETE")
        self.assertEqual(list(self.root.glob("*.part")), [])


class TestPublicHelpers(unittest.TestCase):
    def test_request_options_reach_methods_and_keep_positional_compatibility(self):
        module = importlib.import_module("local_llm_kit.chat")
        self.assertEqual(
            list(inspect.signature(module.chat).parameters).index("backend_instance"), 16
        )
        self.assertEqual(
            list(inspect.signature(module.complete).parameters).index("backend_instance"), 12
        )
        with mock.patch.object(module, "_cached_llm") as client:
            module.chat(
                [], "echo", execute_tools=False, max_iterations=2, tool_timeout=0.5, top_p=0.4
            )
            options = client.return_value.chat.call_args.kwargs
            self.assertFalse(options["execute_tools"])
            self.assertEqual(options["max_iterations"], 2)
            self.assertEqual(options["tool_timeout"], 0.5)
            self.assertEqual(options["top_p"], 0.4)
            self.assertEqual(client.call_args.kwargs["llm_kwargs"]["backend_kwargs"], {})
            module.complete("", "echo", top_k=5, repetition_penalty=1.2)
            self.assertEqual(client.return_value.complete.call_args.kwargs["top_k"], 5)

    def test_pull_flags_reach_the_download_function(self):
        args = build_parser().parse_args(["pull", "owner/repo", "--filename", "m.gguf", "--force"])
        self.assertEqual(args.filename, "m.gguf")
        self.assertTrue(args.force)

        def download(repo_or_path, filename=None, cache_dir=None, revision="main", force=False):
            return repo_or_path, filename, cache_dir, revision, force

        self.assertEqual(
            _adapt_download_call(download, "owner/repo", "/cache", "v1", True, "m.gguf"),
            ("owner/repo", "m.gguf", "/cache", "v1", True),
        )
        self.assertEqual(_parse_choice("required", "--tool-choice"), "required")

    def test_serve_forwards_backend_configuration(self):
        def factory(**kwargs):
            return kwargs

        args = build_parser().parse_args(["serve", "--backend", "echo"])
        self.assertEqual(
            _instantiate_app(factory, args, {"base_url": "http://localhost:8080"})[
                "backend_kwargs"
            ],
            {"base_url": "http://localhost:8080"},
        )

    def test_signature_adapters_do_not_hide_runtime_failures(self):
        calls = []

        def broken_factory(**kwargs):
            calls.append(kwargs)
            raise TypeError("factory failed")

        args = build_parser().parse_args(["serve", "--backend", "echo"])
        with self.assertRaisesRegex(TypeError, "factory failed"):
            _instantiate_app(broken_factory, args, {})
        self.assertEqual(len(calls), 1)

        def download(repo_or_path, /, filename=None, **kwargs):
            return repo_or_path, filename, kwargs

        self.assertEqual(
            _adapt_download_call(download, "owner/repo", None, None, False, "m.gguf"),
            ("owner/repo", "m.gguf", {}),
        )

    def test_invalid_requests_leave_history_usable(self):
        llm = LLM("echo", backend="echo", use_cache=False)
        for messages, error in ((None, TypeError), ([{}], ValueError), (["bad"], TypeError)):
            with self.subTest(messages=messages), self.assertRaises(error):
                llm.chat(messages)
            self.assertEqual(llm.memory.messages, [])
        self.assertIn("choices", llm.chat([]))
        self.assertIn("choices", llm.complete(""))
        with self.assertRaises(TypeError):
            llm.chat([], response_format=[])
        with self.assertRaises(TypeError):
            llm.embed(["ok", None])

    def test_tool_tokens_are_counted_and_invalid_history_is_atomic(self):
        history = MessageHistory()
        call = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "f", "arguments": "x" * 200}}],
        }
        self.assertGreater(history._count_message_tokens(call), 50)
        with self.assertRaises(TypeError):
            history.add_messages([{"role": "user", "content": "ok"}, {"function_call": "bad"}])
        self.assertEqual(history.messages, [])

    def test_embedding_dimensions_and_batches_require_integers(self):
        for invalid in (True, 1.5, "64"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                hash_embed("hi", dim=invalid)
            with self.assertRaises(TypeError):
                EmbeddingClient(backend="hash", batch_size=invalid)

    def test_corrupt_persisted_values_are_not_fabricated(self):
        base = {"ids": ["a"], "texts": ["hi"], "metadatas": [{}], "vectors": [[]]}
        for field, value in (("ids", [None]), ("texts", [None]), ("metadatas", [3]), ("dim", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "store.json"
                path.write_text(json.dumps(dict(base, **{field: value})))
                with self.assertRaises(ValueError):
                    SimpleVectorStore.load(str(path))

    def test_custom_recommended_table_handles_null_fields(self):
        self.assertIn(
            "| ? | - | (any/snapshot) | ? | ? |", models.format_recommended_models([{"name": None}])
        )

    def test_tool_history_survives_all_prompt_templates(self):
        messages = [
            {"role": "user", "content": "original question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [ToolCall("lookup", {}).to_dict()],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "lookup result"},
            {"role": "user", "content": "followup question"},
        ]
        for formatter in (
            ChatMLPromptFormatter,
            Llama2ChatPromptFormatter,
            MistralInstructPromptFormatter,
            VicunaPromptFormatter,
            PlainInstructPromptFormatter,
        ):
            with self.subTest(formatter=formatter):
                prompt = formatter().format_messages(messages)
                self.assertIn("lookup", prompt)
                self.assertIn("lookup result", prompt)
                self.assertEqual(prompt.count("followup question"), 1)


if __name__ == "__main__":
    unittest.main()
