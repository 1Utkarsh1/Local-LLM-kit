"""Regression tests for audit fixes (offline, stdlib only).

Covers the concrete bugs fixed during the audit:
- serializers validate null/partial records with ValueError/TypeError
  (never AttributeError), while preserving happy paths;
- models.py traversal/download guards + CLI pull mapping;
- LLM/CLI/server entry validation.

Run: python -m pytest tests/test_audit_fixes.py -q
"""

import json
import os
import tempfile
import unittest
from unittest import mock


class TestToolCallStrict(unittest.TestCase):
    def test_from_dict_rejects_non_dict(self):
        from local_llm_kit.tools import ToolCall

        for bad in (None, [], "x", 123):
            with self.assertRaises(ValueError, msg="input %r" % (bad,)):
                ToolCall.from_dict(bad)  # type: ignore[arg-type]

    def test_from_dict_rejects_null_function(self):
        from local_llm_kit.tools import ToolCall

        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": None})
        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": "bad"})

    def test_from_dict_rejects_null_or_typed_args(self):
        from local_llm_kit.tools import ToolCall

        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": {"name": "f", "arguments": None}})
        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": {"name": "f", "arguments": 123}})
        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": {"arguments": {}}})

    def test_from_dict_happy_paths_preserved(self):
        from local_llm_kit.tools import ToolCall

        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": '{"a": 1}'}})
        self.assertEqual(tc.arguments, {"a": 1})
        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": ""}})
        self.assertEqual(tc.arguments, {})
        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": {"a": 1}}})
        self.assertEqual(tc.arguments, {"a": 1})

    def test_init_rejects_non_dict_args(self):
        from local_llm_kit.tools import ToolCall

        with self.assertRaises(TypeError):
            ToolCall(name="f", arguments=None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ToolCall(name="", arguments={})


class TestFunctionCallStrict(unittest.TestCase):
    def test_from_dict_rejects_partial(self):
        from local_llm_kit.function_calling import FunctionCall

        with self.assertRaises(ValueError):
            FunctionCall.from_dict(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FunctionCall.from_dict({})
        with self.assertRaises(ValueError):
            FunctionCall.from_dict({"name": "f", "arguments": None})
        with self.assertRaises(ValueError):
            FunctionCall.from_dict({"name": "f", "arguments": 123})

    def test_from_dict_happy(self):
        from local_llm_kit.function_calling import FunctionCall

        fc = FunctionCall.from_dict({"name": "f", "arguments": {"a": 1}})
        self.assertEqual(fc.arguments, {"a": 1})
        fc = FunctionCall.from_dict({"name": "f", "arguments": '{"a": 1}'})
        self.assertEqual(fc.arguments, {"a": 1})

    def test_execute_rejects_non_dict(self):
        from local_llm_kit.function_calling import FunctionRegistry

        reg = FunctionRegistry()
        reg.add_function("f", {"name": "f"}, lambda **kw: kw)
        with self.assertRaises(TypeError):
            reg.execute("f", None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            reg.execute("missing", {})


class TestMemoryStrict(unittest.TestCase):
    def test_add_message_rejects_non_dict(self):
        from local_llm_kit.memory import MessageHistory

        m = MessageHistory()
        with self.assertRaises(TypeError):
            m.add_message(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            m.add_messages("bad")  # type: ignore[arg-type]

    def test_count_rejects_bad_function_call(self):
        from local_llm_kit.memory import MessageHistory

        m = MessageHistory()
        with self.assertRaises(TypeError):
            m._count_message_tokens({"role": "a", "content": "hi", "function_call": "bad"})


class TestRagLoadStrict(unittest.TestCase):
    def _tmp(self, data):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "s.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return p

    def test_null_vectors_rejected(self):
        from local_llm_kit.rag import SimpleVectorStore

        p = self._tmp({"ids": ["a"], "texts": ["hi"], "metadatas": [{}], "vectors": [None]})
        with self.assertRaises(ValueError):
            SimpleVectorStore.load(p)

    def test_none_ids_rejected_not_corrupted(self):
        from local_llm_kit.rag import SimpleVectorStore

        p = self._tmp(
            {
                "ids": ["a", None],
                "texts": ["hi", "ho"],
                "metadatas": [{}, {}],
                "vectors": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        with self.assertRaises(ValueError):
            SimpleVectorStore.load(p)

    def test_missing_keys_rejected(self):
        from local_llm_kit.rag import SimpleVectorStore

        p = self._tmp({})
        with self.assertRaises(ValueError):
            SimpleVectorStore.load(p)

    def test_roundtrip_still_works(self):
        from local_llm_kit.rag import SimpleVectorStore

        s = SimpleVectorStore()
        s.add(["hello world"], metadatas=[{"source": "d"}])
        d = tempfile.mkdtemp()
        p = os.path.join(d, "s.json")
        s.save(p)
        loaded = SimpleVectorStore.load(p)
        self.assertEqual(loaded.texts, ["hello world"])


class TestModelsHardening(unittest.TestCase):
    def test_no_serializer_in_models(self):
        from local_llm_kit import models

        self.assertFalse(hasattr(models, "from_dict"))
        self.assertFalse(hasattr(models, "to_dict"))
        self.assertTrue(callable(models.download_model))

    def test_cli_pull_mapping_includes_repo_or_path(self):
        from local_llm_kit import models
        from local_llm_kit.cli import _MODEL_ARG_KEYS, _adapt_download_call

        self.assertIn("repo_or_path", _MODEL_ARG_KEYS)

        def _fake(repo_or_path, filename=None, cache_dir=None, revision="main"):
            return (repo_or_path, filename, cache_dir, revision)

        self.assertEqual(
            _adapt_download_call(_fake, "my-model", "/tmp/out", "v1", True),
            ("my-model", None, "/tmp/out", "v1"),
        )
        # Real helper maps without "unrecognised signature" (may raise for
        # network/missing file, but must not exit with that message).
        import inspect

        self.assertIn("repo_or_path", inspect.signature(models.download_model).parameters)

    def test_dest_rejects_traversal(self):
        from local_llm_kit import models

        with self.assertRaises(ValueError):
            models._dest_for_repo_file("../../etc", "passwd", "/tmp/cache-test")
        # Basename-only sanitization: "../evil" safely becomes "evil" inside cache.
        dest = models._dest_for_repo_file("owner/name", "../evil", "/tmp/cache-test")
        self.assertTrue(dest.startswith("/tmp/cache-test" + os.sep))
        self.assertTrue(dest.endswith(os.path.join("owner", "name", "evil")))

    def test_find_cached_rejects_traversal(self):
        from local_llm_kit import models

        self.assertIsNone(models._find_cached_repo_file("../../etc", "passwd", "/tmp/nope"))

    def test_download_url_validates(self):
        from local_llm_kit import models

        with self.assertRaises(ValueError):
            models._download_url("https://example.com/x", "/tmp/x", chunk_size=0)
        with self.assertRaises(ValueError):
            models._download_url("https://example.com/x", "/tmp", chunk_size=1024)

    def test_download_url_atomic_cleanup_on_failure(self):
        from local_llm_kit import models

        d = tempfile.mkdtemp()
        dest = os.path.join(d, "f.gguf")
        part = dest + ".part"
        with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("net down")):
            with self.assertRaises(RuntimeError):
                models._download_url("https://huggingface.co/o/n/resolve/main/f.gguf", dest)
        self.assertFalse(os.path.exists(part))
        self.assertFalse(os.path.exists(dest))

    def test_suggested_backend_rejects_non_string(self):
        from local_llm_kit import models

        with self.assertRaises(ValueError):
            models.suggested_backend_for(None)  # type: ignore[arg-type]

    def test_split_shorthand_strips_query(self):
        from local_llm_kit import models

        repo, fname = models._split_shorthand("owner/name/model.gguf?download=true")
        self.assertEqual(repo, "owner/name")
        self.assertEqual(fname, "model.gguf")

    def test_resolve_rejects_bad_types(self):
        from local_llm_kit import models

        with self.assertRaises(ValueError):
            models.resolve_model("owner/name", filename=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            models.list_cached_models(cache_dir=123)  # type: ignore[arg-type]


class TestLLMValidation(unittest.TestCase):
    def test_model_path_rejected(self):
        from local_llm_kit import LLM

        with self.assertRaises(ValueError):
            LLM(model_path=None, backend="echo")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LLM(model_path="  ", backend="echo")

    def test_backend_aliases_work(self):
        from local_llm_kit import LLM

        llm = LLM(model_path="echo", backend="mock", use_cache=False)
        out = llm.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("Echo", out["choices"][0]["message"]["content"])

    def test_chat_rejects_malformed(self):
        from local_llm_kit import LLM

        llm = LLM(model_path="echo", backend="echo", use_cache=False)
        with self.assertRaises(ValueError):
            llm.chat(messages=[{}])
        with self.assertRaises(ValueError):
            llm.chat(messages=None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            llm.chat(messages=[{"role": "user", "content": "hi"}], tools=123)  # type: ignore[arg-type]

    def test_complete_rejects_non_string(self):
        from local_llm_kit import LLM

        llm = LLM(model_path="echo", backend="echo", use_cache=False)
        with self.assertRaises(TypeError):
            llm.complete(prompt=None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            llm.complete(prompt="   ")

    def test_response_format_rejects_wrong_type(self):
        from local_llm_kit import LLM

        llm = LLM(model_path="echo", backend="echo", use_cache=False)
        with self.assertRaises(TypeError):
            llm.chat(messages=[{"role": "user", "content": "hi"}], response_format=[])  # type: ignore[arg-type]

    def test_embed_rejects_bad_input(self):
        from local_llm_kit import LLM

        llm = LLM(model_path="echo", backend="echo", use_cache=False)
        with self.assertRaises(TypeError):
            llm.embed(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            llm.embed(["ok", None])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            llm.embed([])


class TestPromptAndServer(unittest.TestCase):
    def test_message_text_none_is_empty(self):
        from local_llm_kit.prompt_formatting import has_images, message_text

        self.assertEqual(message_text({"role": "user", "content": None}), "")
        self.assertEqual(
            message_text({"role": "user", "content": [{"type": "text", "text": None}]}), ""
        )
        self.assertFalse(has_images("bad"))  # type: ignore[arg-type]
        self.assertFalse(has_images(None))  # type: ignore[arg-type]

    def test_coerce_does_not_emit_none(self):
        from local_llm_kit.server import _coerce_message_content

        out = _coerce_message_content([{"type": "text", "text": None}])
        self.assertNotIn("None", out)
        out = _coerce_message_content(
            [{"type": "image_url", "image_url": {"url": "http://x/y.jpg"}}]
        )
        self.assertIn("[image:", out)

    def test_tool_choice_required_accepted(self):
        from local_llm_kit.cli import _parse_choice

        self.assertEqual(_parse_choice("required", "--tool-choice"), "required")

    def test_native_embeddings_forwarded(self):
        from local_llm_kit.server import _try_get_embeddings

        class FakeEmb:
            def embed(self, texts):
                return {"data": [{"embedding": [0.1, 0.2]}]}

        self.assertEqual(_try_get_embeddings(FakeEmb(), ["hi"]), [[0.1, 0.2]])

    def test_embeddings_dim_type_checked(self):
        from local_llm_kit.embeddings import EmbeddingClient, hash_embed

        with self.assertRaises(TypeError):
            hash_embed(["hi"], dim="64")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            EmbeddingClient(backend="hash", batch_size="32")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
