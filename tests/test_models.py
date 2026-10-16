"""Offline tests for local_llm_kit.models (download, resolution, and caching helpers).

Stdlib unittest and mocks only; no heavy downloads, no external network calls.
Python 3.9 compatible.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from local_llm_kit.models import (
    RECOMMENDED_MODELS,
    _download_url,
    download_gguf,
    download_model,
    format_recommended_models,
    is_gguf_path,
    is_hf_repo_id,
    is_url,
    list_recommended_models,
    resolve_model,
    suggested_backend_for,
)


class TestModelPredicates(unittest.TestCase):
    def test_is_url(self):
        self.assertTrue(is_url("https://huggingface.co/model.gguf"))
        self.assertTrue(is_url("http://localhost:8000/model.gguf"))
        self.assertFalse(is_url("org/repo"))
        self.assertFalse(is_url("/local/path/to/model.gguf"))
        self.assertFalse(is_url(None))  # type: ignore

    def test_is_gguf_path(self):
        self.assertTrue(is_gguf_path("model.gguf"))
        self.assertTrue(is_gguf_path("model.GGUF"))
        self.assertTrue(is_gguf_path("https://example.com/model.gguf?download=true"))
        self.assertFalse(is_gguf_path("model.bin"))
        self.assertFalse(is_gguf_path(None))  # type: ignore

    def test_is_hf_repo_id(self):
        self.assertTrue(is_hf_repo_id("bartowski/Llama-3.2-1B-Instruct-GGUF"))
        self.assertTrue(is_hf_repo_id("Qwen/Qwen2.5-0.5B-Instruct"))
        self.assertFalse(is_hf_repo_id("not-a-repo"))
        self.assertFalse(is_hf_repo_id("org/repo/file.gguf"))
        self.assertFalse(is_hf_repo_id(None))  # type: ignore
        # Null byte in path must not crash with unhandled ValueError/OSError
        self.assertFalse(is_hf_repo_id("org/repo\x00file"))

    def test_suggested_backend_for(self):
        self.assertEqual(suggested_backend_for("model.gguf"), "llamacpp")
        self.assertEqual(suggested_backend_for("org/repo", filename="model.gguf"), "llamacpp")
        self.assertEqual(suggested_backend_for("gpt2"), "transformers")


class TestRecommendedModels(unittest.TestCase):
    def test_list_recommended_models_returns_copy(self):
        recs = list_recommended_models()
        self.assertEqual(len(recs), len(RECOMMENDED_MODELS))
        recs[0]["name"] = "tampered"
        self.assertNotEqual(RECOMMENDED_MODELS[0]["name"], "tampered")

    def test_format_recommended_models_default(self):
        formatted = format_recommended_models()
        self.assertIn("| Model | Repo | File | Backend | Size |", formatted)
        self.assertIn("Qwen2.5 0.5B Instruct", formatted)

    def test_format_recommended_models_null_fields(self):
        custom = [
            {
                "name": None,
                "repo_id": None,
                "filename": None,
                "backend": None,
                "approx_size": None,
            }
        ]
        formatted = format_recommended_models(custom)
        self.assertNotIn("None", formatted)
        self.assertIn("| ? | - | (any/snapshot) | ? | ? |", formatted)


class TestModelDownloads(unittest.TestCase):
    def test_download_model_replaces_tmp_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "model.gguf")

            class FakeResponse:
                def __init__(self):
                    self.data = b"GGUF_TEST_BYTES"
                    self.read_called = False

                def getheader(self, name):
                    if name.lower() == "content-length":
                        return str(len(self.data))
                    return None

                def read(self, chunk_size=1024):
                    if not self.read_called:
                        self.read_called = True
                        return self.data
                    return b""

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            with patch("urllib.request.urlopen", return_value=FakeResponse()):
                path = _download_url(
                    "https://huggingface.co/org/repo/resolve/main/model.gguf", dest
                )
                self.assertEqual(path, dest)
                self.assertTrue(os.path.isfile(dest))
                with open(dest, "rb") as fh:
                    self.assertEqual(fh.read(), b"GGUF_TEST_BYTES")
                # No temp file left behind
                self.assertEqual(len(os.listdir(tmp)), 1)

    def test_download_url_detects_truncated_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "model.gguf")

            class TruncatedResponse:
                def __init__(self):
                    self.read_called = False

                def getheader(self, name):
                    if name.lower() == "content-length":
                        return "100"  # Promised 100 bytes
                    return None

                def read(self, chunk_size=1024):
                    if not self.read_called:
                        self.read_called = True
                        return b"partial_data"  # Only gave 12 bytes
                    return b""

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            with patch("urllib.request.urlopen", return_value=TruncatedResponse()):
                with self.assertRaises(RuntimeError) as ctx:
                    _download_url("https://huggingface.co/org/repo/resolve/main/model.gguf", dest)
                self.assertIn("prematurely", str(ctx.exception))
                self.assertFalse(os.path.exists(dest))

    def test_download_model_cache_reuse_when_force_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = os.path.join(tmp, "org", "repo", "model.gguf")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as fh:
                fh.write("CACHED_CONTENT")

            # With force=False, it finds cached file and does not download
            path = download_model("org/repo", filename="model.gguf", cache_dir=tmp, force=False)
            self.assertEqual(path, os.path.abspath(cache_file))

    def test_download_gguf_validates_suffix(self):
        with self.assertRaises(ValueError):
            download_gguf("org/repo", "model.bin")


class TestResolveModel(unittest.TestCase):
    def test_resolve_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_file = os.path.join(tmp, "my_model.gguf")
            with open(local_file, "w") as fh:
                fh.write("data")
            path, backend = resolve_model(local_file)
            self.assertEqual(path, os.path.abspath(local_file))
            self.assertEqual(backend, "llamacpp")

    def test_resolve_shorthand_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_file = os.path.join(tmp, "org", "repo", "weights.gguf")
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            with open(cached_file, "w") as fh:
                fh.write("gguf")

            path, backend = resolve_model("org/repo:weights.gguf", cache_dir=tmp)
            self.assertEqual(path, os.path.abspath(cached_file))
            self.assertEqual(backend, "llamacpp")


if __name__ == "__main__":
    unittest.main()
