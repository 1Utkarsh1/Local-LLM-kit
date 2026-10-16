"""Offline tests for model download/discovery helpers (local_llm_kit.models).

Focus: an interrupted or failed download must never leave a file behind
that ``list_cached_models()`` / ``resolve_model()`` would later report as a
usable cached model. ``urllib.request.urlopen`` is mocked, so the suite
never touches the network. Stdlib unittest only; Py3.9 compatible.

Run: python -m pytest tests/test_models.py -q
"""

import hashlib
import os
import shutil
import tempfile
import unittest
import urllib.request
from unittest import mock

from local_llm_kit import models

URL = "https://huggingface.co/owner/name/resolve/main/model-q4_k_m.gguf"
URL_KEY = hashlib.sha256(URL.encode("utf-8")).hexdigest()


def _all_files(root):
    """Every file under *root* (absolute paths, sorted)."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            found.append(os.path.join(dirpath, name))
    return sorted(found)


class FakeResponse(object):
    """Minimal ``urlopen`` replacement: one chunk, then EOF or an error."""

    def __init__(self, payload, fail_after=None, exc_factory=None, content_length=None):
        self.payload = payload
        self.fail_after = fail_after  # successful reads before raising
        self.exc_factory = exc_factory or (lambda: OSError("connection reset"))
        self.content_length = len(payload) if content_length is None else content_length
        self.reads = 0

    def getheader(self, name, default=None):
        if name == "Content-Length":
            return str(self.content_length)
        return default

    def read(self, size=-1):
        if self.fail_after is not None and self.reads >= self.fail_after:
            raise self.exc_factory()
        self.reads += 1
        return self.payload if self.reads == 1 else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestDownloadLeavesNoPartialFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="llmk-models-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _download(self, response):
        with mock.patch.object(urllib.request, "urlopen", lambda *a, **kw: response):
            return models.download_model(URL, cache_dir=self.tmp, force=True)

    def test_successful_download_is_complete_and_leaves_no_part_file(self):
        payload = b"GGUF" * 500
        path = self._download(FakeResponse(payload))
        self.assertTrue(os.path.isfile(path), msg="downloaded model missing")
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), payload)
        self.assertEqual([p for p in _all_files(self.tmp) if p.endswith(".part")], [])

    def test_failed_download_leaves_nothing_and_raises_runtime_error(self):
        response = FakeResponse(b"x" * 4096, fail_after=1)
        with self.assertRaises(RuntimeError):
            self._download(response)
        self.assertEqual(_all_files(self.tmp), [], msg="failed download left files behind")

    def test_interrupted_download_leaves_nothing_behind(self):
        response = FakeResponse(
            b"x" * 4096,
            fail_after=1,
            exc_factory=lambda: KeyboardInterrupt("simulated Ctrl-C"),
        )
        with self.assertRaises(KeyboardInterrupt):
            self._download(response)
        self.assertEqual(_all_files(self.tmp), [], msg="interrupted download left files behind")

    def test_oversized_download_is_refused_without_leaving_files(self):
        response = FakeResponse(b"x" * 16, content_length=6 * 1024 * 1024 * 1024)
        with self.assertRaises(RuntimeError) as ctx:
            self._download(response)
        self.assertIn("exceeds", str(ctx.exception))
        self.assertEqual(_all_files(self.tmp), [])

    def test_truncated_download_is_rejected_and_cleaned_up(self):
        """A server that promises more bytes than it sends must not yield a file."""
        response = FakeResponse(b"x" * 4096, content_length=100000)
        with self.assertRaises(RuntimeError) as ctx:
            self._download(response)
        self.assertIn("Truncated download", str(ctx.exception))
        self.assertEqual(_all_files(self.tmp), [])

    def test_directory_destination_is_rejected(self):
        """Reject directory destinations before opening a download."""
        dest = os.path.join(self.tmp, "urls", URL_KEY, "model-q4_k_m.gguf")
        os.makedirs(dest)  # make the destination path a directory
        with self.assertRaises(ValueError):
            self._download(FakeResponse(b"x" * 16))
        self.assertEqual(_all_files(self.tmp), [])

    def test_failed_redownload_keeps_an_existing_model_file(self):
        """A previously downloaded model must survive a later failed attempt."""
        dest = os.path.join(self.tmp, "urls", URL_KEY, "model-q4_k_m.gguf")
        os.makedirs(os.path.dirname(dest))
        with open(dest, "wb") as fh:
            fh.write(b"COMPLETE")
        with self.assertRaises(RuntimeError):
            self._download(FakeResponse(b"x" * 16, fail_after=1))
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), b"COMPLETE")

    def test_interrupted_download_does_not_poison_the_cache(self):
        response = FakeResponse(
            b"x" * 4096,
            fail_after=1,
            exc_factory=lambda: KeyboardInterrupt("simulated Ctrl-C"),
        )
        with self.assertRaises(KeyboardInterrupt):
            self._download(response)

        # Isolate the lookup from the real user HF cache dir.
        hf_cache = os.path.join(self.tmp, "hf-home")
        with mock.patch.object(models, "_hf_hub_cache_dir", return_value=hf_cache):
            resolved, backend = models.resolve_model(
                "owner/name", filename="model-q4_k_m.gguf", cache_dir=self.tmp
            )
            cached = models.list_cached_models(
                cache_dir=self.tmp, search_cwd=False, search_hf_cache=False
            )

        self.assertEqual(resolved, "owner/name/model-q4_k_m.gguf")
        self.assertEqual(backend, "llamacpp")
        self.assertFalse(os.path.isfile(resolved), msg="resolve_model returned a phantom file")
        self.assertEqual(cached, [])


if __name__ == "__main__":
    unittest.main()
