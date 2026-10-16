"""Offline tests for local_llm_kit.cli helpers."""

import argparse
import unittest

from local_llm_kit.cli import _adapt_download_call, _instantiate_app, build_parser


class TestCLIFunctionality(unittest.TestCase):
    def test_pull_parser_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "pull",
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "--filename",
                "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "--force",
            ]
        )
        self.assertEqual(args.model_pos, "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        self.assertEqual(args.filename, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        self.assertTrue(args.force)

    def test_adapt_download_call_with_repo_or_path_and_filename(self):
        seen = {}

        def fake_download(
            repo_or_path, filename=None, cache_dir=None, revision="main", force=False
        ):
            seen["repo_or_path"] = repo_or_path
            seen["filename"] = filename
            seen["cache_dir"] = cache_dir
            seen["revision"] = revision
            seen["force"] = force
            return "/path/to/downloaded"

        result = _adapt_download_call(
            fake_download,
            model="org/repo",
            output_dir="/custom/cache",
            revision="v1.0",
            force=True,
            filename="weights.gguf",
        )
        self.assertEqual(result, "/path/to/downloaded")
        self.assertEqual(seen["repo_or_path"], "org/repo")
        self.assertEqual(seen["filename"], "weights.gguf")
        self.assertEqual(seen["cache_dir"], "/custom/cache")
        self.assertEqual(seen["revision"], "v1.0")
        self.assertTrue(seen["force"])

    def test_instantiate_app_forwards_backend_kwargs(self):
        seen = {}

        def fake_create_app(**kwargs):
            seen.update(kwargs)
            return "app_instance"

        args = argparse.Namespace(model="test-model", backend="echo")
        app = _instantiate_app(
            fake_create_app,
            args=args,
            backend_kwargs={"n_ctx": 2048},
        )
        self.assertEqual(app, "app_instance")
        self.assertEqual(seen["backend_kwargs"], {"n_ctx": 2048})


if __name__ == "__main__":
    unittest.main()
