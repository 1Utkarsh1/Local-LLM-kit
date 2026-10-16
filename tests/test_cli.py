"""Offline tests for the command line interface (local_llm_kit.cli).

Covers: ``--response-format`` handling of a plain JSON Schema vs a full
response_format object (inline and from a file), an end-to-end ``chat``
run on the offline echo backend, ``pull`` argument adaptation against
``models.download_model`` (network mocked), and the documented
``examples/simple_chat.py --backend echo`` invocation. Stdlib unittest
only; no model downloads, no network. Py3.9 compatible.

Run: python -m pytest tests/test_cli.py -q
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from local_llm_kit import models
from local_llm_kit.cli import _resolve_formats, build_parser, main

CITY_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["name", "population"],
}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_main(argv):
    """Run the CLI in-process, capturing stdout/stderr and the exit code."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = main(argv)
    return code, out.getvalue()


def _run_main_expecting_exit(argv):
    """Run the CLI and return ``(exit_code, output)`` for a ``_fail()`` path."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            main(argv)
    except SystemExit as exc:
        return exc.code, out.getvalue()
    raise AssertionError("CLI did not exit; output: %s" % out.getvalue()[-300:])


def _formats_for(response_format):
    args = build_parser().parse_args(
        ["chat", "-m", "echo", "--backend", "echo", "--response-format", response_format]
    )
    return _resolve_formats(args)


class TestResponseFormatResolution(unittest.TestCase):
    """A JSON Schema's own "type" key must not be mistaken for a format type."""

    def test_plain_json_schema_is_wrapped_as_json_schema(self):
        _legacy, modern = _formats_for(json.dumps(CITY_SCHEMA))
        self.assertEqual(modern["type"], "json_schema")
        self.assertEqual(modern["json_schema"]["schema"], CITY_SCHEMA)

    def test_schema_file_is_wrapped_as_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "schema.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(CITY_SCHEMA, fh)
            _legacy, modern = _formats_for(path)
        self.assertEqual(modern["type"], "json_schema")
        self.assertEqual(modern["json_schema"]["schema"], CITY_SCHEMA)

    def test_full_response_format_object_is_passed_through(self):
        _legacy, modern = _formats_for('{"type": "json_object"}')
        self.assertEqual(modern, {"type": "json_object"})
        _legacy, modern = _formats_for(
            json.dumps({"type": "json_schema", "json_schema": {"name": "c", "schema": CITY_SCHEMA}})
        )
        self.assertEqual(modern["json_schema"]["name"], "c")

    def test_shorthands(self):
        self.assertEqual(_formats_for("json")[1], {"type": "json_object"})
        self.assertEqual(_formats_for("text"), (None, None))


class TestChatCommand(unittest.TestCase):
    def test_chat_runs_offline_on_the_echo_backend(self):
        code, out = _run_main(["chat", "-m", "echo", "--backend", "echo", "-q", "hi there"])
        self.assertEqual(code, 0, msg=out[-500:])
        self.assertIn("hi there", out)

    def test_chat_with_a_json_schema_response_format_completes(self):
        code, out = _run_main(
            [
                "chat",
                "-m",
                "echo",
                "--backend",
                "echo",
                "-q",
                "name a city",
                "--response-format",
                json.dumps(CITY_SCHEMA),
            ]
        )
        self.assertEqual(code, 0, msg=out[-500:])
        self.assertNotIn("Unsupported response_format type", out)

    def test_version_flag(self):
        code, out = _run_main(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("local-llm-kit", out)


class TestPullCommand(unittest.TestCase):
    """`pull` must reach models.download_model with the documented arguments."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="llmk-cli-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _pull(self, argv):
        with mock.patch.object(models, "_try_hf_hub_download", return_value="/cache/m.gguf") as dl:
            code, out = _run_main(argv)
        return code, out, dl

    def test_repo_and_filename_shorthand_reaches_the_download_helper(self):
        code, out, dl = self._pull(
            ["pull", "owner/name:file.gguf", "--output-dir", self.tmp, "--revision", "v1"]
        )
        self.assertEqual(code, 0, msg=out[-500:])
        self.assertNotIn("unrecognised signature", out)
        self.assertIn("/cache/m.gguf", out)
        repo_id, filename, cache_dir, revision = dl.call_args[0][:4]
        self.assertEqual(repo_id, "owner/name")
        self.assertEqual(filename, "file.gguf")
        self.assertEqual(cache_dir, self.tmp)
        self.assertEqual(revision, "v1")

    def test_repo_slash_file_reference_also_works(self):
        code, out, dl = self._pull(["pull", "owner/name/file.gguf", "--output-dir", self.tmp])
        self.assertEqual(code, 0, msg=out[-500:])
        self.assertNotIn("unrecognised signature", out)
        repo_id, filename = dl.call_args[0][:2]
        self.assertEqual((repo_id, filename), ("owner/name", "file.gguf"))

    def test_repo_without_filename_reports_the_actionable_error(self):
        code, out = _run_main_expecting_exit(["pull", "owner/name", "--output-dir", self.tmp])
        self.assertEqual(code, 1)
        self.assertIn("requires a filename", out)
        self.assertNotIn("unrecognised signature", out)


class TestDocumentedExampleInvocation(unittest.TestCase):
    """README: `python examples/simple_chat.py --backend echo` works offline."""

    def test_simple_chat_example_runs_offline_as_documented(self):
        repo = _repo_root()
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, os.path.join(repo, "examples", "simple_chat.py"), "--backend", "echo"],
            input="hi\nquit\n",
            capture_output=True,
            text=True,
            cwd=repo,
            env=env,
            timeout=60,
        )
        output = proc.stdout + proc.stderr
        self.assertNotIn("invalid choice", output)
        self.assertNotIn("unrecognized arguments", output)
        self.assertEqual(proc.returncode, 0, msg=output[-500:])
        self.assertIn("Echo:", output, msg="example did not stream a reply: %s" % output[-300:])


if __name__ == "__main__":
    unittest.main()


class TestDoctorCommand(unittest.TestCase):
    """``doctor`` rendering and exit codes, with discovery stubbed (no network)."""

    @staticmethod
    def _report():
        from local_llm_kit.discovery import DiscoveryReport, ModelRecord, ServerReport

        server = ServerReport(
            base_url="http://127.0.0.1:11434/v1",
            provider="ollama",
            live=True,
            latency_ms=3.2,
            models=[
                ModelRecord(
                    id="qwen2.5:0.5b",
                    capabilities=["completion", "tools"],
                    context_length=32768,
                ),
                ModelRecord(
                    id="big:cloud",
                    capabilities=["completion"],
                    context_length=1000000,
                    remote=True,
                    remote_host="https://ollama.com:443",
                ),
            ],
            notes=["capabilities read from ollama /api/tags"],
        )
        dead = ServerReport(
            base_url="http://127.0.0.1:1234/v1",
            provider="lmstudio",
            live=False,
            error="Connection refused",
        )
        return DiscoveryReport(servers=[server, dead], elapsed_ms=4.0, probed=2)

    @staticmethod
    def _empty_report():
        from local_llm_kit.discovery import DiscoveryReport, ServerReport

        return DiscoveryReport(
            servers=[
                ServerReport(
                    base_url="http://127.0.0.1:1234/v1",
                    provider="lmstudio",
                    live=False,
                    error="Connection refused",
                )
            ],
            elapsed_ms=1.0,
            probed=1,
        )

    def _run_with(self, report, argv):
        with mock.patch("local_llm_kit.discovery.discover", return_value=report):
            return _run_main(argv)

    def test_live_server_exits_zero_and_names_the_on_device_model(self):
        code, output = self._run_with(self._report(), ["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("qwen2.5:0.5b", output)
        self.assertIn("ctx 32768", output)
        self.assertIn('LLM("qwen2.5:0.5b", backend="ollama")', output)

    def test_off_device_models_are_flagged_with_their_remote_host(self):
        _code, output = self._run_with(self._report(), ["doctor"])
        self.assertIn("OFF-DEVICE", output)
        self.assertIn("ollama.com", output)

    def test_nothing_running_exits_one(self):
        code, output = self._run_with(self._empty_report(), ["doctor"])
        self.assertEqual(code, 1)
        self.assertIn("No local LLM server responded", output)

    def test_require_local_exits_zero_when_a_local_model_exists(self):
        code, _output = self._run_with(self._report(), ["doctor", "--require-local"])
        self.assertEqual(code, 0)

    def test_require_local_exits_one_when_only_remote_models_exist(self):
        from local_llm_kit.discovery import DiscoveryReport, ModelRecord, ServerReport

        report = DiscoveryReport(
            servers=[
                ServerReport(
                    base_url="http://127.0.0.1:11434/v1",
                    provider="ollama",
                    live=True,
                    models=[ModelRecord(id="big:cloud", remote=True, remote_host="h")],
                )
            ],
            probed=1,
        )
        code, output = self._run_with(report, ["doctor", "--require-local"])
        self.assertEqual(code, 1)
        self.assertIn("No server offering an on-device model", output)

    def test_json_output_is_machine_readable_and_includes_failures(self):
        code, output = self._run_with(self._report(), ["doctor", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["live_count"], 1)
        self.assertEqual(payload["probed"], 2)
        self.assertEqual(len(payload["servers"]), 2)
        self.assertTrue(payload["servers"][0]["all_local"] is False)
        self.assertTrue(payload["servers"][1]["error"])

    def test_quiet_mode_prints_nothing_but_keeps_the_exit_code(self):
        code, output = self._run_with(self._report(), ["doctor", "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(output.strip(), "")

    def test_aliases_are_wired_up(self):
        for alias in ("discover", "probe"):
            with self.subTest(alias=alias):
                code, _output = self._run_with(self._report(), [alias])
                self.assertEqual(code, 0)

    def test_non_numeric_timeout_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["doctor", "--timeout", "soon"])
