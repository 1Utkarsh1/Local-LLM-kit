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
from local_llm_kit.prompt_templates import register_template

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


class TestLogsCommand(unittest.TestCase):
    """Phase 2: `local-llm-kit logs` browse/export subcommands."""

    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, self.db)

    def test_logs_list_no_data(self):
        code, out = _run_main(["logs", "list", "--db", self.db])
        self.assertEqual(code, 0, msg=out)
        self.assertIn("No conversations", out)

    def test_logs_list_after_chat(self):
        code, out = _run_main(
            [
                "chat",
                "-m",
                "echo",
                "--backend",
                "echo",
                "--conversation-log",
                self.db,
                "-q",
                "hi",
            ]
        )
        self.assertEqual(code, 0, msg=out[-500:])
        code, out = _run_main(["logs", "list", "--db", self.db, "--json"])
        self.assertEqual(code, 0, msg=out)
        data = json.loads(out)
        self.assertGreaterEqual(len(data), 1)

    def test_logs_show_json(self):
        _run_main(
            [
                "chat",
                "-m",
                "echo",
                "--backend",
                "echo",
                "--conversation-log",
                self.db,
                "-q",
                "hello",
            ]
        )
        code, out = _run_main(["logs", "list", "--db", self.db, "--json"])
        convs = json.loads(out)
        conv_id = convs[0]["id"]
        code, out = _run_main(["logs", "show", conv_id, "--db", self.db, "--json"])
        self.assertEqual(code, 0, msg=out)
        data = json.loads(out)
        self.assertIn("conversation", data)
        self.assertIn("history", data)

    def test_logs_show_not_found(self):
        code, out = _run_main_expecting_exit(["logs", "show", "nonexistent", "--db", self.db])
        self.assertEqual(code, 1)
        self.assertIn("Conversation not found", out)

    def test_logs_export_text(self):
        _run_main(
            [
                "chat",
                "-m",
                "echo",
                "--backend",
                "echo",
                "--conversation-log",
                self.db,
                "-q",
                "hello world",
            ]
        )
        code, out = _run_main(["logs", "list", "--db", self.db, "--json"])
        convs = json.loads(out)
        conv_id = convs[0]["id"]
        code, out = _run_main(["logs", "export", conv_id, "text", "--db", self.db])
        self.assertEqual(code, 0, msg=out)
        self.assertIn("Conversation", out)
        self.assertIn("hello world", out)

    def test_logs_export_json(self):
        _run_main(
            [
                "chat",
                "-m",
                "echo",
                "--backend",
                "echo",
                "--conversation-log",
                self.db,
                "-q",
                "hello",
            ]
        )
        code, out = _run_main(["logs", "list", "--db", self.db, "--json"])
        convs = json.loads(out)
        conv_id = convs[0]["id"]
        code, out = _run_main(["logs", "export", conv_id, "json", "--db", self.db])
        self.assertEqual(code, 0, msg=out)
        data = json.loads(out)
        self.assertIn("conversation", data)


class TestTemplateCommand(unittest.TestCase):
    """Phase 2: `local-llm-kit template` subcommands."""

    def test_template_list_empty(self):
        code, out = _run_main(["template", "list"])
        self.assertEqual(code, 0, msg=out)
        self.assertIn("No templates", out)

    def test_template_register_render_via_cli(self):
        # register in default registry, render
        register_template("cli_test_tmpl", "Hello {{who}}")
        code, out = _run_main(["template", "render", "cli_test_tmpl", "--var", "who=CLI"])
        self.assertEqual(code, 0, msg=out)
        self.assertEqual(out.strip(), "Hello CLI")

    def test_template_show(self):
        register_template("cli_show_tmpl", "Raw {{who}} content")
        code, out = _run_main(["template", "show", "cli_show_tmpl"])
        self.assertEqual(code, 0, msg=out)
        self.assertEqual(out.strip(), "Raw {{who}} content")

    def test_template_list_shows_registered(self):
        register_template("cli_list_tmpl", "List {{x}}")
        code, out = _run_main(["template", "list"])
        self.assertEqual(code, 0, msg=out)
        self.assertIn("cli_list_tmpl", out)

    def test_template_render_missing_var_fails(self):
        register_template("cli_missing_var", "Hello {{name}}")
        code, out = _run_main_expecting_exit(["template", "render", "cli_missing_var"])
        self.assertEqual(code, 1)
        self.assertIn("missing required variable", out)

    def test_template_render_with_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tmpl.json")
            with open(path, "w") as fh:
                json.dump({"template": "File {{name}}", "name": "filetmpl"}, fh)
            code, out = _run_main(
                [
                    "template",
                    "render",
                    "filetmpl",
                    "--var",
                    "name=Disk",
                    "--registry-file",
                    path,
                ]
            )
            self.assertEqual(code, 0, msg=out)
            self.assertEqual(out.strip(), "File Disk")

    def test_template_not_found(self):
        code, out = _run_main_expecting_exit(["template", "show", "does_not_exist"])
        self.assertEqual(code, 1)
        self.assertIn("template not found", out)


if __name__ == "__main__":
    unittest.main()
