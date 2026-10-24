"""Opt-in integration tests against a REAL local LLM server.

These are the tests that certify the discovery layer and the Ollama capability
reporting against an actual server rather than a fixture. They are skipped
unless ``LOCAL_LLM_KIT_LIVE=1`` is set, so the default suite stays offline.

    LOCAL_LLM_KIT_LIVE=1 python -m pytest tests/test_live_servers.py -q

What they require:
  * A local OpenAI-compatible or Ollama server listening on one of the
    well-known ports (``local-llm-kit doctor`` prints the catalog).
  * For the inference tests, at least one model that runs **on this machine**.
    Provider-routed models (Ollama ``:cloud``) are deliberately never used:
    they leave the device, so they cannot certify local inference.
  * No credentials and no network access beyond loopback.

Tests skip individually when the machine cannot satisfy them, and report why.
"""

import json
import os
import unittest

from local_llm_kit import LLM, discover
from local_llm_kit.discovery import find_local_server, normalize_base_url, probe_endpoint

LIVE = os.environ.get("LOCAL_LLM_KIT_LIVE", "").strip() == "1"
SKIP_REASON = "set LOCAL_LLM_KIT_LIVE=1 (with a local server running) to run"

_report_cache = {}


def live_report():
    """Discover once per test session and reuse."""
    if "report" not in _report_cache:
        _report_cache["report"] = discover(timeout=1.5)
    return _report_cache["report"]


def on_device_server():
    """The first live server offering an on-device model, or None."""
    for server in live_report().on_device():
        return server
    return None


def on_device_model(server):
    """The first model on *server* that is not provider-routed."""
    for model in server.local_models:
        return model
    return None


@unittest.skipUnless(LIVE, SKIP_REASON)
class TestDiscoveryAgainstRealServer(unittest.TestCase):
    def test_a_local_server_is_found_within_the_interactive_budget(self):
        report = live_report()
        self.assertTrue(report.live, "no local server found; run `local-llm-kit doctor`")
        self.assertLess(report.elapsed_ms, 5000, "discovery must stay interactive")
        print(
            "\n[live] probed %d endpoints in %sms; live: %s"
            % (report.probed, report.elapsed_ms, [s.base_url for s in report.live])
        )

    def test_provider_is_identified_from_live_payloads(self):
        for server in live_report().live:
            with self.subTest(base_url=server.base_url):
                self.assertNotEqual(server.provider, "unknown")
                self.assertNotEqual(server.provider, "custom")
                print("\n[live] %s -> %s" % (server.base_url, server.provider))

    def test_reported_capabilities_match_a_direct_probe(self):
        for server in live_report().live:
            with self.subTest(base_url=server.base_url):
                fresh = probe_endpoint(server.base_url, timeout=1.5)
                self.assertEqual(
                    sorted(m.id for m in fresh.models), sorted(m.id for m in server.models)
                )

    def test_off_device_models_are_detected_when_present(self):
        report = live_report()
        remote = [m for s in report.live for m in s.remote_models]
        for model in remote:
            with self.subTest(model=model.id):
                self.assertTrue(model.remote_host, "an off-device model needs a remote host")
                print("\n[live] off-device: %s -> %s" % (model.id, model.remote_host))

    def test_find_local_server_prefers_on_device_capable_servers(self):
        found = find_local_server(timeout=1.5)
        if found is None:
            self.skipTest("no server with an on-device model is running")
        self.assertTrue(found.has_local_models)


@unittest.skipUnless(LIVE, SKIP_REASON)
class TestOllamaCapabilitiesAgainstRealServer(unittest.TestCase):
    """Certify the capability reporting against a real Ollama ``/api/show``."""

    def setUp(self):
        server = on_device_server()
        if server is None or server.provider != "ollama":
            self.skipTest("no Ollama server with an on-device model is running")
        self.server = server
        self.model = on_device_model(server)

    def test_show_reports_a_context_length_the_adapter_now_uses(self):
        backend = LLM(
            self.model.id,
            backend="ollama",
            use_cache=False,
            backend_kwargs={"host": self.server.base_url.rsplit("/v1", 1)[0]},
        ).backend
        raw = backend.show()
        reported = backend.detect_capabilities()
        self.assertTrue(reported["detected"], reported)
        server_context = None
        for key, value in (raw.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                server_context = value
        self.assertIsNotNone(server_context, "server did not publish a context length")
        self.assertEqual(backend.get_context_window(), server_context)
        print(
            "\n[live] %s: adapter now reports ctx=%d (was hardcoded 4096); capabilities=%s"
            % (self.model.id, backend.get_context_window(), reported["capabilities"])
        )

    def test_local_model_is_never_flagged_as_remote(self):
        self.assertFalse(self.model.remote)
        self.assertIsNone(self.model.remote_host)

    def test_cloud_models_are_flagged_when_the_server_lists_any(self):
        remote = self.server.remote_models
        if not remote:
            self.skipTest("this server lists no provider-routed models")
        self.assertTrue(all(m.remote_host for m in remote))


@unittest.skipUnless(LIVE, SKIP_REASON)
class TestRealInferenceOnDevice(unittest.TestCase):
    """Prove the end-to-end path works against a genuinely local model."""

    def setUp(self):
        server = on_device_server()
        if server is None:
            self.skipTest("no server with an on-device model is running")
        if server.provider != "ollama":
            self.skipTest("inference check currently targets the Ollama backend")
        model = on_device_model(server)
        if model is None:
            self.skipTest("no on-device model available")
        self.server = server
        self.model = model
        self.llm = LLM(
            model.id,
            backend="ollama",
            use_cache=False,
            backend_kwargs={"host": server.base_url.rsplit("/v1", 1)[0]},
        )

    def test_chat_round_trip(self):
        response = self.llm.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=16,
            think=False,  # Keep the short smoke-test budget for visible output.
        )
        content = response["choices"][0]["message"]["content"]
        self.assertTrue(content.strip(), "model returned empty content")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertGreater(response["usage"]["completion_tokens"], 0)
        print("\n[live] %s chat -> %r" % (self.model.id, content.strip()[:60]))

    def test_streaming_round_trip(self):
        pieces = []
        for chunk in self.llm.chat(
            [{"role": "user", "content": "Count to three."}],
            max_tokens=24,
            stream=True,
            think=False,
        ):
            pieces.append(chunk["choices"][0]["delta"].get("content") or "")
        self.assertTrue("".join(pieces).strip())
        print("\n[live] %s streamed %d chunks" % (self.model.id, len(pieces)))

    def test_context_window_is_the_server_value(self):
        raw = self.llm.backend.show()
        expected = None
        for key, value in (raw.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                expected = value
        self.assertIsNotNone(expected)
        self.assertEqual(self.llm.detect_backend_capabilities()["context_window"], expected)

    def test_require_local_accepts_this_model(self):
        self.assertTrue(self.llm.assert_local())


@unittest.skipUnless(LIVE, SKIP_REASON)
class TestRequireLocalAgainstRealServer(unittest.TestCase):
    def test_a_remote_model_is_refused_on_a_real_server(self):
        report = live_report()
        remote = [(s, m) for s in report.live for m in s.remote_models]
        if not remote:
            self.skipTest("no provider-routed model is available to refuse")
        server, model = remote[0]
        with self.assertRaises(RuntimeError) as ctx:
            LLM(
                model.id,
                backend="ollama",
                use_cache=False,
                require_local=True,
                backend_kwargs={"host": server.base_url.rsplit("/v1", 1)[0]},
            )
        self.assertIn("not on this machine", str(ctx.exception))
        print("\n[live] refused %r as required: %s" % (model.id, str(ctx.exception)[:80]))


class TestNormalizeAgainstRealUrls(unittest.TestCase):
    """Runs always: pure string handling, no server needed."""

    def test_loopback_forms_normalise_consistently(self):
        for raw in ("localhost:11434", "http://localhost:11434", "http://localhost:11434/v1"):
            self.assertEqual(normalize_base_url(raw), "http://localhost:11434/v1")

    def test_live_report_serialises(self):
        if not LIVE:
            self.skipTest(SKIP_REASON)
        json.dumps(live_report().to_dict())


if __name__ == "__main__":
    unittest.main()
