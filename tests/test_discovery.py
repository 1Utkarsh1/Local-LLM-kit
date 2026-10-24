"""Offline tests for local server discovery and capability reporting.

Everything here runs against a local stdlib HTTP fixture or a closed port, so
the suite stays offline and deterministic. The fixture reproduces the response
shapes of Ollama, llama.cpp's llama-server, LM Studio and apfel as their own
documentation/source describes them.

Real-server checks live in ``tests/test_live_servers.py`` and are opt-in.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_kit import LLM, discover, find_local_server, probe_endpoint
from local_llm_kit.backends import OPENAI_SERVER_URLS
from local_llm_kit.backends.ollama import OllamaBackend
from local_llm_kit.discovery import (
    DEFAULT_TIMEOUT,
    ENDPOINTS,
    DiscoveryReport,
    ModelRecord,
    ServerReport,
    normalize_base_url,
)

# Profile names the fixture understands.
OLLAMA = "ollama"
LLAMA_SERVER = "llama-server"
LMSTUDIO = "lmstudio"
APFEL = "apfel"
NOT_JSON = "not-json"
BROKEN = "broken"
NO_MODELS = "no-models"


class Handler(BaseHTTPRequestHandler):
    """Protocol fixture: serves the documented shape for ``profile``."""

    profile = OLLAMA
    requests = []

    def log_message(self, *args):
        pass

    # -- responses -----------------------------------------------------

    def _json(self, payload, status=200):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _text(self, body, status=200):
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        Handler.requests.append(("GET", self.path))
        path = self.path.split("?")[0]
        profile = Handler.profile

        if path == "/":
            if profile == NOT_JSON:
                return self._text("Ollama is running")
            return self._json({"status": "ok"})

        if path in ("/api/v1/models", "/api/v0/models"):
            if profile == LMSTUDIO and path == "/api/v1/models":
                return self._json(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "google/gemma-3-4b",
                                "capabilities": {
                                    "vision": True,
                                    "trained_for_tool_use": True,
                                },
                                "max_context_length": 131072,
                                "format": "gguf",
                            }
                        ]
                    }
                )
            return self._json({"error": "not found"}, status=404)

        if path.endswith("/models"):
            if profile == BROKEN:
                return self._json({"error": "boom"}, status=500)
            if profile == NOT_JSON:
                # A web server or proxy squatting on the port: HTTP 200, not JSON.
                return self._text("<!doctype html><title>AirPlay</title>")
            return self._json(self._models_payload())

        if path == "/api/tags":
            if profile == OLLAMA:
                return self._json(self._ollama_tags())
            return self._json({"error": "not found"}, status=404)

        if path == "/props":
            if profile == LLAMA_SERVER:
                return self._json(
                    {
                        "model_path": "/models/tiny.gguf",
                        "total_slots": 1,
                        "default_generation_settings": {"n_ctx": 8192},
                        "modalities": {"vision": False},
                    }
                )
            return self._json({"error": "not found"}, status=404)

        return self._json({"error": "not found"}, status=404)

    def do_POST(self):
        Handler.requests.append(("POST", self.path))
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/show":
            if Handler.profile != OLLAMA:
                return self._json({"error": "not found"}, status=404)
            if body.get("model") == "broken:latest":
                return self._json({"error": "server exploded"}, status=500)
            return self._json(
                {
                    "capabilities": ["completion", "tools"],
                    "model_info": {"qwen2.context_length": 32768},
                }
            )
        return self._json({"error": "not found"}, status=404)

    # -- payloads ------------------------------------------------------

    def _models_payload(self):
        profile = Handler.profile
        if profile == OLLAMA:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "qwen2.5:0.5b",
                        "object": "model",
                        "owned_by": "library",
                        "details": {"family": "qwen2", "context_length": 32768},
                    }
                ],
            }
        if profile == LLAMA_SERVER:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "/models/tiny.gguf",
                        "object": "model",
                        "owned_by": "llamacpp",
                        "meta": {"n_ctx_train": 131072, "n_params": 8030261312},
                    }
                ],
            }
        if profile == LMSTUDIO:
            return {
                "object": "list",
                "data": [
                    {"id": "google/gemma-3-4b", "object": "model", "owned_by": "organization_owner"}
                ],
            }
        if profile == APFEL:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "apple-foundationmodel",
                        "object": "model",
                        "owned_by": "apple",
                        "context_window": 4096,
                        "capabilities": ["tool_calling", "guided_generation"],
                    }
                ],
            }
        if profile == NO_MODELS:
            # text-generation-webui answers 200 with an empty list until a
            # model is loaded; a probe must not call that "ready".
            return {"object": "list", "data": []}
        return {"object": "list", "data": [{"id": "custom-model", "object": "model"}]}

    def _ollama_tags(self):
        return {
            "models": [
                {
                    "name": "qwen2.5:0.5b",
                    "model": "qwen2.5:0.5b",
                    "details": {"family": "qwen2", "context_length": 32768},
                    "capabilities": ["completion", "tools"],
                },
                {
                    "name": "big:cloud",
                    "model": "big:cloud",
                    "remote_model": "big",
                    "remote_host": "https://ollama.com:443",
                    "details": {"family": "big"},
                    "capabilities": ["completion", "tools", "thinking"],
                },
            ]
        }


class Fixture:
    """Run the fixture server with one profile at a time."""

    def __init__(self):
        self.server = None
        self.thread = None

    def __enter__(self):
        Handler.requests.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_port
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def profile(name):
    """Context manager selecting the fixture's response profile."""

    class _Profile:
        def __enter__(self):
            Handler.profile = name
            return self

        def __exit__(self, *exc):
            Handler.profile = OLLAMA

    return _Profile()


def dead_url():
    """A loopback port that is bound then released, so nothing is listening."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    server.server_close()
    return "http://127.0.0.1:%d/v1" % port


class TestNormalizeBaseUrl(unittest.TestCase):
    def test_accepts_what_users_actually_paste(self):
        cases = {
            "localhost:11434": "http://localhost:11434/v1",
            "127.0.0.1:1234": "http://127.0.0.1:1234/v1",
            "http://localhost:8000": "http://localhost:8000/v1",
            "http://localhost:8000/v1": "http://localhost:8000/v1",
            "http://localhost:8000/v1/": "http://localhost:8000/v1",
            "http://localhost:8000/v1/chat/completions": "http://localhost:8000/v1",
            "http://localhost:8000/v1/models": "http://localhost:8000/v1",
            "https://box.local:8443/v1": "https://box.local:8443/v1",
            "http://h:8080/v1/models?x=1": "http://h:8080/v1",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_base_url(raw), expected)

    def test_rejects_empty_and_hostless_input(self):
        for bad in ("", "   ", "http://"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    normalize_base_url(bad)
        with self.assertRaises(ValueError):
            normalize_base_url(None)


class TestProviderFingerprinting(unittest.TestCase):
    def test_each_profile_is_identified_from_its_own_payload(self):
        cases = [
            (OLLAMA, "ollama"),
            (LLAMA_SERVER, "llama-server"),
            (LMSTUDIO, "lmstudio"),
            (APFEL, "apfel"),
        ]
        with Fixture() as fixture:
            for name, expected in cases:
                with self.subTest(profile=name), profile(name):
                    report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
                    self.assertTrue(report.live, report.error)
                    self.assertEqual(report.provider, expected)

    def test_a_non_json_200_is_not_mistaken_for_a_model_listing(self):
        # Ports in the probe range are often taken by other software (macOS
        # AirPlay answers 200 on :5000), so a 200 alone must not mean "server".
        with Fixture() as fixture, profile(NOT_JSON):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertFalse(report.live)
        self.assertEqual(report.models, [])
        self.assertIn("invalid JSON", report.error or "")


class TestCapabilityEnrichment(unittest.TestCase):
    def test_ollama_context_length_and_capabilities_come_from_the_server(self):
        with Fixture() as fixture, profile(OLLAMA):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        model = next(m for m in report.models if m.id == "qwen2.5:0.5b")
        self.assertEqual(model.context_length, 32768)
        self.assertEqual(model.capabilities, ["completion", "tools"])
        self.assertTrue(model.supports_tools)
        self.assertFalse(model.supports_vision)

    def test_llama_server_context_comes_from_props(self):
        with Fixture() as fixture, profile(LLAMA_SERVER):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertEqual(report.context_window, 8192)
        self.assertIs(report.vision, False)
        self.assertTrue(any("tiny.gguf" in note for note in report.notes))

    def test_lmstudio_native_api_supplies_vision_and_tool_capabilities(self):
        with Fixture() as fixture, profile(LMSTUDIO):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        model = report.models[0]
        self.assertIn("vision", model.capabilities)
        self.assertIn("tools", model.capabilities)
        self.assertEqual(model.context_length, 131072)

    def test_apfel_capabilities_come_from_its_own_model_payload(self):
        with Fixture() as fixture, profile(APFEL):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertEqual(report.context_window, None)
        self.assertEqual(report.models[0].context_length, 4096)
        self.assertEqual(report.models[0].capabilities, ["tool_calling", "guided_generation"])

    def test_enrichment_failure_degrades_to_a_note_not_an_exception(self):
        with Fixture() as fixture, profile(BROKEN):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertFalse(report.live)
        self.assertIsNotNone(report.error)
        self.assertIsInstance(report.notes, list)


class TestOffDeviceDetection(unittest.TestCase):
    def test_remote_host_marks_a_model_as_off_device(self):
        with Fixture() as fixture, profile(OLLAMA):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        remote = [m for m in report.models if m.remote]
        self.assertEqual([m.id for m in remote], ["big:cloud"])
        self.assertEqual(remote[0].remote_host, "https://ollama.com:443")
        self.assertTrue(any("off-device" in note for note in report.notes))

    def test_a_mixed_server_still_offers_on_device_models(self):
        with Fixture() as fixture, profile(OLLAMA):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertTrue(report.has_local_models)
        self.assertFalse(report.all_local)
        self.assertEqual([m.id for m in report.local_models], ["qwen2.5:0.5b"])

    def test_cloud_tag_without_remote_host_is_still_off_device(self):
        record = ModelRecord(id="x:cloud", capabilities=[])
        self.assertFalse(record.remote)
        from local_llm_kit.discovery import _looks_remote

        self.assertTrue(_looks_remote("x:cloud", {})[0])
        self.assertFalse(_looks_remote("x:latest", {})[0])


class TestReportSemantics(unittest.TestCase):
    def _report(self):
        local_only = ServerReport(
            base_url="http://127.0.0.1:10001/v1",
            provider="ollama",
            live=True,
            models=[ModelRecord(id="small", local_verified=True)],
        )
        mixed = ServerReport(
            base_url="http://127.0.0.1:10002/v1",
            provider="ollama",
            live=True,
            models=[
                ModelRecord(id="small", local_verified=True),
                ModelRecord(id="big:cloud", remote=True, remote_host="h"),
            ],
        )
        remote_only = ServerReport(
            base_url="http://127.0.0.1:10003/v1",
            provider="ollama",
            live=True,
            models=[ModelRecord(id="big:cloud", remote=True, remote_host="h")],
        )
        return DiscoveryReport(servers=[remote_only, mixed, local_only], probed=3)

    def test_all_local_requires_models_and_no_remote_ones(self):
        report = self._report()
        self.assertEqual([s.base_url for s in report.local_only()], ["http://127.0.0.1:10001/v1"])
        self.assertEqual(
            [s.base_url for s in report.on_device()],
            ["http://127.0.0.1:10002/v1", "http://127.0.0.1:10001/v1"],
        )

    def test_require_local_skips_remote_only_servers(self):
        report = self._report()
        self.assertEqual(
            report.first_live(require_local=True).base_url, "http://127.0.0.1:10002/v1"
        )
        self.assertEqual(report.first_live().base_url, "http://127.0.0.1:10002/v1")

    def test_report_is_json_serialisable(self):
        payload = json.dumps(self._report().to_dict())
        self.assertIn("servers", json.loads(payload))


class TestDiscover(unittest.TestCase):
    def test_probes_extra_urls_only_when_asked(self):
        with Fixture() as fixture, profile(OLLAMA):
            report = discover(include_known=False, extra_urls=[fixture.url], timeout=2.0)
        self.assertEqual(report.probed, 1)
        self.assertTrue(report.servers[0].live)

    def test_duplicate_endpoints_are_probed_once(self):
        # Ollama (11434) and apfel (11434) share a port, as do llama-server,
        # LocalAI, mlx-lm and llamafile (8080) - each must be probed once.
        report = discover(include_known=True, extra_urls=[], timeout=0.05)
        urls = [s.base_url for s in report.servers]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertEqual(report.probed, len(set(normalize_base_url(e.base_url) for e in ENDPOINTS)))

    def test_unreachable_endpoints_are_reported_not_raised(self):
        report = discover(include_known=False, extra_urls=[dead_url()], timeout=0.5)
        self.assertEqual(len(report.offline), 1)
        self.assertTrue(report.offline[0].error)
        self.assertIsNone(report.offline[0].status)

    def test_find_local_server_returns_none_when_nothing_answers(self):
        self.assertIsNone(
            find_local_server(include_known=False, extra_urls=[dead_url()], timeout=0.3)
        )

    def test_find_local_server_returns_the_on_device_one(self):
        with Fixture() as fixture, profile(OLLAMA):
            found = find_local_server(include_known=False, extra_urls=[fixture.url], timeout=2.0)
        self.assertIsNotNone(found)
        self.assertEqual(found.provider, "ollama")
        self.assertTrue(found.has_local_models)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            discover(timeout=0)
        with self.assertRaises(ValueError):
            discover(max_workers=0)
        with self.assertRaises(ValueError):
            probe_endpoint("http://127.0.0.1:1/v1", timeout=-1)
        with self.assertRaises(ValueError):
            discover(include_known=False, extra_urls=[""])

    def test_default_timeout_is_short_enough_for_interactive_use(self):
        self.assertLessEqual(DEFAULT_TIMEOUT, 1.0)


class TestOllamaTruthfulCapabilities(unittest.TestCase):
    def test_list_models_exposes_capabilities_context_and_remote(self):
        with Fixture() as fixture, profile(OLLAMA):
            models = OllamaBackend("qwen2.5:0.5b", host=fixture.url).list_models()
        by_id = {m["id"]: m for m in models}
        self.assertEqual(by_id["qwen2.5:0.5b"]["capabilities"], ["completion", "tools"])
        self.assertEqual(by_id["qwen2.5:0.5b"]["context_length"], 32768)
        self.assertNotIn("remote", by_id["qwen2.5:0.5b"])
        self.assertTrue(by_id["big:cloud"]["remote"])
        self.assertEqual(by_id["big:cloud"]["remote_host"], "https://ollama.com:443")

    def test_detect_capabilities_reports_the_real_context_window(self):
        with Fixture() as fixture, profile(OLLAMA):
            backend = OllamaBackend("qwen2.5:0.5b", host=fixture.url)
            before = backend.get_context_window()
            result = backend.detect_capabilities()
        self.assertEqual(before, 4096)
        self.assertTrue(result["detected"])
        self.assertEqual(result["context_window"], 32768)
        self.assertEqual(backend.get_context_window(), 32768)
        self.assertFalse(backend.remote)

    def test_supports_vision_follows_the_server_not_the_adapter(self):
        with Fixture() as fixture, profile(OLLAMA):
            backend = OllamaBackend("qwen2.5:0.5b", host=fixture.url)
            self.assertTrue(backend.supports_vision)  # optimistic before detection
            backend.detect_capabilities()
            self.assertFalse(backend.supports_vision)  # server says text only

    def test_detection_failure_keeps_the_fallback_and_reports_the_error(self):
        backend = OllamaBackend("qwen2.5:0.5b", host=dead_url(), timeout=0.4)
        result = backend.detect_capabilities(timeout=0.4)
        self.assertFalse(result["detected"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(backend.get_context_window(), 4096)

    def test_auto_detect_makes_the_llm_report_the_real_window(self):
        with Fixture() as fixture, profile(OLLAMA):
            llm = LLM(
                "qwen2.5:0.5b",
                backend="ollama",
                use_cache=False,
                backend_kwargs={"host": fixture.url, "auto_detect_capabilities": True},
            )
        self.assertEqual(llm.context_window, 32768)
        self.assertEqual(llm.memory.context_window, 32768)

    def test_model_info_reports_capabilities_once_detected(self):
        with Fixture() as fixture, profile(OLLAMA):
            backend = OllamaBackend("qwen2.5:0.5b", host=fixture.url)
            backend.detect_capabilities()
            info = backend.model_info()
        self.assertEqual(info["capabilities"], ["completion", "tools"])
        self.assertFalse(info["supports_vision"])


class TestRequireLocal(unittest.TestCase):
    def test_native_transformers_and_llamacpp_are_accepted(self):
        from local_llm_kit.backends.llamacpp import LlamaCppBackend
        from local_llm_kit.backends.transformers import TransformersBackend

        for cls in (TransformersBackend, LlamaCppBackend):
            with self.subTest(backend=cls.__name__):
                backend = object.__new__(cls)
                backend.model_path = "local-checkpoint"
                llm = LLM("local-checkpoint", backend_instance=backend, require_local=True)
                self.assertTrue(llm.assert_local())

    def test_on_device_model_is_accepted(self):
        with Fixture() as fixture, profile(OLLAMA):
            llm = LLM(
                "qwen2.5:0.5b",
                backend="ollama",
                use_cache=False,
                require_local=True,
                backend_kwargs={"host": fixture.url},
            )
            self.assertTrue(llm.assert_local())
            self.assertEqual(llm.context_window, 32768)

    def test_off_device_model_is_refused_with_the_remote_host(self):
        with Fixture() as fixture, profile(OLLAMA):
            with self.assertRaises(RuntimeError) as ctx:
                LLM(
                    "big:cloud",
                    backend="ollama",
                    use_cache=False,
                    require_local=True,
                    backend_kwargs={"host": fixture.url},
                )
        self.assertIn("ollama.com", str(ctx.exception))

    def test_unverifiable_guarantee_is_refused_not_assumed(self):
        with self.assertRaises(RuntimeError) as ctx:
            LLM(
                "qwen2.5:0.5b",
                backend="ollama",
                use_cache=False,
                require_local=True,
                backend_kwargs={"host": dead_url(), "timeout": 0.4},
            )
        self.assertIn("Could not confirm", str(ctx.exception))

    def test_backends_without_a_remote_concept_are_a_no_op(self):
        llm = LLM("echo", backend="echo", use_cache=False, require_local=True)
        self.assertTrue(llm.assert_local())
        self.assertFalse(llm.detect_backend_capabilities()["supported"])


class TestLlamaServerPresetPort(unittest.TestCase):
    def test_preset_matches_the_current_upstream_default(self):
        # llama.cpp moved its default port from 8080 to 9931 (PR #30159,
        # merged 2026-10-09). The Docker image still binds 8080.
        self.assertEqual(OPENAI_SERVER_URLS["llama-server"], "http://localhost:9931/v1")

    def test_both_historical_ports_are_probed(self):
        ports = {normalize_base_url(e.base_url) for e in ENDPOINTS}
        self.assertIn("http://127.0.0.1:9931/v1", ports)
        self.assertIn("http://127.0.0.1:8080/v1", ports)


if __name__ == "__main__":
    unittest.main()


class TestEndpointCatalog(unittest.TestCase):
    def test_docker_model_runner_keeps_its_engines_prefix(self):
        # OpenAI compatibility lives under /engines, not at the root
        # (Docker model-runner API reference), so the prefix must survive
        # normalisation rather than being rewritten to /v1.
        normalized = normalize_base_url("http://127.0.0.1:12434/engines/v1")
        self.assertEqual(normalized, "http://127.0.0.1:12434/engines/v1")
        self.assertIn(normalized, {normalize_base_url(e.base_url) for e in ENDPOINTS})

    def test_fm_serve_port_is_probed(self):
        ports = {normalize_base_url(e.base_url) for e in ENDPOINTS}
        self.assertIn("http://127.0.0.1:1976/v1", ports)

    def test_shared_ports_do_not_break_identification(self):
        # 11434 is both Ollama's and apfel's default; 8080 is llama.cpp's old
        # default, LocalAI's, MLX-LM's and llamafile's.
        by_url = {}
        for endpoint in ENDPOINTS:
            by_url.setdefault(normalize_base_url(endpoint.base_url), []).append(endpoint.provider)
        self.assertGreater(len(by_url["http://127.0.0.1:11434/v1"]), 1)
        self.assertGreater(len(by_url["http://127.0.0.1:8080/v1"]), 1)


class TestNotReadyServers(unittest.TestCase):
    def test_an_empty_model_list_is_live_but_not_usable(self):
        with Fixture() as fixture, profile(NO_MODELS):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertTrue(report.live)
        self.assertEqual(report.models, [])
        self.assertFalse(report.has_local_models)
        self.assertFalse(report.all_local)
        self.assertTrue(any("lists no models" in note for note in report.notes))

    def test_a_server_that_is_still_loading_reports_the_status(self):
        with Fixture() as fixture, profile(BROKEN):
            report = probe_endpoint(fixture.url + "/v1", timeout=2.0)
        self.assertFalse(report.live)
        self.assertEqual(report.status, 500)
        self.assertTrue(any("still be loading" in note for note in report.notes))

    def test_loading_servers_are_kept_out_of_the_on_device_set(self):
        with Fixture() as fixture, profile(NO_MODELS):
            report = discover(include_known=False, extra_urls=[fixture.url], timeout=2.0)
        self.assertEqual(len(report.live), 1)
        self.assertEqual(report.on_device(), [])
        self.assertIsNone(report.first_live(require_local=True))
