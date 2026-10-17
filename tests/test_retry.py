"""Offline tests for the retry policy and its wiring into the HTTP backends.

The policy is exercised directly, and end-to-end against a local fixture that
counts requests and can be told to fail a given number of times first. No
network, no models.

The behaviour under test:
  * transient failures (refused connection, 429, 5xx) are retried,
  * client errors (4xx other than 429) are not,
  * ``max_retries=0`` restores single-attempt behaviour,
  * a stream is retried only before the first frame is emitted,
  * ``Retry-After`` is honoured and clamped.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_kit.backends.ollama import OllamaBackend
from local_llm_kit.backends.openai_compat import OpenAICompatBackend
from local_llm_kit.retry import (
    DEFAULT_RETRY,
    MAX_RETRY_AFTER,
    RETRYABLE_STATUS,
    RetryPolicy,
    coerce_policy,
    is_retryable_status,
    retry_after_seconds,
)


class _FakeHeaders:
    def __init__(self, value=None):
        self._value = value

    def get(self, name, default=None):
        if name == "Retry-After":
            return self._value
        return default


def dead_url():
    """A loopback port that was bound then released."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    return "http://127.0.0.1:%d" % port


class Handler(BaseHTTPRequestHandler):
    """Fixture that fails the first ``fail_times`` requests, then succeeds."""

    fail_times = 0
    fail_status = 503
    retry_after = None
    attempts = 0
    paths = []

    def log_message(self, *args):
        pass

    def _respond(self, payload, status=200, content_type="application/json"):
        encoded = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if self.retry_after is not None:
            self.send_header("Retry-After", str(self.retry_after))
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        Handler.attempts += 1
        Handler.paths.append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        if Handler.attempts <= Handler.fail_times:
            return self._respond({"error": "still loading"}, status=Handler.fail_status)

        # Ollama's native routes use a different envelope than /v1.
        if self.path == "/api/chat":
            if body.get("stream"):
                lines = [
                    {"message": {"content": "hello"}, "done": False},
                    {"message": {"content": ""}, "done": True, "done_reason": "stop"},
                ]
                return self._respond(
                    "".join(json.dumps(line) + "\n" for line in lines),
                    content_type="application/x-ndjson",
                )
            return self._respond(
                {"message": {"content": "hello"}, "done": True, "done_reason": "stop"}
            )

        if body.get("stream"):
            frames = [
                {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            return self._respond(
                "".join("data: %s\n\n" % json.dumps(f) for f in frames) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return self._respond(
            {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
        )

    def do_GET(self):
        Handler.attempts += 1
        Handler.paths.append(self.path)
        if Handler.attempts <= Handler.fail_times:
            return self._respond({"error": "still loading"}, status=Handler.fail_status)
        return self._respond({"object": "list", "data": [{"id": "fixture"}]})


class Fixture:
    def __init__(self, fail_times=0, fail_status=503, retry_after=None):
        Handler.fail_times = fail_times
        Handler.fail_status = fail_status
        Handler.retry_after = retry_after
        Handler.attempts = 0
        Handler.paths = []

    def __enter__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_port
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


# No sleeping in the test suite: delays are zeroed so only the counts matter.
def fast(max_retries):
    """A policy with the given retry count and no real delay."""
    return RetryPolicy(max_retries=max_retries, initial_delay=0.0, max_delay=0.0)


FAST = fast(2)


class TestRetryPolicyValues(unittest.TestCase):
    def test_defaults_match_the_official_openai_client(self):
        self.assertEqual(DEFAULT_RETRY.max_retries, 2)
        self.assertEqual(DEFAULT_RETRY.initial_delay, 0.5)
        self.assertEqual(DEFAULT_RETRY.max_delay, 8.0)

    def test_validation_rejects_nonsense(self):
        for kwargs in (
            {"max_retries": -1},
            {"max_retries": True},
            {"max_retries": 1.5},
            {"initial_delay": -0.1},
            {"max_delay": -1},
            {"initial_delay": 5.0, "max_delay": 1.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    RetryPolicy(**kwargs)

    def test_zero_retries_disables_the_policy(self):
        self.assertFalse(RetryPolicy(max_retries=0).enabled)
        self.assertTrue(RetryPolicy(max_retries=1).enabled)

    def test_backoff_doubles_and_is_capped(self):
        policy = RetryPolicy(max_retries=5, initial_delay=0.5, max_delay=2.0, jitter=False)
        self.assertEqual(policy.delay_for(1), 0.5)
        self.assertEqual(policy.delay_for(2), 1.0)
        self.assertEqual(policy.delay_for(3), 2.0)
        self.assertEqual(policy.delay_for(4), 2.0)

    def test_jitter_stays_within_half_to_full_backoff(self):
        policy = RetryPolicy(max_retries=5, initial_delay=1.0, max_delay=8.0, jitter=True)
        for attempt in (1, 2, 3):
            delay = policy.delay_for(attempt)
            nominal = min(1.0 * (2 ** (attempt - 1)), 8.0)
            self.assertGreaterEqual(delay, nominal / 2)
            self.assertLessEqual(delay, nominal)

    def test_retry_after_wins_over_computed_backoff(self):
        policy = RetryPolicy(max_retries=5, initial_delay=1.0, jitter=False)
        self.assertEqual(policy.delay_for(1, retry_after=3.0), 3.0)
        self.assertEqual(policy.delay_for(5, retry_after=0.25), 0.25)

    def test_retry_after_is_clamped(self):
        policy = RetryPolicy(max_retries=5, initial_delay=1.0, jitter=False)
        self.assertEqual(policy.delay_for(1, retry_after=999.0), MAX_RETRY_AFTER)
        self.assertLessEqual(policy.delay_for(1, retry_after=999.0), MAX_RETRY_AFTER)

    def test_negative_retry_after_falls_back_to_backoff(self):
        policy = RetryPolicy(max_retries=5, initial_delay=2.0, jitter=False)
        self.assertEqual(policy.delay_for(1, retry_after=-1.0), 2.0)

    def test_sleep_returns_the_delay_it_used(self):
        self.assertEqual(RetryPolicy(max_retries=1, initial_delay=0.0).sleep(1), 0.0)


class TestRetryableClassification(unittest.TestCase):
    def test_transient_statuses_are_retryable(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_retryable_status(status))
                self.assertIn(status, RETRYABLE_STATUS)

    def test_client_errors_are_not_retryable(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(is_retryable_status(status))

    def test_none_status_is_not_retryable(self):
        self.assertFalse(is_retryable_status(None))
        self.assertFalse(RetryPolicy().should_retry_status(None))


class TestRetryAfterParsing(unittest.TestCase):
    def test_delta_seconds_are_parsed(self):
        self.assertEqual(retry_after_seconds(_FakeHeaders("2")), 2.0)
        self.assertEqual(retry_after_seconds(_FakeHeaders("0.5")), 0.5)

    def test_missing_or_bogus_values_return_none(self):
        for value in (None, "", "  ", "Wed, 21 Oct 2026 07:28:00 GMT", "-3", "soon"):
            with self.subTest(value=value):
                self.assertIsNone(retry_after_seconds(_FakeHeaders(value)))

    def test_no_header_container_is_tolerated(self):
        self.assertIsNone(retry_after_seconds(object()))
        self.assertIsNone(retry_after_seconds(None))

    def test_large_values_are_clamped(self):
        self.assertEqual(retry_after_seconds(_FakeHeaders("100000")), MAX_RETRY_AFTER)


class TestCoercePolicy(unittest.TestCase):
    def test_none_yields_the_default(self):
        self.assertIs(coerce_policy(None), DEFAULT_RETRY)

    def test_int_becomes_a_policy(self):
        self.assertEqual(coerce_policy(3).max_retries, 3)
        self.assertEqual(coerce_policy(0).max_retries, 0)

    def test_policy_passes_through(self):
        policy = RetryPolicy(max_retries=7)
        self.assertIs(coerce_policy(policy), policy)

    def test_other_types_are_rejected(self):
        for bad in (True, "3", 1.5, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    coerce_policy(bad)


class TestOpenAICompatRetries(unittest.TestCase):
    def _backend(self, url, **kwargs):
        return OpenAICompatBackend("fixture", base_url=url + "/v1", timeout=5.0, **kwargs)

    def test_transient_5xx_is_retried_then_succeeds(self):
        with Fixture(fail_times=2, fail_status=503) as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            backend.retry = FAST
            result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["text"], "hello")
        self.assertEqual(Handler.attempts, 3)

    def test_429_is_retried(self):
        with Fixture(fail_times=1, fail_status=429) as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            backend.retry = FAST
            result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["text"], "hello")
        self.assertEqual(Handler.attempts, 2)

    def test_client_error_is_not_retried(self):
        with Fixture(fail_times=99, fail_status=400) as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            backend.retry = FAST
            with self.assertRaises(RuntimeError) as ctx:
                backend.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(Handler.attempts, 1)

    def test_retries_are_exhausted_then_the_status_is_reported(self):
        with Fixture(fail_times=99, fail_status=503) as fixture:
            backend = self._backend(fixture.url, max_retries=1)
            backend.retry = fast(1)
            with self.assertRaises(RuntimeError) as ctx:
                backend.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 503", str(ctx.exception))
        self.assertEqual(Handler.attempts, 2)

    def test_more_retries_means_more_attempts(self):
        for retries in (0, 1, 3):
            with self.subTest(max_retries=retries):
                with Fixture(fail_times=99, fail_status=503) as fixture:
                    backend = self._backend(fixture.url, max_retries=retries)
                    backend.retry = fast(retries)
                    with self.assertRaises(RuntimeError):
                        backend.chat([{"role": "user", "content": "hi"}])
                self.assertEqual(Handler.attempts, retries + 1)

    def test_zero_retries_makes_exactly_one_attempt(self):
        with Fixture(fail_times=99, fail_status=503) as fixture:
            backend = self._backend(fixture.url, max_retries=0)
            with self.assertRaises(RuntimeError):
                backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(Handler.attempts, 1)

    def test_refused_connection_is_retried_then_reported(self):
        backend = self._backend(dead_url(), max_retries=2)
        backend.retry = FAST
        with self.assertRaises(ConnectionError) as ctx:
            backend.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not reachable", str(ctx.exception))

    def test_connection_retries_can_be_disabled(self):
        backend = self._backend(dead_url(), max_retries=0)
        with self.assertRaises(ConnectionError):
            backend.chat([{"role": "user", "content": "hi"}])

    def test_model_listing_is_retried_too(self):
        with Fixture(fail_times=1, fail_status=503) as fixture:
            backend = self._backend(fixture.url, max_retries=1)
            backend.retry = FAST
            models = backend.list_models()
        self.assertEqual(models[0]["id"], "fixture")
        self.assertEqual(Handler.attempts, 2)

    def test_streaming_retries_before_the_first_frame(self):
        with Fixture(fail_times=2, fail_status=503) as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            backend.retry = FAST
            chunks = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(c.get("text", "") for c in chunks), "hello")
        self.assertEqual(Handler.attempts, 3)

    def test_streaming_does_not_replay_after_frames_arrive(self):
        with Fixture() as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            backend.retry = FAST
            chunks = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(c.get("text", "") for c in chunks), "hello")
        self.assertEqual(Handler.attempts, 1)

    def test_retry_after_header_is_consumed(self):
        with Fixture(fail_times=1, fail_status=429, retry_after=0) as fixture:
            backend = self._backend(fixture.url, max_retries=2)
            result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["text"], "hello")
        self.assertEqual(Handler.attempts, 2)

    def test_retry_policy_is_configurable_by_int_or_object(self):
        self.assertEqual(self._backend("http://127.0.0.1:1").retry, DEFAULT_RETRY)
        self.assertEqual(self._backend("http://127.0.0.1:1", max_retries=5).retry.max_retries, 5)
        custom = RetryPolicy(max_retries=9)
        self.assertIs(self._backend("http://127.0.0.1:1", max_retries=custom).retry, custom)
        with self.assertRaises(TypeError):
            self._backend("http://127.0.0.1:1", max_retries="lots")


class TestOllamaRetries(unittest.TestCase):
    def test_transient_5xx_is_retried(self):
        with Fixture(fail_times=2, fail_status=503) as fixture:
            backend = OllamaBackend("fixture", host=fixture.url, timeout=5.0, max_retries=2)
            backend.retry = FAST
            result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["text"], "hello")
        self.assertEqual(Handler.attempts, 3)

    def test_client_error_is_not_retried(self):
        with Fixture(fail_times=99, fail_status=400) as fixture:
            backend = OllamaBackend("fixture", host=fixture.url, timeout=5.0, max_retries=2)
            backend.retry = FAST
            with self.assertRaises(RuntimeError) as ctx:
                backend.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(Handler.attempts, 1)

    def test_zero_retries_makes_exactly_one_attempt(self):
        with Fixture(fail_times=99, fail_status=503) as fixture:
            backend = OllamaBackend("fixture", host=fixture.url, timeout=5.0, max_retries=0)
            with self.assertRaises(RuntimeError):
                backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(Handler.attempts, 1)

    def test_connection_failure_message_still_points_at_ollama_serve(self):
        backend = OllamaBackend("fixture", host=dead_url(), timeout=0.4, max_retries=0)
        with self.assertRaises(ConnectionError) as ctx:
            backend.chat([{"role": "user", "content": "hi"}])
        self.assertIn("ollama serve", str(ctx.exception))

    def test_streaming_retries_before_the_first_frame(self):
        with Fixture(fail_times=1, fail_status=503) as fixture:
            backend = OllamaBackend("fixture", host=fixture.url, timeout=5.0, max_retries=1)
            backend.retry = FAST
            chunks = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(c.get("text", "") for c in chunks), "hello")
        self.assertEqual(Handler.attempts, 2)

    def test_capability_detection_does_not_retry(self):
        # Detection is best-effort and must stay fast, so it is single-attempt
        # even when the backend itself retries.
        backend = OllamaBackend("fixture", host=dead_url(), timeout=0.3, max_retries=5)
        result = backend.detect_capabilities(timeout=0.3)
        self.assertFalse(result["detected"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
