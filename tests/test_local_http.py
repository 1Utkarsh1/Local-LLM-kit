"""Exercise real stdlib HTTP/SSE transport against a local protocol fixture."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_kit import LLM
from local_llm_kit.backends import OPENAI_SERVER_URLS


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, body, self.headers.get("Authorization")))
        if self.path.endswith("/chat/completions"):
            if body.get("stream"):
                events = [
                    {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
                self.respond(
                    "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                    + "data: [DONE]\n\n",
                    "text/event-stream",
                )
            else:
                self.respond(
                    {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
                )
        elif self.path == "/api/chat":
            result = {"message": {"content": "hello"}, "done": True, "done_reason": "stop"}
            self.respond(json.dumps(result) + "\n" if body.get("stream") else result)
        elif self.path.endswith("/embeddings") or self.path == "/api/embed":
            values = body["input"]
            values = [values] if isinstance(values, str) else values
            vectors = [[float(index), 2.0] for index in range(len(values))]
            if self.path == "/api/embed":
                self.respond({"embeddings": vectors})
            else:
                self.respond(
                    {
                        "data": [
                            {"index": index, "embedding": vector}
                            for index, vector in enumerate(vectors)
                        ]
                    }
                )
        else:
            self.send_error(404)

    def respond(self, body, content_type="application/json"):
        encoded = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class TestLocalHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        Handler.requests.clear()

    def test_all_presets_complete_stream_and_embed_over_http(self):
        for provider in OPENAI_SERVER_URLS:
            with self.subTest(provider=provider):
                llm = LLM(
                    "fixture-model",
                    backend=provider,
                    use_cache=False,
                    backend_kwargs={"base_url": self.url + "/v1", "api_key": "fixture-key"},
                )
                messages = [{"role": "user", "content": "Hello"}]
                response = llm.chat(messages)
                self.assertEqual(response["choices"][0]["message"]["content"], "hello")
                chunks = list(llm.chat(messages, stream=True))
                self.assertEqual(
                    "".join(item["choices"][0]["delta"].get("content") or "" for item in chunks),
                    "hello",
                )
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
                if llm.backend.supports_embeddings:
                    self.assertEqual(llm.embed(["one", "two"])["data"][1]["embedding"], [1.0, 2.0])
                self.assertEqual(Handler.requests[-1][2], "Bearer fixture-key")

    def test_native_ollama_uses_its_routes_over_http(self):
        llm = LLM(
            "fixture-model", backend="ollama", use_cache=False, backend_kwargs={"host": self.url}
        )
        messages = [{"role": "user", "content": "Hello"}]
        self.assertEqual(llm.chat(messages)["choices"][0]["message"]["content"], "hello")
        self.assertEqual(
            list(llm.chat(messages, stream=True))[-1]["choices"][0]["finish_reason"], "stop"
        )
        self.assertEqual(llm.embed(["one"])["data"][0]["embedding"], [0.0, 2.0])
        self.assertEqual(
            [entry[0] for entry in Handler.requests], ["/api/chat", "/api/chat", "/api/embed"]
        )


if __name__ == "__main__":
    unittest.main()
