"""Native chat, tools and structured outputs with mocked upstream APIs."""

import json
import unittest
from unittest import mock

from local_llm_kit import LLM, tool
from local_llm_kit.backends.ollama import OllamaBackend
from local_llm_kit.backends.openai_compat import OpenAICompatBackend


@tool
def lookup(x: int) -> dict:
    """Look up a test value."""
    return {"value": x + 1}


MESSAGES = [{"role": "user", "content": "Look up 2."}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "value", "schema": {"type": "object"}}}


def openai_response(content=None, calls=None):
    return {
        "choices": [
            {
                "message": {"content": content, "tool_calls": calls},
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ]
    }


def native_call(name="lookup", arguments=None, call_id="call_native"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"x": 2})},
    }


def sse(delta=None, finish=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}]}


def make_llm(backend):
    return LLM("test-model", backend_instance=backend, use_cache=False)


class TestOpenAINativeChat(unittest.TestCase):
    def test_native_calls_return_unexecuted_with_original_ids(self):
        backend = OpenAICompatBackend("test-model")
        with mock.patch.object(
            backend, "_post_json", return_value=openai_response(calls=[native_call()])
        ) as post:
            llm = make_llm(backend)
            response = llm.chat(MESSAGES, tools=[lookup], execute_tools=False)
        self.assertEqual(post.call_args.args[0], "/chat/completions")
        payload = post.call_args.args[1]
        self.assertEqual(payload["messages"], MESSAGES)
        self.assertEqual(payload["tools"], [lookup.spec])
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_native")
        self.assertFalse(any(message["role"] == "tool" for message in llm.memory.messages))
        self.assertEqual(llm.list_tools(), [])

    def test_tool_loop_preserves_native_messages_and_request_scope(self):
        backend = OpenAICompatBackend("test-model")
        replies = [
            openai_response(calls=[native_call()]),
            openai_response("The value is 3."),
            openai_response("Hello."),
        ]
        with mock.patch.object(backend, "_post_json", side_effect=replies) as post:
            llm = make_llm(backend)
            response = llm.chat(MESSAGES, tools=[lookup])
            self.assertEqual(response["choices"][0]["message"]["content"], "The value is 3.")
            followup = post.call_args_list[1].args[1]["messages"]
            self.assertEqual(followup[-1]["role"], "tool")
            self.assertEqual(followup[-1]["tool_call_id"], "call_native")
            self.assertEqual(json.loads(followup[-1]["content"]), {"value": 3})
            self.assertNotIn("function_call", followup[-2])
            self.assertNotIn("name", followup[-1])
            self.assertEqual(llm.list_tools(), [])
            llm.chat([{"role": "user", "content": "Hello"}])
            self.assertNotIn("tools", post.call_args.args[1])

    def test_structured_output_and_sampling_reach_native_endpoint(self):
        backend = OpenAICompatBackend("test-model")
        with mock.patch.object(
            backend, "_post_json", return_value=openai_response('{"value": 3}')
        ) as post:
            response = make_llm(backend).chat(
                MESSAGES, response_format=SCHEMA, seed=7, stop=["END"]
            )
        payload = post.call_args.args[1]
        self.assertEqual(payload["response_format"], SCHEMA)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(json.loads(response["choices"][0]["message"]["content"]), {"value": 3})

    def test_stream_assembles_interleaved_tool_argument_fragments(self):
        backend = OpenAICompatBackend("test-model")
        events = [
            sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "function": {"name": "lookup", "arguments": '{"x":'},
                        },
                        {
                            "index": 1,
                            "id": "call_b",
                            "function": {"name": "lookup", "arguments": '{"x":'},
                        },
                    ]
                }
            ),
            sse(
                {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": "4}"}},
                        {"index": 0, "function": {"arguments": "2}"}},
                    ]
                }
            ),
            sse(finish="tool_calls"),
        ]
        with mock.patch.object(backend, "_post_sse", return_value=iter(events)):
            chunks = list(
                make_llm(backend).chat(MESSAGES, tools=[lookup], stream=True, execute_tools=False)
            )
        final = chunks[-1]["choices"][0]
        self.assertEqual(final["finish_reason"], "tool_calls")
        calls = final["delta"]["tool_calls"]
        self.assertEqual([call["index"] for call in calls], [0, 1])
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual(
            [json.loads(call["function"]["arguments"]) for call in calls], [{"x": 2}, {"x": 4}]
        )

    def test_streaming_tool_execution_requeries_chat_with_results(self):
        backend = OpenAICompatBackend("test-model")
        streams = [
            iter([sse({"tool_calls": [dict(native_call(), index=0)]}), sse(finish="tool_calls")]),
            iter([sse({"content": "Done."}), sse(finish="stop")]),
        ]
        with mock.patch.object(backend, "_post_sse", side_effect=streams) as post:
            llm = make_llm(backend)
            chunks = list(llm.chat(MESSAGES, tools=[lookup], stream=True))
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.args[1]["messages"][-1]["role"], "tool")
        self.assertEqual(post.call_args.args[1]["messages"][-1]["tool_call_id"], "call_native")
        self.assertEqual(
            "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks), "Done."
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(llm.list_tools(), [])

    def test_native_errors_are_not_retried_as_prompt_completions(self):
        backend = OpenAICompatBackend("test-model")
        with mock.patch.object(
            backend, "_post_json", side_effect=TypeError("upstream failure")
        ) as post:
            with self.assertRaisesRegex(TypeError, "upstream failure"):
                make_llm(backend).chat(MESSAGES)
        self.assertEqual(post.call_count, 1)


class TestOllamaNativeChat(unittest.TestCase):
    def test_image_parts_convert_to_native_images_without_fetching_urls(self):
        backend = OllamaBackend("test-model")
        content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]
        messages = [{"role": "user", "content": content}]
        with mock.patch.object(
            backend, "_post_json", return_value={"message": {"content": "an image"}}
        ) as post:
            make_llm(backend).chat(messages)
        native = post.call_args.args[1]["messages"][0]
        self.assertEqual(native["content"], "Describe this")
        self.assertEqual(native["images"], ["aGVsbG8="])
        self.assertIsInstance(messages[0]["content"], list)
        content[1]["image_url"]["url"] = "https://example.com/private.png"
        with mock.patch.object(backend, "_post_json") as post:
            with self.assertRaisesRegex(ValueError, "inline base64"):
                make_llm(backend).chat(messages)
            post.assert_not_called()

    def test_native_tool_results_use_ollama_shapes(self):
        backend = OllamaBackend("test-model")
        replies = [
            {
                "message": {
                    "content": "",
                    "thinking": "Call lookup first.",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"x": 2}}}],
                }
            },
            {"message": {"content": "The value is 3."}, "done": True},
        ]
        with mock.patch.object(backend, "_post_json", side_effect=replies) as post:
            result = make_llm(backend).chat(MESSAGES, tools=[lookup])
        self.assertEqual(result["choices"][0]["message"]["content"], "The value is 3.")
        self.assertEqual(post.call_args.args[0], "/api/chat")
        payload = post.call_args.args[1]
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["messages"][-2]["thinking"], "Call lookup first.")
        self.assertEqual(
            payload["messages"][-2]["tool_calls"][0]["function"]["arguments"], {"x": 2}
        )
        self.assertEqual(payload["messages"][-1]["tool_name"], "lookup")
        self.assertNotIn("tool_call_id", payload["messages"][-1])

    def test_tool_only_stream_chunks_are_not_dropped(self):
        backend = OllamaBackend("test-model")
        events = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"x": 2}}}],
                },
                "done": False,
            },
            {"message": {"content": ""}, "done": True},
        ]
        with mock.patch.object(backend, "_post_json_lines", return_value=iter(events)):
            chunks = list(
                make_llm(backend).chat(MESSAGES, tools=[lookup], execute_tools=False, stream=True)
            )
        final = chunks[-1]["choices"][0]
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual(
            json.loads(final["delta"]["tool_calls"][0]["function"]["arguments"]), {"x": 2}
        )

    def test_schema_and_seed_are_mapped_to_ollama_fields(self):
        backend = OllamaBackend("test-model")
        with mock.patch.object(
            backend, "_post_json", return_value={"message": {"content": "{}"}}
        ) as post:
            make_llm(backend).chat(MESSAGES, response_format=SCHEMA, seed=7)
        self.assertEqual(post.call_args.args[1]["format"], {"type": "object"})
        self.assertEqual(post.call_args.args[1]["options"]["seed"], 7)

    def test_structured_stream_can_call_tools_without_json_repair_requests(self):
        backend = OllamaBackend("test-model")
        events = [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"x": 2}}}]
                },
                "done": True,
            }
        ]
        with (
            mock.patch.object(backend, "_post_json_lines", return_value=iter(events)) as stream,
            mock.patch.object(
                backend, "_post_json", side_effect=AssertionError("unexpected JSON repair request")
            ),
        ):
            chunks = list(
                make_llm(backend).chat(
                    MESSAGES,
                    tools=[lookup],
                    response_format=SCHEMA,
                    stream=True,
                    execute_tools=False,
                )
            )
        self.assertEqual(stream.call_count, 1)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")


class TestReportedUsage(unittest.TestCase):
    def test_chat_and_completion_keep_reported_counts(self):
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 64,
            "total_tokens": 184,
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
        for route in ("chat", "complete"):
            with self.subTest(route=route):
                backend = OpenAICompatBackend("test-model")
                payload = openai_response("Short text")
                payload["choices"][0]["text"] = "Short text"
                payload["usage"] = usage
                with mock.patch.object(backend, "_post_json", return_value=payload):
                    llm = make_llm(backend)
                    result = llm.chat(MESSAGES) if route == "chat" else llm.complete("Hello")
                self.assertEqual(result["usage"], usage)

    def test_tool_rounds_sum_each_reported_usage(self):
        backend = OpenAICompatBackend("test-model")
        replies = [openai_response(calls=[native_call()]), openai_response("Done")]
        replies[0]["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
        replies[1]["usage"] = {"prompt_tokens": 150, "completion_tokens": 10}
        with mock.patch.object(backend, "_post_json", side_effect=replies):
            result = make_llm(backend).chat(MESSAGES, tools=[lookup])
        self.assertEqual(
            result["usage"], {"prompt_tokens": 250, "completion_tokens": 30, "total_tokens": 280}
        )

    def test_ollama_native_counts_reach_public_responses(self):
        backend = OllamaBackend("test-model")
        payload = {
            "message": {"content": "Short text"},
            "response": "Short text",
            "prompt_eval_count": 100,
            "eval_count": 64,
        }
        with mock.patch.object(backend, "_post_json", return_value=payload):
            for result in (make_llm(backend).chat(MESSAGES), make_llm(backend).complete("Hi")):
                self.assertEqual(
                    result["usage"],
                    {"prompt_tokens": 100, "completion_tokens": 64, "total_tokens": 164},
                )

    def test_zero_counts_and_invalid_count_fallback(self):
        backend = OpenAICompatBackend("test-model")
        for reported in (
            {"prompt_tokens": 0, "completion_tokens": 0},
            {"prompt_tokens": -1, "completion_tokens": 4},
            {"prompt_tokens": True, "completion_tokens": 4},
            {"prompt_tokens": "100", "completion_tokens": 4},
            None,
        ):
            with self.subTest(reported=reported):
                payload = dict(openai_response("Hello"), usage=reported)
                with mock.patch.object(backend, "_post_json", return_value=payload):
                    result = make_llm(backend).chat(MESSAGES)
                if reported == {"prompt_tokens": 0, "completion_tokens": 0}:
                    self.assertEqual(result["usage"]["total_tokens"], 0)
                else:
                    self.assertGreater(result["usage"]["prompt_tokens"], 0)
                    self.assertTrue(all(type(value) is int for value in result["usage"].values()))


if __name__ == "__main__":
    unittest.main()
