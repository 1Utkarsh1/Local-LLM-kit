"""Optional conformance against the installed official OpenAI Python SDK.

Run LOCAL_LLM_KIT_OPENAI_INTEROP=1 pytest tests/test_responses_sdk.py.
All requests stay within FastAPI TestClient; no credentials or remote inference.
"""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LOCAL_LLM_KIT_OPENAI_INTEROP") != "1",
    reason="opt-in OpenAI SDK interoperability test",
)


class FixtureLLM:
    model_path = "fixture"

    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        tools = kwargs.get("tools")
        tool_reply = messages[-1]["role"] == "tool"
        text = '{"age":3}' if kwargs.get("response_format") else "works"
        message = {"role": "assistant", "content": text}
        if tools and not tool_reply:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_original",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"local.txt"}'},
                    }
                ],
            }
        usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
        if kwargs.get("stream"):
            if message.get("tool_calls"):
                return iter(
                    [
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_original",
                                                "function": {
                                                    "name": "read",
                                                    "arguments": '{"path":',
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {"index": 0, "function": {"arguments": '"local.txt"}'}}
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "usage": usage,
                        },
                    ]
                )
            return iter(
                [
                    {"choices": [{"delta": {"content": text[:2]}}]},
                    {
                        "choices": [{"delta": {"content": text[2:]}, "finish_reason": "stop"}],
                        "usage": usage,
                    },
                ]
            )
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": usage,
        }


@pytest.fixture
def client():
    openai = pytest.importorskip("openai")
    testclient = pytest.importorskip("fastapi.testclient")
    from local_llm_kit.server import create_app

    llm = FixtureLLM()
    with testclient.TestClient(create_app(llm=llm)) as http_client:
        with openai.OpenAI(
            api_key="unused-local-test", base_url="http://testserver/v1", http_client=http_client
        ) as sdk:
            yield sdk, llm


TOOL = {
    "type": "function",
    "name": "read",
    "description": "Read a local fixture",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


def test_sdk_responses_text_and_usage(client):
    sdk, _ = client
    response = sdk.responses.create(model="fixture", input="hello", store=False)
    assert response.output_text == "works"
    assert response.usage.input_tokens == 11 and response.usage.output_tokens == 5


def test_sdk_function_call_and_result_round_trip(client):
    sdk, llm = client
    response = sdk.responses.create(model="fixture", input="read a file", tools=[TOOL])
    call = response.output[0]
    assert call.type == "function_call" and call.call_id == "call_original"
    assert json.loads(call.arguments) == {"path": "local.txt"}
    assert llm.calls[-1][1]["tools"][0]["function"]["name"] == "read"
    follow_up = sdk.responses.create(
        model="fixture",
        input=[
            {"role": "user", "content": "read a file"},
            call.model_dump(),
            {"type": "function_call_output", "call_id": call.call_id, "output": "file content"},
        ],
        tools=[TOOL],
    )
    assert follow_up.output_text == "works"
    assert llm.calls[-1][0][-1] == {
        "role": "tool",
        "tool_call_id": "call_original",
        "content": "file content",
    }
    assert llm.calls[-1][1]["execute_tools"] is False


@pytest.mark.parametrize("tools", [[], [TOOL]])
def test_sdk_stream_manager_can_build_final_text_and_tool_calls(client, tools):
    sdk, _ = client
    with sdk.responses.stream(model="fixture", input="hello", tools=tools) as stream:
        events = list(stream)
        response = stream.get_final_response()
    assert response.status == "completed"
    assert response.usage.total_tokens == 16
    assert [event.sequence_number for event in events] == list(range(len(events)))
    if tools:
        assert response.output[0].call_id == "call_original"
        assert json.loads(response.output[0].arguments) == {"path": "local.txt"}
    else:
        assert response.output_text == "works"


def test_sdk_schema_format_and_image_parts_are_forwarded(client):
    sdk, llm = client
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
        "additionalProperties": False,
    }
    response = sdk.responses.create(
        model="fixture",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "age?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,eA=="},
                ],
            }
        ],
        text={"format": {"type": "json_schema", "name": "Age", "schema": schema}},
    )
    assert json.loads(response.output_text) == {"age": 3}
    assert llm.calls[-1][0][0]["content"][1]["type"] == "image_url"
    assert llm.calls[-1][1]["response_format"]["json_schema"]["schema"] == schema


def test_responses_invalid_fields_are_400_and_stream_failures_are_events(client):
    sdk, llm = client
    http = sdk._client
    for payload in (
        {"tools": [{"type": "function", "function": "invalid"}]},
        {"stream": "false"},
        {"text": {"format": []}},
        {"reasoning": {}},
    ):
        response = http.post("http://testserver/v1/responses", json=dict(input="hi", **payload))
        assert response.status_code == 400
    llm.chat = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure"))
    response = http.post("http://testserver/v1/responses", json={"input": "hi", "stream": True})
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["message"] == "fixture failure"
