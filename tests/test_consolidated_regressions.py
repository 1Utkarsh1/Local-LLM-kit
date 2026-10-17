"""Regression cases found by reviewing the community PRs together."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from local_llm_kit import LLM, Agent, ConversationLog, PromptTemplate, TemplateRegistry, tool
from local_llm_kit.backends.apple_fm import AppleFMBackend, _stop_deltas
from local_llm_kit.backends.echo import EchoBackend
from local_llm_kit.discovery import ModelRecord, ServerReport, normalize_base_url
from local_llm_kit.hardware import estimate_model_vram, recommend_quantization
from local_llm_kit.mcp import MCPTool, _HttpTransport
from local_llm_kit.rag import RAGPipeline, SimpleVectorStore, bm25_scores, reciprocal_rank_fusion
from local_llm_kit.retry import RetryPolicy, retry_after_seconds
from local_llm_kit.structured import extract_json, validate_schema
from tests.test_apple_fm import FakeSession, FakeSystemModel, install_fake_sdk, mac_only


class ToolBackend(EchoBackend):
    supports_native_chat = True

    def __init__(self):
        super().__init__()
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        if messages[-1]["role"] == "tool":
            return {"text": "done"}
        return {
            "text": "",
            "tool_calls": [
                {"id": "original-id", "function": {"name": "renamed", "arguments": '{"value": 4}'}}
            ],
        }


def test_agent_real_llm_executes_only_once_and_preserves_decorator_name():
    seen = []

    @tool(name="renamed")
    def action(value: int):
        seen.append(value)
        return value * 2

    backend = ToolBackend()
    llm = LLM("fixture", backend_instance=backend)
    result = Agent(llm, tools=[action]).run("call it")
    assert result.success and result.content == "done"
    assert seen == [4] and result.tool_calls_executed == 1
    assert backend.requests[-1][-1]["tool_call_id"] == "original-id"
    assert backend.requests[-1][-1]["content"] == "8"
    assert not llm.list_tools()


def test_agent_does_not_repair_truncated_executable_arguments():
    seen = []
    llm = mock.Mock()
    llm.chat.side_effect = [
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "bad",
                                "function": {"name": "action", "arguments": '{"value":"unfinished'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "ask for a correction"}}]},
    ]

    def action(value):
        seen.append(value)

    result = Agent(llm, tools=[action]).run("test")
    assert seen == []
    assert "Error" in result.steps[2]["content"]


def test_schema_types_and_nested_constraints_are_enforced():
    pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 0}},
        "required": ["age"],
    }
    for age in ("not-an-integer", -1, True):
        with pytest.raises(ValueError):
            validate_schema({"age": age}, schema)
    validate_schema({"age": 3}, schema)


def test_json_repair_preserves_quoted_literals_and_delimiters():
    assert extract_json('{"text":"True, None, } and False",}') == {
        "text": "True, None, } and False"
    }
    assert extract_json("{'text': 'True, None', 'flag': True}") == {
        "text": "True, None",
        "flag": True,
    }
    with pytest.raises(ValueError):
        extract_json('{"path":"/unfinished')


def test_missing_schema_validator_fails_before_inference():
    llm = mock.Mock()
    with mock.patch.dict("sys.modules", {"jsonschema": None}):
        from local_llm_kit import chat_structured

        with pytest.raises(ImportError, match="structured"):
            chat_structured(llm, [{"role": "user", "content": "test"}], schema={"type": "object"})
    llm.chat.assert_not_called()


def test_locality_rejects_remote_and_unknown_compat_endpoints():
    with pytest.raises(RuntimeError, match="non-loopback"):
        LLM(
            "remote-model",
            backend="openai-compat",
            use_cache=False,
            backend_kwargs={"base_url": "https://example.com/v1"},
            require_local=True,
        )
    unknown = ServerReport(
        "http://127.0.0.1:1/v1", "generic", live=True, models=[ModelRecord("local-sounding-model")]
    )
    assert not unknown.local_models
    with mock.patch("local_llm_kit.discovery.probe_endpoint", return_value=unknown):
        with pytest.raises(RuntimeError, match="unknown"):
            LLM(
                "local-sounding-model", backend="openai-compat", use_cache=False, require_local=True
            )
    assert normalize_base_url("http://[::1]:1234/v1/models") == "http://[::1]:1234/v1"


def test_http_mcp_returns_result_before_sse_connection_closes():
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                (
                    "data: "
                    + json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}})
                    + "\n\n"
                ).encode()
            )
            self.wfile.flush()
            release.wait(2)
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        transport = _HttpTransport("http://127.0.0.1:%s/mcp" % server.server_port, timeout=0.3)
        assert transport.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"] == {
            "ok": True
        }
        assert not release.is_set()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join()


def test_mcp_collision_is_atomic_and_closes_new_connection():
    llm = LLM("echo", backend="echo", use_cache=False)

    @tool(name="files__read")
    def read():
        return "local"

    llm.add_tool(read)
    server = mock.Mock(name="server", _opened=False, closed=True)
    server.name = "files"
    server.list_tools.return_value = [MCPTool("write"), MCPTool("read")]
    with pytest.raises(ValueError, match="collides"):
        llm.add_mcp_server(server)
    server.close.assert_called_once()
    assert [item["function"]["name"] for item in llm.list_tools()] == ["files__read"]


def test_in_memory_logging_is_shared_and_interaction_is_atomic():
    with ConversationLog(":memory:") as log:
        with ThreadPoolExecutor(max_workers=3) as pool:
            ids = list(
                pool.map(
                    lambda i: log.log_interaction(
                        "echo", [{"role": "user", "content": str(i)}], "ok"
                    ),
                    range(8),
                )
            )
        assert len(log.get_conversations()) == 8
        assert all(len(log.get_messages(cid)) == 2 for cid in ids)
        before = log.get_conversations()
        with pytest.raises(ValueError):
            log.log_interaction("echo", [{"content": "invalid role"}], "bad")
        assert log.get_conversations() == before


def test_logging_preserves_tool_ids_usage_and_complete_multimodal_payload():
    call = {
        "id": "stable-call-id",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "stable-call-id", "name": "read", "content": "file"},
    ]
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "done"}, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    with ConversationLog(":memory:") as log:
        cid = log.log_interaction("fixture", messages, response)
        saved = log.get_messages(cid)
        exported = json.loads(log.export_conversation(cid, "openai"))["messages"]
        assert exported[:3] == messages
        assert saved[-1]["finish_reason"] == "length"
        assert saved[-1]["prompt_tokens"] == 12 and saved[-1]["completion_tokens"] == 3
        assert log.get_tool_calls(saved[1]["id"])[0]["call_id"] == "stable-call-id"


@pytest.mark.parametrize("stream", [False, True])
def test_llm_logging_is_opt_in_and_handles_completed_streams(stream):
    with ConversationLog(":memory:") as log:
        llm = LLM("echo", backend="echo", use_cache=False, conversation_log=log)
        result = llm.chat([{"role": "user", "content": "hi"}], stream=stream)
        if stream:
            list(result)
        cid = log.get_conversations()[0]["id"]
        assert len(log.get_messages(cid)) == 2
        assert (
            json.loads(log.export_conversation(cid, "openai"))["messages"][-1]["role"]
            == "assistant"
        )


def test_plain_templates_work_without_yaml_and_directory_errors_are_visible(tmp_path):
    text = tmp_path / "plain.txt"
    text.write_text('Hello {{name}} {"json":true}', encoding="utf-8")
    with mock.patch.dict("sys.modules", {"yaml": None}):
        assert (
            PromptTemplate.from_file(str(text)).render(name="World") == 'Hello World {"json":true}'
        )
    (tmp_path / "bad.json").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        TemplateRegistry().load_directory(str(tmp_path))


def test_hardware_rejects_unknown_measurements_and_invalid_assumptions():
    with mock.patch(
        "local_llm_kit.hardware.get_system_memory",
        return_value={"available_gpu_vram_gb": 0, "available_ram_gb": 0, "notes": []},
    ):
        with pytest.raises(ValueError, match="unknown"):
            recommend_quantization(8)
    for kwargs in (
        {"params_b": float("nan")},
        {"params_b": 8, "quant": "made-up"},
        {"params_b": 8, "kv_quant": "unknown"},
        {"params_b": 8, "context_length": 2.5},
    ):
        with pytest.raises(ValueError):
            estimate_model_vram(**kwargs)
    exact = estimate_model_vram(8, num_layers=32, num_kv_heads=8, head_dim=128)
    assert not exact["architecture_assumed"]


def test_retrieval_filter_delete_hybrid_mmr_and_pipeline():
    def embed(text):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    store = SimpleVectorStore(embed_fn=embed)
    ids = store.add(
        ["alpha one", "alpha two", "beta only"],
        metadatas=[{"group": "a"}, {"group": "a"}, {"group": "b"}],
    )
    hits = store.search("alpha", mode="hybrid", lambda_mult=0.4, filter={"group": "a"})
    assert len(hits) == 2 and all(hit["metadata"]["group"] == "a" for hit in hits)
    assert store.delete([ids[0]]) == 1
    assert store.search("beta", mode="lexical", filter={"group": "b"})[0]["id"] == ids[2]
    pipeline = RAGPipeline(LLM("echo", backend="echo"))
    pipeline.add_documents("small offline alpha document", chunk_size=100, chunk_overlap=0)
    assert pipeline.query("alpha")["sources"]
    assert reciprocal_rank_fusion([[1, 1]])[1] == reciprocal_rank_fusion([[1]])[1]
    with pytest.raises(ValueError):
        bm25_scores("x", ["x"], k1=float("nan"))


def test_retry_policy_rejects_nonfinite_delays_and_headers():
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RetryPolicy(initial_delay=value)
        assert retry_after_seconds({"Retry-After": str(value)}) is None


def test_apple_sync_and_stream_timeouts_cancel_cooperative_sdk():
    FakeSystemModel.available, FakeSystemModel.window = True, 4096
    cancelled = threading.Event()

    async def respond(self, *_args, **_kwargs):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    async def stream(self, *_args, **_kwargs):
        try:
            await asyncio.sleep(60)
            yield "never"
        finally:
            cancelled.set()

    platform_patch, version_patch = mac_only()
    with platform_patch, version_patch, install_fake_sdk():
        backend = AppleFMBackend(timeout=0.05)
        with mock.patch.object(FakeSession, "respond", respond):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                backend.generate("hi")
        assert cancelled.is_set()
        cancelled.clear()
        with mock.patch.object(FakeSession, "stream_response", stream):
            with pytest.raises(TimeoutError):
                list(backend.generate_stream("hi"))
        assert cancelled.wait(0.5)
    assert "".join(_stop_deltas(iter(["ok ST", "OP leak"]), ["STOP"])) == "ok "


def test_mcp_documented_positional_name_constructs_and_registers_server():
    llm = LLM("echo", backend="echo", use_cache=False)
    server = mock.Mock(_opened=True, closed=False)
    server.list_tools.return_value = [MCPTool("read")]
    with mock.patch("local_llm_kit.mcp.MCPServer", return_value=server) as constructor:
        assert llm.add_mcp_server("files", command="fixture") == ["files__read"]
    assert constructor.call_args.args[0] == "files"
    llm.close()
    assert llm.list_tools() == []


def test_logging_deduplicates_full_history_and_tool_calls():
    call = {"id": "same", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    user = {"role": "user", "content": "hello"}
    assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
    with ConversationLog(":memory:") as log:
        cid = log.log_interaction(
            "fixture", [user], {"choices": [{"message": assistant}]}, tools_used=[call]
        )
        messages = log.get_messages(cid)
        assert len(log.get_tool_calls(messages[-1]["id"])) == 1
        log.log_interaction(
            "fixture",
            [user, assistant, {"role": "tool", "tool_call_id": "same", "content": "result"}],
            "done",
            conversation_id=cid,
        )
        assert len(log.get_messages(cid)) == 4


def test_hardware_does_not_invent_ram_or_pool_gpu_capacity():
    from local_llm_kit.hardware import get_system_memory

    def command(*args):
        if args[0] == "nvidia-smi":
            return "8192,4096\n8192,3072\n"
        raise OSError("unavailable")

    with (
        mock.patch("local_llm_kit.hardware.sys.platform", "linux"),
        mock.patch("local_llm_kit.hardware.open", side_effect=OSError("no meminfo"), create=True),
        mock.patch("local_llm_kit.hardware._command", side_effect=command),
    ):
        measured = get_system_memory()
    assert measured["total_ram_gb"] == 0
    assert measured["gpu_vram_gb"] == 8 and measured["available_gpu_vram_gb"] == 4
    assert len(measured["gpus"]) == 2


def test_reasoning_metadata_survives_nonstream_and_stream_native_chat():
    from local_llm_kit.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend("fixture", max_retries=0)
    llm = LLM("fixture", backend_instance=backend)
    with mock.patch.object(
        backend,
        "_post_json",
        return_value={
            "choices": [{"message": {"content": "answer", "reasoning_content": "metadata"}}]
        },
    ):
        response = llm.chat([{"role": "user", "content": "hello"}])
    assert response["choices"][0]["message"]["reasoning_content"] == "metadata"
    with mock.patch.object(
        backend,
        "_post_sse",
        return_value=iter(
            [
                {"choices": [{"delta": {"reasoning_content": "metadata"}}]},
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            ]
        ),
    ):
        chunks = list(llm.chat([{"role": "user", "content": "hello"}], stream=True))
    assert (
        "".join(chunk["choices"][0]["delta"].get("reasoning_content", "") for chunk in chunks)
        == "metadata"
    )


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_invalid_tool_timeout_fails_before_inference_or_execution(timeout):
    backend = ToolBackend()
    llm = LLM("fixture", backend_instance=backend)
    with pytest.raises(ValueError, match="tool_timeout"):
        llm.chat([{"role": "user", "content": "hello"}], tool_timeout=timeout)
    assert backend.requests == []
