"""A bounded tool-calling loop with request-scoped tools and error feedback."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .tools import ToolCall, ToolRegistry, _validate_tool_timeout, normalize_tools


@dataclass
class AgentResult:
    """Final answer, full exchange, and number of attempted tool calls."""

    content: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_executed: int = 0
    success: bool = True
    iterations: int = 0


class Agent:
    """Run a synchronous LLM with Python tools and connected MCPServer instances.

    Tools execute only when explicitly passed here. MCP tool names are prefixed
    with the server name. The caller owns MCP connection lifetimes. A timeout
    bounds waiting for a Python tool, but cannot undo or interrupt its side effects.
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[Sequence[Any]] = None,
        mcp_clients: Optional[Sequence[Any]] = None,
        system: Optional[str] = None,
        max_steps: int = 10,
        tool_timeout: Optional[float] = 10.0,
    ):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        _validate_tool_timeout(tool_timeout)
        self.llm = llm
        self.system = system
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.mcp_clients = list(mcp_clients or [])
        self._registry = ToolRegistry()
        for item in tools or []:
            if callable(item):
                spec = normalize_tools([item])[0]["function"]
                implementation = item
            elif isinstance(item, dict) and normalize_tools([item]):
                spec = normalize_tools([item])[0]["function"]
                implementation = None
            else:
                raise TypeError("Agent tools must be callables or function tool specs.")
            self._register(spec, implementation)

    def _register(self, spec: Dict[str, Any], implementation: Any) -> None:
        name = spec["name"]
        if self._registry.has_function(name):
            raise ValueError("Duplicate agent tool name: %s" % name)
        self._registry.add_function(name=name, schema=spec, implementation=implementation)

    def _collect_all_tool_specs(self) -> List[Dict[str, Any]]:
        from .mcp import sanitize_tool_name

        # Rebuild the MCP portion per run so disconnected/changed servers cannot
        # leave stale implementations behind in a reused agent.
        registry = ToolRegistry()
        registry.functions.update(self._registry.functions)
        registry.implementations.update(self._registry.implementations)
        for client in self.mcp_clients:
            for tool in client.list_tools():
                name = sanitize_tool_name("%s__%s" % (client.name, tool.name))
                if registry.has_function(name):
                    raise ValueError("Duplicate agent tool name: %s" % name)

                def invoke(_client=client, _name=tool.name, **arguments):
                    return _client.call_tool(_name, arguments)

                registry.add_function(name, tool.to_openai(name)["function"], invoke)
        self._run_registry = registry
        return registry.get_tool_list()

    def run(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Return a final answer or an unsuccessful result at the turn limit.

        Invalid JSON tool arguments are reported to the model without executing
        them. They are never repaired into an action the model did not specify.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if kwargs.get("stream"):
            raise ValueError("Agent.run does not support stream=True.")
        history = [{"role": "system", "content": self.system}] if self.system else []
        history.extend(dict(message) for message in messages or [])
        history.append({"role": "user", "content": prompt})
        specs = self._collect_all_tool_specs()
        by_name = {spec["function"]["name"]: spec for spec in specs}
        calls = 0
        for step in range(self.max_steps):
            chat_kwargs = dict(kwargs, execute_tools=False)
            if specs:
                chat_kwargs["tools"] = specs
            response = self.llm.chat(messages=history, **chat_kwargs)
            choices = response.get("choices", [])
            if not choices:
                return AgentResult("", history, calls, False, step + 1)
            message = dict(choices[0].get("message", {}), role="assistant")
            history.append(message)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return AgentResult(message.get("content") or "", history, calls, True, step + 1)
            for record in tool_calls:
                calls += 1
                try:
                    call = ToolCall.from_dict(record)
                    result = self._run_registry.execute_call(
                        call, list(by_name), by_name.get(call.name), self.tool_timeout
                    )
                except (TypeError, ValueError) as exc:
                    result = (
                        "Error: invalid tool call (%s). Provide a complete JSON object."
                        % str(exc)[:200]
                    )
                function = record.get("function", {}) if isinstance(record, dict) else {}
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            record.get("id", "call_%s" % calls)
                            if isinstance(record, dict)
                            else "call_%s" % calls
                        ),
                        "name": function.get("name", "") if isinstance(function, dict) else "",
                        "content": result,
                    }
                )
        return AgentResult("", history, calls, False, self.max_steps)
