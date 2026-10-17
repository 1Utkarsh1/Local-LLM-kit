"""
Autonomous multi-turn agent loop with self-correcting tool execution.

Small local models (Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B) frequently hallucinate
tool arguments, emit invalid JSON, or encounter tool exceptions. Traditional
wrappers abort on the first error.

This module provides :class:`Agent`:
1. Multi-turn execution combining python functions (``@tool``) and external MCP servers.
2. Resilient JSON argument extraction via :mod:`local_llm_kit.structured`.
3. **Self-correcting error recovery**: Tool runtime exceptions are caught, formatted,
   and fed back as ``tool`` messages so the model can inspect errors and retry.
4. Execution loop limits (``max_steps``) preventing infinite loops.

Example::

    from local_llm_kit import LLM, tool
    from local_llm_kit.agent import Agent
    from local_llm_kit.mcp import MCPClient

    @tool
    def search_local_files(query: str) -> str:
        '''Search local documents.'''
        return f"Found 2 matches for {query}"

    llm = LLM("llama3.2:3b", backend="ollama")
    agent = Agent(llm, tools=[search_local_files], max_steps=5)
    result = agent.run("What files mention 'quarterly budget'?")
    print(result.content)

Python 3.9+ compatible.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .structured import extract_json
from .tools import normalize_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an agent execution run."""

    content: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_executed: int = 0
    success: bool = True
    iterations: int = 0


class Agent:
    """Multi-turn tool-calling agent with error self-correction.

    Args:
        llm: A :class:`local_llm_kit.LLM` instance.
        tools: List of Python functions (e.g. decorated with ``@tool``) or tool specs.
        mcp_clients: Optional list of :class:`local_llm_kit.mcp.MCPClient` instances.
        system: Optional system instruction prompt.
        max_steps: Maximum reasoning / tool-execution turns allowed. Defaults to ``10``.
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[Sequence[Any]] = None,
        mcp_clients: Optional[Sequence[Any]] = None,
        system: Optional[str] = None,
        max_steps: int = 10,
    ):
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.llm = llm
        self.system = system
        self.max_steps = max_steps
        self.mcp_clients = list(mcp_clients or [])

        # Index python tools
        self._python_tool_funcs: Dict[str, Callable] = {}
        self._tool_specs: List[Dict[str, Any]] = []

        if tools:
            for item in tools:
                if callable(item):
                    name = getattr(item, "__name__", str(item))
                    self._python_tool_funcs[name] = item
                elif isinstance(item, dict) and "function" in item:
                    fname = item["function"].get("name")
                    if fname and hasattr(item, "fn"):
                        self._python_tool_funcs[fname] = item.fn
            self._tool_specs.extend(normalize_tools(list(tools)))

    def _collect_all_tool_specs(self) -> List[Dict[str, Any]]:
        """Collect tool specs from local python tools and active MCP clients."""
        specs = list(self._tool_specs)
        for client in self.mcp_clients:
            try:
                specs.extend(client.list_tools())
            except Exception as exc:
                logger.warning("Failed to list tools from MCP client: %s", exc)
        return specs

    def _execute_tool_call(self, name: str, raw_args: Any) -> str:
        """Execute a tool by name, handling argument parsing and self-correction."""
        # 1. Parse arguments safely
        if isinstance(raw_args, str):
            try:
                args = extract_json(raw_args) if raw_args.strip() else {}
            except Exception as exc:
                return (
                    f"Error parsing JSON arguments for tool '{name}': {exc}. "
                    f"Raw input was: {raw_args!r}. Please provide valid JSON arguments."
                )
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        # 2. Check local Python tools
        if name in self._python_tool_funcs:
            fn = self._python_tool_funcs[name]
            try:
                result = fn(**args)
                return str(result)
            except Exception as exc:
                logger.debug("Tool %r raised exception: %s", name, exc)
                return (
                    f"Error executing tool '{name}': {type(exc).__name__}: {exc}. "
                    "Please check the required argument types and values and try again."
                )

        # 3. Check MCP clients
        for client in self.mcp_clients:
            try:
                # Attempt call on client
                return client.call_tool(name, args)
            except Exception as exc:
                # If error is a tool failure or not found on this client
                err_msg = str(exc)
                if "not found" in err_msg.lower():
                    continue
                return f"MCP tool '{name}' error: {err_msg}. Please review input arguments."

        return f"Tool '{name}' is not registered or available. Available tools: {list(self._python_tool_funcs.keys())}"

    def run(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent loop until the model delivers a final answer or hits max_steps.

        Args:
            prompt: User request.
            messages: Optional prior conversation history.
            **kwargs: Additional options forwarded to ``llm.chat()``.

        Returns:
            :class:`AgentResult` containing final text, step history, and execution metrics.
        """
        history: List[Dict[str, Any]] = []
        if self.system:
            history.append({"role": "system", "content": self.system})
        if messages:
            history.extend(messages)
        history.append({"role": "user", "content": prompt})

        tool_specs = self._collect_all_tool_specs()
        tool_calls_executed = 0

        for step in range(self.max_steps):
            chat_kwargs = dict(kwargs)
            if tool_specs:
                chat_kwargs["tools"] = tool_specs

            resp = self.llm.chat(messages=history, **chat_kwargs)
            choices = resp.get("choices", [])
            if not choices:
                return AgentResult(
                    content="",
                    steps=history,
                    tool_calls_executed=tool_calls_executed,
                    success=False,
                    iterations=step + 1,
                )

            assistant_msg = choices[0].get("message", {})
            history.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # Model finished its turn and provided a final text answer
                final_content = assistant_msg.get("content") or ""
                return AgentResult(
                    content=final_content,
                    steps=history,
                    tool_calls_executed=tool_calls_executed,
                    success=True,
                    iterations=step + 1,
                )

            # Execute tool calls
            for tc in tool_calls:
                tool_calls_executed += 1
                tc_id = tc.get("id", f"call_{tool_calls_executed}")
                fn_info = tc.get("function", {})
                name = fn_info.get("name", "")
                raw_args = fn_info.get("arguments", "{}")

                result_text = self._execute_tool_call(name, raw_args)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": result_text,
                }
                history.append(tool_msg)

        # Reached max_steps without terminating in pure text
        last_content = history[-1].get("content", "") if history else ""
        return AgentResult(
            content=last_content,
            steps=history,
            tool_calls_executed=tool_calls_executed,
            success=False,
            iterations=self.max_steps,
        )
