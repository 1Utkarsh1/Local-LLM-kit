#!/usr/bin/env python3
"""Example: Autonomous Multi-Turn Agent with Self-Correction and MCP tools.

Demonstrates:
1. Defining Python functions with @tool.
2. Initializing an autonomous Agent with LLM.
3. Multi-turn tool execution with automatic error recovery.
4. Connecting external Model Context Protocol (MCP) servers.

Run:
    python examples/mcp_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_llm_kit import LLM, tool
from local_llm_kit.agent import Agent


# 1. Define custom tools
@tool
def calculate(expression: str) -> str:
    """Safely calculate a simple mathematical expression."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Unsafe characters in expression: {expression}")
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Failed to evaluate {expression}: {exc}") from exc


@tool
def lookup_stock(symbol: str) -> str:
    """Look up current price for a stock ticker symbol."""
    prices = {"AAPL": "$235.00", "GOOGL": "$182.50", "NVDA": "$138.25"}
    sym = symbol.upper().strip()
    if sym not in prices:
        raise KeyError(f"Ticker '{sym}' not found. Available tickers: {list(prices.keys())}")
    return f"{sym}: {prices[sym]}"


def main():
    print("=== Local LLM Kit: Autonomous Agent with Self-Correction ===\n")

    # Connect to local model (using echo mock backend so this demo runs 100% offline)
    llm = LLM("llama3.2:3b", backend="echo")

    # Create agent
    agent = Agent(
        llm=llm,
        tools=[calculate, lookup_stock],
        system="You are an intelligent assistant with access to math and financial tools.",
        max_steps=5,
    )

    print(f"Agent initialized with tools: {[calculate.__name__, lookup_stock.__name__]}")
    print("Running agent request: 'Calculate (100 * 2.5) + 15'...")
    res = agent.run("Calculate (100 * 2.5) + 15")
    print("\nResult:")
    print(res.content)
    print(
        f"\nExecution stats: {res.tool_calls_executed} tool calls, {res.iterations} iterations, success={res.success}"
    )


if __name__ == "__main__":
    main()
