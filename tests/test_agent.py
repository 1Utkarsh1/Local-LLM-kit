"""Unit and integration tests for autonomous Agent and self-correcting tool loops."""

import unittest
from unittest.mock import MagicMock

from local_llm_kit.agent import Agent, AgentResult
from local_llm_kit.tools import tool


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two integers."""
    return a + b


@tool
def flaky_db_query(table: str) -> str:
    """Query a table."""
    if table != "users":
        raise ValueError(f"Table '{table}' does not exist. Available tables: 'users'")
    return "User: Alice"


class TestAgent(unittest.TestCase):
    def test_single_turn_no_tools(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "The sky is blue."}}]
        }

        agent = Agent(llm=mock_llm)
        result = agent.run("What color is the sky?")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.content, "The sky is blue.")
        self.assertTrue(result.success)
        self.assertEqual(result.tool_calls_executed, 0)
        self.assertEqual(result.iterations, 1)

    def test_multi_turn_tool_execution(self):
        mock_llm = MagicMock()
        # Turn 1: Model requests calculate_sum(a=10, b=25)
        # Turn 2: Model outputs final answer
        mock_llm.chat.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "calculate_sum",
                                        "arguments": '{"a": 10, "b": 25}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The sum of 10 and 25 is 35.",
                        }
                    }
                ]
            },
        ]

        agent = Agent(llm=mock_llm, tools=[calculate_sum])
        result = agent.run("What is 10 + 25?")

        self.assertEqual(result.content, "The sum of 10 and 25 is 35.")
        self.assertTrue(result.success)
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(result.iterations, 2)
        # Verify tool output was fed into conversation history
        history = result.steps
        self.assertTrue(
            any(msg.get("role") == "tool" and msg.get("content") == "35" for msg in history)
        )

    def test_self_correcting_error_feedback(self):
        mock_llm = MagicMock()
        # Turn 1: Model calls flaky_db_query with invalid table 'accounts' -> raises ValueError
        # Turn 2: Model receives error feedback, self-corrects to 'users'
        # Turn 3: Model receives 'User: Alice' and gives final answer
        mock_llm.chat.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_err",
                                    "function": {
                                        "name": "flaky_db_query",
                                        "arguments": '{"table": "accounts"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_fix",
                                    "function": {
                                        "name": "flaky_db_query",
                                        "arguments": '{"table": "users"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Found user Alice in the users table.",
                        }
                    }
                ]
            },
        ]

        agent = Agent(llm=mock_llm, tools=[flaky_db_query])
        result = agent.run("Query accounts")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_calls_executed, 2)
        self.assertIn("Found user Alice", result.content)

        # Confirm the error was captured in history
        history = result.steps
        tool_err_msg = [m for m in history if m.get("tool_call_id") == "call_err"][0]
        self.assertIn("Table 'accounts' does not exist", tool_err_msg["content"])

    def test_max_steps_guard(self):
        mock_llm = MagicMock()
        # Model keeps calling tools forever
        mock_llm.chat.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "loop",
                                "function": {
                                    "name": "calculate_sum",
                                    "arguments": '{"a": 1, "b": 1}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        agent = Agent(llm=mock_llm, tools=[calculate_sum], max_steps=3)
        result = agent.run("Loop forever")

        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.tool_calls_executed, 3)


if __name__ == "__main__":
    unittest.main()
