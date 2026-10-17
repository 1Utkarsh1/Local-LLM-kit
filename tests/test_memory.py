"""History truncation keeps tool exchanges atomic.

Regression tests for split tool-call/result pairs: truncation and
summarization must never strand an assistant ``tool_calls`` message without
its ``tool`` results (or vice versa).
"""

import unittest

from local_llm_kit.memory import MessageHistory


def _exchange(n):
    return [
        {"role": "user", "content": "question %d" % n},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-%d" % n,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"n": %d}' % n},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-%d" % n, "content": "sunny %d" % n},
    ]


class TestToolExchangeTruncation(unittest.TestCase):
    def test_truncation_removes_whole_exchanges(self):
        history = MessageHistory(context_window=10**9)
        history.add_messages([{"role": "system", "content": "sys"}])
        for n in range(4):
            history.add_messages(_exchange(n))
        history.context_window = 1  # force truncation of everything non-system
        history._truncate_if_needed()
        roles = [m.get("role") for m in history.messages]
        self.assertEqual(roles[0], "system")
        # Every remaining assistant tool_calls message keeps its tool result.
        for i, message in enumerate(history.messages):
            if message.get("tool_calls"):
                following = [m.get("role") for m in history.messages[i + 1 : i + 3]]
                self.assertIn("tool", following)
        # No orphaned tool results without their assistant call.
        for i, message in enumerate(history.messages):
            if message.get("role") == "tool":
                owners = [
                    m
                    for m in history.messages[:i]
                    if m.get("role") == "assistant" and m.get("tool_calls")
                ]
                self.assertTrue(owners)

    def test_summarizer_does_not_start_inside_exchange(self):
        history = MessageHistory(context_window=10**9)
        history.add_messages(_exchange(0) + _exchange(1))
        history._summarize_history()
        first_kept = history.messages[-1] if history.messages else None
        # The kept tail must not begin with a bare tool result.
        kept_roles = [
            m.get("role")
            for m in history.messages
            if m.get("role") != "system" or "summarized" not in str(m.get("content", ""))
        ]
        self.assertNotEqual(kept_roles[0], "tool")
        self.assertIsNotNone(first_kept)

    def test_plain_history_truncation_unchanged(self):
        history = MessageHistory(context_window=10**9)
        history.add_messages([{"role": "user", "content": "hello world, this is a test"}])
        history.context_window = 1
        history._truncate_if_needed()
        # Single over-limit message takes the pre-existing summarize path.
        self.assertEqual(len(history.messages), 2)
        self.assertIn("summarized", history.messages[0]["content"])
        self.assertEqual(history.messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
