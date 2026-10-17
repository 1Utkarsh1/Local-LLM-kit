"""Unit and regression tests for resilient JSON repair and structured chat."""

import json
import unittest
from unittest.mock import MagicMock

from local_llm_kit.structured import chat_structured, extract_json, repair_json, validate_schema


class TestStructuredJSON(unittest.TestCase):
    def test_repair_markdown_code_fences(self):
        raw = '```json\n{\n  "status": "ok",\n  "code": 200\n}\n```'
        repaired = repair_json(raw)
        data = json.loads(repaired)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["code"], 200)

    def test_repair_conversational_wrapping(self):
        raw = 'Here is the result you requested:\n{"summary": "done"}\nLet me know if you need anything else!'
        data = extract_json(raw)
        self.assertEqual(data["summary"], "done")

    def test_repair_trailing_commas(self):
        raw = '{"items": [1, 2, 3,], "user": {"id": 1,},}'
        data = extract_json(raw)
        self.assertEqual(data["items"], [1, 2, 3])
        self.assertEqual(data["user"]["id"], 1)

    def test_repair_python_literals(self):
        raw = '{"active": True, "archived": False, "details": None}'
        data = extract_json(raw)
        self.assertTrue(data["active"])
        self.assertFalse(data["archived"])
        self.assertIsNone(data["details"])

    def test_repair_single_quotes(self):
        raw = "{'name': 'Bob', 'role': 'editor'}"
        data = extract_json(raw)
        self.assertEqual(data["name"], "Bob")
        self.assertEqual(data["role"], "editor")

    def test_repair_unquoted_keys(self):
        raw = '{name: "Charlie", score: 95}'
        data = extract_json(raw)
        self.assertEqual(data["name"], "Charlie")
        self.assertEqual(data["score"], 95)

    def test_repair_truncated_braces(self):
        raw = '{"users": [{"id": 1}, {"id": 2'
        data = extract_json(raw)
        self.assertEqual(len(data["users"]), 2)
        self.assertEqual(data["users"][0]["id"], 1)
        self.assertEqual(data["users"][1]["id"], 2)

    def test_extract_json_direct(self):
        raw = '{"valid": true}'
        self.assertEqual(extract_json(raw), {"valid": True})

    def test_schema_validation(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        # Valid
        validate_schema({"name": "David", "age": 30}, schema)

        # Missing field
        with self.assertRaises(ValueError) as ctx:
            validate_schema({"name": "David"}, schema)
        self.assertIn("age", str(ctx.exception))

    def test_chat_structured_retry_loop(self):
        mock_llm = MagicMock()

        # Turn 1: Malformed JSON that cannot be parsed
        # Turn 2: Valid JSON
        mock_llm.chat.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "I apologize, but this is not JSON at all: <<invalid>>"
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": '{"resolved": true, "attempt": 2}'}}]},
        ]

        messages = [{"role": "user", "content": "Return status"}]
        result = chat_structured(mock_llm, messages, max_retries=2)

        self.assertEqual(result["resolved"], True)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(mock_llm.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
