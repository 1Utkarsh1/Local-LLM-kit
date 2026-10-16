#!/usr/bin/env python3
"""Example: Resilient Structured Outputs and Zero-Dependency JSON Auto-Repair.

Demonstrates:
1. Automated repair of corrupted, malformed, or conversational local model outputs.
2. Self-correcting structured chat loop with JSON schema validation.

Run:
    python examples/structured_json.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_llm_kit import extract_json, repair_json


def main():
    print("=== Local LLM Kit: Resilient Structured Output & Auto-Repair ===\n")

    # Sample malformed output commonly produced by small local LLMs:
    # - Markdown code fences
    # - Conversational preface and postscript
    # - Trailing commas
    # - Single-quoted strings
    # - Python literals (True, None)
    # - Truncated braces
    malformed_output = """
    Certainly! Here is the JSON representation you requested:
    ```json
    {
        'status': 'success',
        'code': 200,
        'is_active': True,
        'metadata': None,
        'tags': ['local-ai', 'agent', 'mcp',],
        'metrics': {
            'latency_ms': 12.5,
    ```
    I hope this helps! Reach out if you need further adjustments.
    """

    print("Raw Corrupted LLM Output:")
    print(malformed_output.strip())

    print("\nRepairing JSON via repair_json()...")
    cleaned = repair_json(malformed_output)
    print("Cleaned JSON String:")
    print(cleaned)

    print("\nParsed Data Object via extract_json():")
    data = extract_json(malformed_output)
    print(data)
    print(f"Data type: {type(data).__name__}")
    print(
        f"Status: {data.get('status')}, Active: {data.get('is_active')}, Tags: {data.get('tags')}"
    )


if __name__ == "__main__":
    main()
