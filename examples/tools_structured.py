"""Modern tool calling + structured output. Runs fully offline.

Demonstrates `@tool` + `tools`/`tool_choice` with fallback to legacy
`add_function`/`functions`, and `response_format` with fallback to
`format="json"`.

Usage:
    python examples/tools_structured.py
    python examples/tools_structured.py --backend ollama --model llama3.2:3b
"""
import argparse
import json

from local_llm_kit import LLM, tool


@tool(description="Get the current weather for a city")
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the weather for a location."""
    return {"temperature": 22, "unit": unit, "location": location}


CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "temperature": {"type": "integer"},
        "unit": {"type": "string"},
    },
    "required": ["name", "temperature", "unit"],
}


def chat_with_tools(llm: LLM):
    kwargs = dict(
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[get_weather.spec],  # type: ignore[attr-defined]
        tool_choice="auto",
    )
    try:
        return llm.chat(**kwargs)
    except TypeError:
        # Older client: legacy functions/function_call API
        legacy_fn = get_weather.spec["function"]  # type: ignore[attr-defined]
        return llm.chat(
            messages=kwargs["messages"],
            functions=[legacy_fn],
            function_call="auto",
        )


def chat_structured(llm: LLM):
    messages = [{"role": "user",
                 "content": "Return Paris weather as JSON with keys name/temperature/unit."}]
    try:
        return llm.chat(
            messages=messages,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "city", "schema": CITY_SCHEMA}},
        )
    except TypeError:
        return llm.chat(messages=messages, format="json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tools + structured output demo")
    ap.add_argument("--model", "-m", default="echo", help="Model path/name")
    ap.add_argument("--backend", "-b", default="echo", help="Backend (default: echo)")
    args = ap.parse_args()

    llm = LLM(model_path=args.model, backend=args.backend)
    try:
        llm.add_tool(get_weather)
    except AttributeError:  # very old client without add_tool
        llm.add_function(
            name="get_weather",
            schema=get_weather.spec["function"],  # type: ignore[attr-defined]
            implementation=get_weather,
        )

    print("== tool call ==")
    print(json.dumps(chat_with_tools(llm), indent=2, default=str)[:2000])
    print("== structured ==")
    resp = chat_structured(llm)
    print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
