"""
Simple example script for local_llm_kit chat.
"""

import argparse
import sys

from local_llm_kit import chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple chat with a local LLM")
    parser.add_argument("--model", "-m", required=True, help="Path to the model or model name")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["transformers", "llamacpp"],
        help="Backend to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Initial message
    print(f"Simple chat with {args.model}")
    print("Type 'exit' or 'quit' to end the conversation.")
    print()

    # Chat loop
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    try:
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate and display response
            try:
                print("Assistant: ", end="", flush=True)
                response_text = ""

                # Stream the response token by token
                for chunk in chat(
                    messages=messages, model_path=args.model, backend=args.backend, stream=True
                ):
                    # Extract and print chunk text
                    if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            print(delta["content"], end="", flush=True)
                            response_text += delta["content"]

                print()  # Add newline after response

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                print(f"\nError: {e}")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
