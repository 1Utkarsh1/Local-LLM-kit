"""
Backend for llama.cpp models via llama-cpp-python.
"""

import json
import logging
from typing import Any, Dict, Iterator, Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama, LlamaGrammar

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning(
        "llama-cpp-python package not available. Install with 'pip install llama-cpp-python'"
    )
    LLAMA_CPP_AVAILABLE = False


class LlamaCppBackend(BaseBackend):
    """
    Backend for llama.cpp models via llama-cpp-python binding.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_batch: int = 512,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the llama.cpp backend.

        Args:
            model_path: Path to the model file (.gguf)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size
            n_batch: Batch size for prompt evaluation
            verbose: Whether to enable verbose output
            **kwargs: Additional arguments to pass to Llama constructor
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python package is not installed. "
                "Install it with 'pip install llama-cpp-python'"
            )

        self.model_path = model_path
        self.n_ctx = n_ctx

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text completion for a prompt."""
        if stream:
            # If stream is requested, use generate_stream and collect all chunks
            chunks = list(
                self.generate_stream(
                    prompt=prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    **kwargs,
                )
            )

            # Combine chunks
            text = "".join(chunk["text"] for chunk in chunks)

            # Combine logprobs if requested
            combined_logprobs = None
            if logprobs and chunks and "logprobs" in chunks[0]:
                combined_logprobs = {"tokens": [], "token_logprobs": [], "top_logprobs": []}

                for chunk in chunks:
                    if "logprobs" in chunk:
                        chunk_logprobs = chunk["logprobs"]
                        combined_logprobs["tokens"].extend(chunk_logprobs.get("tokens", []))
                        combined_logprobs["token_logprobs"].extend(
                            chunk_logprobs.get("token_logprobs", [])
                        )

                        if "top_logprobs" in chunk_logprobs:
                            combined_logprobs["top_logprobs"].extend(
                                chunk_logprobs.get("top_logprobs", [])
                            )

            result = {"text": text}
            if combined_logprobs:
                result["logprobs"] = combined_logprobs

            return result

        # Setup generation parameters
        params = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "logprobs": top_logprobs if logprobs else None,
        }

        # Add grammar if provided
        if "grammar" in kwargs:
            grammar = kwargs.pop("grammar")
            if isinstance(grammar, str):
                params["grammar"] = LlamaGrammar.from_string(grammar)
            elif isinstance(grammar, dict):
                params["grammar"] = LlamaGrammar.from_string(json.dumps(grammar))

        # Generate
        output = self.model(prompt=prompt, **params, **kwargs)

        # Extract text and tokens
        if isinstance(output, dict):
            text = output.get("choices", [{}])[0].get("text", "")
        else:
            # Handle list output format
            text = output[0]["choices"][0]["text"]

        result = {"text": text}

        # Add logprobs if requested
        if logprobs and top_logprobs:
            try:
                tokens_with_logprobs = []

                if isinstance(output, dict) and "choices" in output:
                    choice = output["choices"][0]
                    if "logprobs" in choice:
                        tokens_with_logprobs = choice["logprobs"]["tokens"]
                        token_logprobs = choice["logprobs"]["token_logprobs"]
                        top_logprobs_data = choice["logprobs"]["top_logprobs"]
                elif isinstance(output, list):
                    choice = output[0]["choices"][0]
                    if "logprobs" in choice:
                        tokens_with_logprobs = choice["logprobs"]["tokens"]
                        token_logprobs = choice["logprobs"]["token_logprobs"]
                        top_logprobs_data = choice["logprobs"]["top_logprobs"]

                if tokens_with_logprobs:
                    result["logprobs"] = {
                        "tokens": tokens_with_logprobs,
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs_data,
                    }
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Failed to extract logprobs: {e}")

        return result

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """Stream text completion for a prompt."""
        # Setup generation parameters
        params = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "logprobs": top_logprobs if logprobs else None,
            "stream": True,
        }

        # Add grammar if provided
        if "grammar" in kwargs:
            grammar = kwargs.pop("grammar")
            if isinstance(grammar, str):
                params["grammar"] = LlamaGrammar.from_string(grammar)
            elif isinstance(grammar, dict):
                params["grammar"] = LlamaGrammar.from_string(json.dumps(grammar))

        # Generate with streaming
        for chunk in self.model(prompt=prompt, **params, **kwargs):
            # Extract delta text from chunk
            if "choices" in chunk and chunk["choices"]:
                delta_text = chunk["choices"][0].get("text", "")

                # Build chunk result
                result = {"text": delta_text}

                # Add logprobs if available
                if logprobs and top_logprobs:
                    try:
                        if "logprobs" in chunk["choices"][0]:
                            logprobs_data = chunk["choices"][0]["logprobs"]
                            result["logprobs"] = {
                                "tokens": logprobs_data.get("tokens", []),
                                "token_logprobs": logprobs_data.get("token_logprobs", []),
                                "top_logprobs": logprobs_data.get("top_logprobs", []),
                            }
                    except (KeyError, IndexError, TypeError):
                        pass

                yield result

    def get_context_window(self) -> int:
        """Get the context window size of the model."""
        return self.n_ctx

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.model.tokenize(text.encode("utf-8")))


# Function to create JSON grammar for structured outputs
def create_json_grammar() -> str:
    """
    Create a grammar for JSON generation.

    Returns:
        Grammar string in BNF format
    """
    grammar = """
        root ::= object
        
        value ::= object | array | string | number | ("true" | "false" | "null")
        
        object ::= "{" ws (string_colon_value (comma_string_colon_value)*)? ws "}"
        
        string_colon_value ::= ws string ws ":" ws value
        
        comma_string_colon_value ::= ws "," ws string ws ":" ws value
        
        array ::= "[" ws (value (comma_value)*)? ws "]"
        
        comma_value ::= ws "," ws value
        
        string ::= "\"" (
            [^"\\] |
            "\\\\" |
            "\\\"" |
            "\\/" |
            "\\b" |
            "\\f" |
            "\\n" |
            "\\r" |
            "\\t" |
            "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
        )* "\""
        
        number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        
        ws ::= [ \\t\\n\\r]*
    """
    return grammar
