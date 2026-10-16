"""Optional AirLLM adapter. Imports and model loading happen on construction.

Generation uses AirLLM's PyTorch API or its macOS MLX API. Streaming is
buffered (one completed chunk); GPU/model compatibility needs hardware
validation. This adapter does not download weights until explicitly selected.
"""

import threading
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseBackend


class AirLLMBackend(BaseBackend):
    """AirLLM text inference using local or cached trusted model checkpoints."""

    backend_name = "airllm"

    def __init__(
        self, model_path: str, device: Optional[str] = None, max_seq_len: int = 2048, **kwargs: Any
    ):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string.")
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer.")
        try:
            from airllm import AutoModel
        except ImportError as exc:
            raise ImportError(
                "AirLLM needs its optional dependencies: pip install 'local-llm-kit[airllm]'."
            ) from exc
        options = dict(kwargs, max_seq_len=max_seq_len)
        if device is not None:
            options["device"] = device
        self.model = AutoModel.from_pretrained(model_path, **options)
        self.tokenizer = self.model.tokenizer
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self._mlx = type(self.model).__module__.endswith("airllm_llama_mlx")
        self.device = getattr(self.model, "device", device or "cuda:0")
        # Offloaded layers are stateful; cached backends cannot generate concurrently.
        self._lock = threading.Lock()

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
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if logprobs or kwargs.get("seed") is not None:
            raise NotImplementedError(
                "The AirLLM adapter does not support logprobs or per-call seeds."
            )
        stops = [stop] if isinstance(stop, str) else stop or []
        if not isinstance(stops, list) or any(
            not isinstance(item, str) or not item for item in stops
        ):
            raise ValueError("stop must be a non-empty string or list of non-empty strings.")
        if max_new_tokens == 0:
            return {"text": "", "finish_reason": "length"}
        with self._lock:
            inputs = self.tokenizer(
                prompt,
                return_tensors="np" if self._mlx else "pt",
                return_attention_mask=False,
                truncation=False,
            )
            input_ids = inputs["input_ids"]
            prompt_length = len(input_ids[0])
            if prompt_length + max_new_tokens > self.max_seq_len:
                raise ValueError(
                    "Prompt plus requested tokens exceeds max_seq_len; increase it explicitly."
                )
            if self._mlx:
                import mlx.core as mx

                output = self.model.generate(
                    mx.array(input_ids), temperature=temperature, max_new_tokens=max_new_tokens
                )
                text = output
                generated_length = None
            else:
                output = self.model.generate(
                    input_ids.to(self.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                    return_dict_in_generate=True,
                    **{key: value for key, value in kwargs.items() if key != "seed"},
                )
                sequences = getattr(output, "sequences", output)
                generated = sequences[0][prompt_length:]
                generated_length = len(generated)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not isinstance(text, str):
            raise RuntimeError("AirLLM returned an unsupported output shape.")
        cutoff = min((text.find(item) for item in stops if item in text), default=len(text))
        finish = "length" if generated_length == max_new_tokens else "stop"
        return {"text": text[:cutoff], "finish_reason": "stop" if cutoff < len(text) else finish}

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        # shortcut: AirLLM streaming is buffered; upgrade when both upstream APIs expose stable token callbacks.
        yield self.generate(prompt, **kwargs)

    def get_context_window(self) -> int:
        return self.max_seq_len

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
