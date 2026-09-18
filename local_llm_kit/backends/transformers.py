"""
Hugging Face Transformers backend for model inference.
"""
import logging
import queue
import threading
from typing import Dict, Any, Iterator, Optional, List, Union, Tuple

from .base import BaseBackend

logger = logging.getLogger(__name__)

try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        TextIteratorStreamer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers package not available. Install with 'pip install transformers'")
    TRANSFORMERS_AVAILABLE = False


class TransformersBackend(BaseBackend):
    """
    Backend for Hugging Face Transformers models.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        **kwargs
    ):
        """
        Initialize the Transformers backend.
        
        Args:
            model_path: Path or name of the model to load
            device: Device to load the model on ("cpu", "cuda", "mps", etc.)
            torch_dtype: Torch data type to use
            low_cpu_mem_usage: Whether to use low CPU memory usage when loading
            **kwargs: Additional arguments to pass to model loading
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers package is not installed. "
                "Install it with 'pip install transformers'"
            )
        
        self.model_path = model_path
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Determine torch dtype
        if torch_dtype:
            if torch_dtype == "float16":
                dtype = torch.float16
            elif torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif torch_dtype == "float32":
                dtype = torch.float32
            else:
                raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        elif device == "cuda":
            # Use half precision by default on CUDA
            dtype = torch.float16
        else:
            # Use full precision on CPU
            dtype = torch.float32
            
        self.torch_dtype = dtype
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="left",
            **kwargs
        )
        
        # Enable padding for batched inference
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        logger.info(f"Loading model from {model_path} to {device} with dtype {dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device if device != "cpu" else None,
            **kwargs
        )
        
        # Move model to device if using CPU
        if device == "cpu":
            self.model = self.model.to(device)
        
        # Set stop tokens for generation
        self.stop_token_ids = []
        
        # Try to find stop tokens from tokenizer
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)
            
        # Add common stop tokens found in different models
        common_stop_tokens = ["<|endoftext|>", "</s>", "<|im_end|>"]
        for token in common_stop_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.append(token_id)
                
        # Deduplicate
        self.stop_token_ids = list(set(self.stop_token_ids))
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion for a prompt."""
        if stream:
            # If stream is requested, use generate_stream and collect all chunks
            chunks = list(self.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                **kwargs
            ))
            
            # Combine chunks
            text = "".join(chunk["text"] for chunk in chunks)
            
            # Combine logprobs if needed
            combined_logprobs = None
            if logprobs and chunks and "logprobs" in chunks[0]:
                combined_logprobs = {
                    "tokens": [],
                    "token_logprobs": [],
                    "top_logprobs": []
                }
                
                for chunk in chunks:
                    if "logprobs" in chunk:
                        chunk_logprobs = chunk["logprobs"]
                        combined_logprobs["tokens"].extend(chunk_logprobs.get("tokens", []))
                        combined_logprobs["token_logprobs"].extend(chunk_logprobs.get("token_logprobs", []))
                        
                        if "top_logprobs" in chunk_logprobs:
                            combined_logprobs["top_logprobs"].extend(chunk_logprobs.get("top_logprobs", []))
            
            result = {"text": text}
            if combined_logprobs:
                result["logprobs"] = combined_logprobs
                
            return result
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Prepare generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),  # Avoid division by zero
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        # Add stopping criteria based on stop tokens
        if self.stop_token_ids:
            stopping_criteria = StoppingCriteriaList([
                _StopOnTokens(self.stop_token_ids)
            ])
            gen_kwargs["stopping_criteria"] = stopping_criteria
            
        # Generate
        with torch.no_grad():
            # Generate output
            output = self.model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=logprobs,  # Only get scores if logprobs requested
                **gen_kwargs
            )
            
            # Extract generated text
            generated_ids = output.sequences[0][len(input_ids[0]):]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        result = {"text": text}
        
        # Add logprobs if requested
        if logprobs and hasattr(output, "scores"):
            result["logprobs"] = self._extract_logprobs(
                output.sequences[0][len(input_ids[0]):], 
                output.scores, 
                top_logprobs
            )
            
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
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Stream text completion for a prompt."""
        # Check if we can use TextIteratorStreamer
        if not hasattr(transformers, "TextIteratorStreamer"):
            raise ImportError(
                "TextIteratorStreamer not available in your transformers version. "
                "Please upgrade to a newer version."
            )
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            timeout=10.0
        )
        
        # Prepare generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),  # Avoid division by zero
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        
        # Add stopping criteria based on stop tokens
        if self.stop_token_ids:
            stopping_criteria = StoppingCriteriaList([
                _StopOnTokens(self.stop_token_ids)
            ])
            gen_kwargs["stopping_criteria"] = stopping_criteria
        
        # Setup for logprob calculation if needed
        scores_queue = None
        if logprobs:
            scores_queue = queue.Queue()
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
            
            # Override the streamer's put method to capture scores
            original_put = streamer.put
            
            def put_with_scores(value):
                if isinstance(value, dict) and "scores" in value:
                    scores_queue.put(value["scores"])
                return original_put(value)
            
            streamer.put = put_with_scores
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=self._generate_in_thread,
            args=(self.model, input_ids, gen_kwargs)
        )
        generation_thread.start()
        
        # Track tokens for logprobs if needed
        token_ids = []
        scores_list = []
        
        # Stream output tokens
        last_text = ""
        for new_text in streamer:
            # Extract delta text (only the new part)
            delta_text = new_text[len(last_text):]
            last_text = new_text
            
            # Get token ids for this chunk
            if logprobs:
                try:
                    if not scores_queue.empty():
                        scores = scores_queue.get_nowait()
                        scores_list.append(scores)
                        
                        # Tokenize the delta text to get token ids
                        delta_token_ids = self.tokenizer.encode(delta_text)
                        token_ids.extend(delta_token_ids)
                except queue.Empty:
                    pass
            
            # Build chunk result
            chunk = {"text": delta_text}
            
            # Add logprobs if available
            if logprobs and len(token_ids) > 0 and len(scores_list) > 0:
                chunk_logprobs = self._calculate_chunk_logprobs(
                    token_ids[-1:],  # Use only the last token
                    scores_list[-1:],  # Use only the last score
                    top_logprobs
                )
                if chunk_logprobs:
                    chunk["logprobs"] = chunk_logprobs
            
            yield chunk
    
    def _generate_in_thread(self, model, input_ids, gen_kwargs):
        """Helper method to run model generation in a separate thread."""
        try:
            with torch.no_grad():
                model.generate(input_ids, **gen_kwargs)
        except Exception as e:
            logger.error(f"Error in generation thread: {e}")
            # Try to put error in streamer
            streamer = gen_kwargs.get("streamer")
            if streamer:
                try:
                    streamer.put(f"[ERROR: {e}]")
                    streamer.end()
                except:
                    pass
    
    def _extract_logprobs(
        self,
        token_ids: torch.Tensor,
        scores: Tuple[torch.Tensor],
        top_n: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Extract log probabilities from model outputs.
        
        Args:
            token_ids: Generated token IDs
            scores: Model output scores
            top_n: Number of top tokens to include
            
        Returns:
            Dictionary with token logprobs
        """
        if not scores:
            return {}
            
        top_n = top_n or 5
        token_ids_list = token_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids_list)
        
        # Calculate logprobs and find top tokens
        all_token_logprobs = []
        all_top_logprobs = []
        
        for i, (token_id, score) in enumerate(zip(token_ids_list, scores)):
            # Get full logprobs
            logits = score[0]  # Shape: [vocab_size]
            logprobs = torch.log_softmax(logits, dim=-1)
            
            # Get logprob for the chosen token
            token_logprob = logprobs[token_id].item()
            all_token_logprobs.append(token_logprob)
            
            # Get top tokens
            if top_n > 0:
                top_values, top_indices = torch.topk(logprobs, k=top_n)
                top_dict = {}
                
                for j, (value, idx) in enumerate(zip(top_values, top_indices)):
                    top_token = self.tokenizer.convert_ids_to_tokens(idx.item())
                    top_dict[top_token] = value.item()
                
                all_top_logprobs.append(top_dict)
        
        return {
            "tokens": tokens,
            "token_logprobs": all_token_logprobs,
            "top_logprobs": all_top_logprobs if top_n > 0 else None
        }
    
    def _calculate_chunk_logprobs(
        self,
        token_ids: List[int],
        scores: List[torch.Tensor],
        top_n: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Calculate log probabilities for a chunk of tokens.
        
        Args:
            token_ids: List of token IDs
            scores: List of model output scores
            top_n: Number of top tokens to include
            
        Returns:
            Dictionary with token logprobs for this chunk
        """
        if not scores or not token_ids:
            return {}
            
        top_n = top_n or 5
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Calculate logprobs and find top tokens
        all_token_logprobs = []
        all_top_logprobs = []
        
        for i, (token_id, score) in enumerate(zip(token_ids, scores)):
            # Get full logprobs
            logits = score[0]  # Shape: [vocab_size]
            logprobs = torch.log_softmax(logits, dim=-1)
            
            # Get logprob for the chosen token
            token_logprob = logprobs[token_id].item()
            all_token_logprobs.append(token_logprob)
            
            # Get top tokens
            if top_n > 0:
                top_values, top_indices = torch.topk(logprobs, k=top_n)
                top_dict = {}
                
                for j, (value, idx) in enumerate(zip(top_values, top_indices)):
                    top_token = self.tokenizer.convert_ids_to_tokens(idx.item())
                    top_dict[top_token] = value.item()
                
                all_top_logprobs.append(top_dict)
        
        return {
            "tokens": tokens,
            "token_logprobs": all_token_logprobs,
            "top_logprobs": all_top_logprobs if top_n > 0 else None
        }
    
    def get_context_window(self) -> int:
        """Get the context window size of the model."""
        # Try to get from config
        if hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        
        # Try to get from model attributes
        if hasattr(self.model, "max_seq_len"):
            return self.model.max_seq_len
        
        # Try common model families
        model_type = getattr(self.model.config, "model_type", "").lower()
        
        if "llama" in model_type:
            return 4096  # Most LLaMA models
        elif "mistral" in model_type:
            return 8192  # Mistral models
        elif "gpt-2" in model_type:
            return 1024  # GPT-2
        elif "gpt-j" in model_type:
            return 2048  # GPT-J
        elif "gpt-neo" in model_type:
            return 2048  # GPT-Neo
        elif "bloom" in model_type:
            return 2048  # BLOOM
        
        # Fallback to a reasonable default
        logger.warning(
            f"Could not determine context window for model {self.model_path}. "
            "Using default of 2048."
        )
        return 2048
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))


if TRANSFORMERS_AVAILABLE:
    _StoppingCriteriaBase = StoppingCriteria
else:  # pragma: no cover - import-time fallback so the module stays importable
    _StoppingCriteriaBase = object


class _StopOnTokens(_StoppingCriteriaBase):  # type: ignore[valid-type, misc]
    """Criteria to stop generation on specific token IDs."""
    
    def __init__(self, stop_token_ids: List[int]):
        """Initialize with the token IDs to stop on."""
        self.stop_token_ids = stop_token_ids
        
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Return True if generation should stop."""
        if not self.stop_token_ids:
            return False
            
        # Check if the last token is in stop_token_ids
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
                
        return False 