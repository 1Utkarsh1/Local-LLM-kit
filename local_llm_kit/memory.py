"""
Memory and message history management for chat conversations.
"""
import json
from typing import List, Dict, Any, Optional


class MessageHistory:
    """
    Manages message history for multi-turn conversations.
    
    Features:
    - Maintains message history
    - Auto-truncates messages to stay within context window
    - Basic summarization (stub) for long histories
    """
    
    def __init__(self, context_window: int = 4096, token_estimator=None):
        """
        Initialize message history manager.
        
        Args:
            context_window: Maximum context window size in tokens
            token_estimator: Function to estimate token count (defaults to character-based estimate)
        """
        self.messages: List[Dict[str, Any]] = []
        self.context_window = context_window
        self.token_estimator = token_estimator or self._estimate_tokens
        
    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add messages to history.
        
        Args:
            messages: List of message dictionaries to add
        """
        self.messages.extend(messages)
        self._truncate_if_needed()
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a single message to history.
        
        Args:
            message: Message dictionary to add
        """
        self.messages.append(message)
        self._truncate_if_needed()
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get current message history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear message history."""
        self.messages = []
        
    def _truncate_if_needed(self) -> None:
        """Truncate message history if it exceeds the context window."""
        total_tokens = self._count_tokens(self.messages)
        
        if total_tokens <= self.context_window:
            return
        
        # Keep removing oldest messages until we're under the token limit
        while total_tokens > self.context_window and len(self.messages) > 2:
            # Always keep the system message if present
            start_idx = 1 if self.messages and self.messages[0].get("role") == "system" else 0
            
            # Remove the oldest non-system message
            if start_idx < len(self.messages):
                removed_msg = self.messages.pop(start_idx)
                total_tokens -= self._count_message_tokens(removed_msg)
        
        # If still over limit, add a summary message and remove more
        if total_tokens > self.context_window:
            self._summarize_history()
    
    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Estimated token count
        """
        return sum(self._count_message_tokens(msg) for msg in messages)
    
    def _count_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Count tokens in a single message.
        
        Args:
            message: Message dictionary
            
        Returns:
            Estimated token count
        """
        # Extract all text content from the message
        text_parts = []
        
        # Add role
        text_parts.append(message.get("role", ""))
        
        # Add content if present (plain string or vision-style part list)
        content = message.get("content")
        if isinstance(content, str):
            if content:
                text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type", "text") == "text":
                        text = part.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif part.get("type") in ("image_url", "image"):
                        text_parts.append("[image]")
                    elif part.get("text"):
                        text_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    text_parts.append(part)
        
        # Add function call if present
        if message.get("function_call"):
            func_call = message["function_call"]
            text_parts.append(func_call.get("name", ""))
            if isinstance(func_call.get("arguments"), str):
                text_parts.append(func_call["arguments"])
            elif func_call.get("arguments"):
                text_parts.append(json.dumps(func_call["arguments"]))
        
        # Add function name if present
        if message.get("name"):
            text_parts.append(message["name"])
        
        # Combine all text parts and estimate tokens
        combined_text = " ".join(text_parts)
        return self.token_estimator(combined_text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count based on characters (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 chars per token for English text
        return max(1, len(text) // 4)
    
    def _summarize_history(self) -> None:
        """
        Create a summary of older messages to reduce token usage.
        
        Note: This is a basic implementation that simply keeps the most
        recent messages and replaces older ones with a summary message.
        """
        # Keep system message if present
        system_message = None
        if self.messages and self.messages[0].get("role") == "system":
            system_message = self.messages[0]
            remaining_messages = self.messages[1:]
        else:
            remaining_messages = self.messages
        
        # Keep the most recent messages (last 4 turns = 8 messages)
        keep_count = min(8, max(2, len(remaining_messages) // 2))
        recent_messages = remaining_messages[-keep_count:] if keep_count > 0 else []
        
        # Create a simple summary message
        summary_message = {
            "role": "system",
            "content": f"[Previous conversation history summarized: {len(remaining_messages) - keep_count} messages omitted]"
        }
        
        # Rebuild message list
        new_messages = []
        if system_message:
            new_messages.append(system_message)
        new_messages.append(summary_message)
        new_messages.extend(recent_messages)
        
        self.messages = new_messages 