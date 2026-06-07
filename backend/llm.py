"""
Jarvis V1 — LLM Integration (Ollama + DeepSeek-R1)
=====================================================
Reusable wrapper for local LLM inference via Ollama.

Architecture decisions:
    - Uses the official `ollama` Python SDK for reliability
    - Streaming is the default mode (better UX for terminal)
    - DeepSeek-R1's <think>...</think> tokens are parsed and separated:
      * `thinking` content is stored for debugging/logging
      * `response` content is what the user sees
    - Health check validates Ollama connectivity before chat starts
    - Retry logic with exponential backoff for transient failures

The LLM is treated as a REASONING ENGINE ONLY.
It has no memory. Context and memory are injected externally by the
ContextBuilder (backend/context.py).
"""

import re
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import ollama

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Response data structure
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    
    Separates the clean user-facing response from internal thinking tokens,
    allowing the memory system to analyze thinking while keeping the UI clean.
    """
    content: str           # Clean response (thinking stripped)
    thinking: str = ""     # Raw <think>...</think> content (for debugging/storage)
    raw_content: str = ""  # Full unprocessed response
    model: str = ""
    total_duration_ms: int = 0
    token_count: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Think-token parser
# ---------------------------------------------------------------------------

# Regex to match <think>...</think> blocks (including multi-line)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking_tokens(raw_text: str) -> tuple[str, str]:
    """
    Separate <think>...</think> blocks from the clean response.
    
    Args:
        raw_text: Full LLM output potentially containing think tags.
    
    Returns:
        Tuple of (clean_response, thinking_content).
        - clean_response: Text with all <think> blocks removed and stripped.
        - thinking_content: Concatenated content from all <think> blocks.
    """
    thinking_parts = _THINK_PATTERN.findall(raw_text)
    thinking = "\n".join(part.strip() for part in thinking_parts if part.strip())

    # Remove all <think> blocks from the response
    clean = _THINK_PATTERN.sub("", raw_text).strip()

    return clean, thinking


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Wrapper around the Ollama API for local LLM inference.
    
    Provides:
    - Streaming and non-streaming chat
    - Health checking
    - Think-token parsing for DeepSeek-R1
    - Retry logic for transient failures
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self._model = settings.llm.model
        self._base_url = settings.llm.ollama_base_url
        self._timeout = settings.llm.request_timeout
        self._temperature = settings.llm.temperature
        self._max_retries = settings.llm.max_retries
        self._strip_thinking = settings.llm.strip_thinking_tokens

        # Initialize the Ollama client
        self._client = ollama.Client(host=self._base_url)

        logger.info(
            "OllamaClient initialized — model=%s, url=%s, strip_thinking=%s",
            self._model, self._base_url, self._strip_thinking,
        )

    def check_health(self) -> dict:
        """
        Verify Ollama server is reachable and the model is available.
        
        Returns:
            Dict with 'status' ('healthy'/'unhealthy'), 'model', and 'error' if any.
        """
        try:
            # List available models to verify connectivity
            models_response = self._client.list()
            available_models = [
                m.model for m in models_response.models  # pylint: disable=no-member
            ] if hasattr(models_response, 'models') else []

            model_available = any(
                self._model in m for m in available_models
            )

            if not model_available:
                logger.warning(
                    "Model '%s' not found. Available: %s",
                    self._model, available_models,
                )
                return {
                    "status": "unhealthy",
                    "model": self._model,
                    "error": f"Model '{self._model}' not found in Ollama",
                    "available_models": available_models,
                }

            logger.info("Ollama health check passed — model '%s' available", self._model)
            return {
                "status": "healthy",
                "model": self._model,
                "available_models": available_models,
            }

        except Exception as e:
            logger.error("Ollama health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "model": self._model,
                "error": str(e),
            }

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to the LLM (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles: 'system', 'user', 'assistant'.
            stream: If True, use streaming (use chat_stream instead for 
                    generator-based streaming).
        
        Returns:
            LLMResponse with clean content, thinking, and metadata.
        """
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "LLM request attempt %d/%d — %d messages",
                    attempt, self._max_retries, len(messages),
                )

                # Merge options
                req_options = {
                    "temperature": self._temperature,
                }
                if "options" in kwargs:
                    req_options.update(kwargs.pop("options"))

                response = self._client.chat(
                    model=self._model,
                    messages=messages,
                    options=req_options,
                    **kwargs
                )

                raw_content = response.message.content or ""  # pylint: disable=no-member
                clean_content, thinking = parse_thinking_tokens(raw_content)

                # Extract timing/token info if available
                total_duration = getattr(response, 'total_duration', 0) or 0
                prompt_tokens = getattr(response, 'prompt_eval_count', 0) or 0
                response_tokens = getattr(response, 'eval_count', 0) or 0
                total_tokens = prompt_tokens + response_tokens

                result = LLMResponse(
                    content=clean_content if self._strip_thinking else raw_content,
                    thinking=thinking,
                    raw_content=raw_content,
                    model=self._model,
                    total_duration_ms=total_duration // 1_000_000,  # ns to ms
                    token_count=response_tokens,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                )

                logger.info(
                    "LLM response — %d tokens, %dms",
                    result.token_count, result.total_duration_ms,
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt, self._max_retries, str(e),
                )
                if attempt < self._max_retries:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)

        # All retries exhausted
        logger.error("LLM request failed after %d attempts: %s", self._max_retries, last_error)
        raise ConnectionError(
            f"Failed to get LLM response after {self._max_retries} attempts: {last_error}"
        )

    def chat_stream(
        self,
        messages: list[dict],
    ) -> Generator[str, None, LLMResponse]:
        """
        Stream a chat response token-by-token.
        
        Yields clean text chunks (with thinking stripped) for real-time
        terminal display. Returns the full LLMResponse at the end.
        
        Usage:
            gen = client.chat_stream(messages)
            for chunk in gen:
                print(chunk, end="", flush=True)
            # After iteration, get the full response:
            # response = gen.value  (via StopIteration)
        
        Yields:
            String chunks of the clean response.
        
        Returns:
            LLMResponse with the complete response after streaming ends.
        """
        raw_parts: list[str] = []
        in_thinking = False
        thinking_parts: list[str] = []
        prompt_tokens = 0
        response_tokens = 0
        total_duration = 0

        try:
            stream = self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self._temperature,
                },
            )

            for chunk in stream:
                token = chunk.message.content or ""
                raw_parts.append(token)
                
                if getattr(chunk, "done", False):
                    prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                    response_tokens = getattr(chunk, "eval_count", 0) or 0
                    total_duration = getattr(chunk, "total_duration", 0) or 0

                if not self._strip_thinking:
                    yield token
                    continue

                # --- State machine for <think> tag stripping ---
                # We process the accumulated raw text to handle tags
                # that might be split across chunks
                accumulated = "".join(raw_parts)

                if not in_thinking:
                    # Check if we've entered a <think> block
                    think_start = accumulated.rfind("<think>")
                    if think_start != -1:
                        # Output everything before <think>
                        pre_think = accumulated[len("".join(raw_parts[:-1])):think_start]
                        if pre_think:
                            yield pre_think
                        in_thinking = True
                    else:
                        # Not in thinking — yield the token directly
                        # But be careful about partial "<think" at the end
                        if not any(
                            "<think>"[:i] == accumulated[-i:]
                            for i in range(1, min(8, len(accumulated) + 1))
                        ):
                            yield token
                else:
                    # Inside <think> block — check for closing tag
                    think_end = accumulated.rfind("</think>")
                    if think_end != -1:
                        # Extract thinking content
                        think_start = accumulated.rfind("<think>")
                        thinking_content = accumulated[think_start + 7:think_end]
                        thinking_parts.append(thinking_content.strip())
                        in_thinking = False
                        # Yield anything after </think>
                        after = accumulated[think_end + 8:]
                        if after:
                            yield after

        except Exception as e:
            logger.error("Streaming error: %s", str(e))
            raise

        # Build the final response
        raw_content = "".join(raw_parts)
        clean_content, thinking = parse_thinking_tokens(raw_content)

        return LLMResponse(
            content=clean_content,
            thinking=thinking,
            raw_content=raw_content,
            model=self._model,
            total_duration_ms=total_duration // 1_000_000,
            token_count=response_tokens,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=prompt_tokens + response_tokens,
        )

    def generate_summary(self, text: str, max_words: int = 50) -> str:
        """
        Use the LLM to generate a brief summary of text.
        
        Used by the memory extractor for conversation summarization.
        
        Args:
            text: Text to summarize.
            max_words: Target summary length.
        
        Returns:
            Summary string.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"Summarize the following text in {max_words} words or fewer. "
                    "Be concise and capture only the key points. "
                    "Output only the summary, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ]

        response = self.chat(messages)
        return response.content
