"""
Jarvis Phase 3 — LM Studio Integration
========================================
Local LLM inference via LM Studio's OpenAI-compatible API.
"""

import os
import time
from typing import Generator, Optional

from openai import OpenAI

from backend.logger import get_logger
from backend.llm import LLMResponse, parse_thinking_tokens
from configs.settings import get_settings

logger = get_logger(__name__)


def _is_model_reloaded_error(error: Exception) -> bool:
    """Detect LM Studio model-swap / reload errors that need a longer wait."""
    msg = str(error).lower()
    return "model reloaded" in msg or "model is currently loading" in msg


def _is_fatal_model_error(error: Exception) -> bool:
    """Errors that should not be retried — fallback immediately."""
    msg = str(error).lower()
    return any(
        token in msg
        for token in (
            "crashed",
            "model_not_found",
            "invalid model",
            "not found",
            "does not exist",
        )
    )


class LMStudioClient:
    """
    Wrapper around LM Studio's OpenAI-compatible API.
    Mirrors the interface of OllamaClient so the ModelRouter can swap them easily.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self._base_url = settings.providers.lm_studio.base_url
        self._model = settings.providers.lm_studio.model
        self._timeout = settings.providers.lm_studio.timeout
        self._strip_thinking = settings.llm.strip_thinking_tokens
        self._temperature = settings.llm.temperature
        self._max_retries = settings.llm.max_retries

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="lm-studio",
            timeout=self._timeout,
        )

        logger.info(
            "LMStudioClient initialized — model=%s, url=%s, timeout=%ds",
            self._model, self._base_url, self._timeout,
        )

    def check_health(self) -> dict:
        """Verify LM Studio server is reachable and get available models."""
        try:
            models_response = self._client.models.list()
            available_models = [m.id for m in models_response.data]

            logger.info("LM Studio health check passed (%d models)", len(available_models))
            return {
                "status": "healthy",
                "model": self._model,
                "available_models": available_models,
            }

        except Exception as e:
            logger.warning("LM Studio health check failed: %s", e)
            return {
                "status": "unhealthy",
                "model": self._model,
                "error": str(e),
                "available_models": [],
            }

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        """Exponential backoff; longer wait when LM Studio is reloading a model."""
        if _is_model_reloaded_error(error):
            return min(15.0, 5.0 * attempt)
        return float(2 ** attempt)

    def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Synchronous chat request with retries."""
        start_time = time.time()
        last_error = None
        max_retries = kwargs.pop("max_retries", self._max_retries)

        req_temperature = kwargs.get("temperature", self._temperature)
        req_max_tokens = kwargs.get("max_tokens", 2048)
        req_timeout = kwargs.pop("timeout", None)

        for attempt in range(1, max_retries + 1):
            try:
                mapped_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ]

                create_kwargs = {
                    "model": self._model,
                    "messages": mapped_messages,
                    "temperature": req_temperature,
                    "max_tokens": req_max_tokens,
                }
                if req_timeout is not None:
                    create_kwargs["timeout"] = req_timeout

                response = self._client.chat.completions.create(**create_kwargs)

                raw_content = response.choices[0].message.content
                clean_content, thinking = parse_thinking_tokens(raw_content)
                final_content = clean_content if self._strip_thinking else raw_content

                duration_ms = int((time.time() - start_time) * 1000)
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = (
                    response.usage.completion_tokens if response.usage else len(raw_content) // 4
                )
                total_tokens = (
                    response.usage.total_tokens if response.usage else prompt_tokens + response_tokens
                )

                return LLMResponse(
                    content=final_content,
                    thinking=thinking,
                    raw_content=raw_content,
                    model=self._model,
                    total_duration_ms=duration_ms,
                    token_count=response_tokens,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "LM Studio chat failed (attempt %d/%d): %s", attempt, max_retries, e
                )
                if _is_fatal_model_error(e):
                    logger.error("LM Studio fatal model error — skipping retries")
                    break
                if attempt < max_retries:
                    delay = self._retry_delay(attempt, e)
                    if _is_model_reloaded_error(e):
                        logger.info("LM Studio model reloading — waiting %.1fs", delay)
                    time.sleep(delay)

        logger.error(
            "LM Studio chat request failed after %d attempts: %s", max_retries, last_error
        )
        raise ConnectionError(
            f"Failed to get LLM response after {max_retries} attempts: {last_error}"
        )

    def chat_stream(self, messages: list[dict], **kwargs) -> Generator[tuple[str, str, LLMResponse], None, None]:
        """
        Stream chat response with retries on transient failures.
        Yields (clean_chunk, thinking_chunk, None) during generation,
        and (final_clean, final_thinking, LLMResponse) at completion.
        """
        start_time = time.time()
        req_temperature = kwargs.get("temperature", self._temperature)
        req_max_tokens = kwargs.get("max_tokens", 2048)
        max_retries = self._max_retries
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                mapped_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ]

                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=mapped_messages,
                    temperature=req_temperature,
                    max_tokens=req_max_tokens,
                    stream=True,
                )

                full_content = ""
                in_think_block = False

                for chunk in stream:
                    if len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    full_content += delta

                    clean_chunk = delta
                    think_chunk = ""

                    if "<think>" in delta:
                        in_think_block = True
                        parts = delta.split("<think>")
                        clean_chunk = parts[0]
                        think_chunk = parts[1] if len(parts) > 1 else ""
                    elif "</think>" in delta:
                        in_think_block = False
                        parts = delta.split("</think>")
                        think_chunk = parts[0]
                        clean_chunk = parts[1] if len(parts) > 1 else ""
                    else:
                        if in_think_block:
                            think_chunk = delta
                            clean_chunk = ""

                    if self._strip_thinking and in_think_block:
                        clean_chunk = ""

                    yield clean_chunk, think_chunk, None

                duration_ms = int((time.time() - start_time) * 1000)
                clean, thinking = parse_thinking_tokens(full_content)
                response_tokens = len(full_content) // 4
                final_resp = LLMResponse(
                    content=clean if self._strip_thinking else full_content,
                    thinking=thinking,
                    raw_content=full_content,
                    model=self._model,
                    total_duration_ms=duration_ms,
                    token_count=response_tokens,
                    prompt_tokens=0,
                    response_tokens=response_tokens,
                    total_tokens=response_tokens,
                )
                yield "", "", final_resp
                return

            except Exception as e:
                last_error = e
                logger.warning(
                    "LM Studio stream failed (attempt %d/%d): %s", attempt, max_retries, e
                )
                if attempt < max_retries:
                    time.sleep(self._retry_delay(attempt, e))

        logger.error("LM Studio stream request failed after %d attempts: %s", max_retries, last_error)
        raise ConnectionError(
            f"Failed to stream LLM response after {max_retries} attempts: {last_error}"
        )
