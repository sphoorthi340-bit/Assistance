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

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="lm-studio"  # LM Studio requires a placeholder key
        )

        logger.info(
            "LMStudioClient initialized — model=%s, url=%s",
            self._model, self._base_url
        )

    def check_health(self) -> dict:
        """Verify LM Studio server is reachable and get available models."""
        try:
            models_response = self._client.models.list()
            available_models = [m.id for m in models_response.data]

            if not self._model or self._model not in available_models:
                # If configured model isn't there, just pick the first one as default if we need to
                pass

            logger.info("LM Studio health check passed")
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

    def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """
        Synchronous chat request.
        """
        start_time = time.time()
        
        last_error = None
        max_retries = getattr(self, '_max_retries', 3)
        
        # Override defaults with kwargs if provided
        req_temperature = kwargs.get("temperature", self._temperature)
        req_max_tokens = kwargs.get("max_tokens", 2048)  # Prevent infinite KV cache allocation crash
        
        for attempt in range(1, max_retries + 1):
            try:
                # Map messages to strictly what OpenAI supports
                mapped_messages = []
                for msg in messages:
                    # OpenAI format matches standard role/content
                    mapped_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=mapped_messages,
                    temperature=req_temperature,
                    max_tokens=req_max_tokens,
                )

                raw_content = response.choices[0].message.content
                clean_content, thinking = parse_thinking_tokens(raw_content)

                if self._strip_thinking:
                    final_content = clean_content
                else:
                    final_content = raw_content

                duration_ms = int((time.time() - start_time) * 1000)
                
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else len(raw_content) // 4
                total_tokens = response.usage.total_tokens if response.usage else prompt_tokens + response_tokens
                token_count = response_tokens

                return LLMResponse(
                    content=final_content,
                    thinking=thinking,
                    raw_content=raw_content,
                    model=self._model,
                    total_duration_ms=duration_ms,
                    token_count=token_count,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens
                )

            except Exception as e:
                last_error = e
                logger.warning("LM Studio chat failed (attempt %d/%d): %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2)
        
        logger.error("LM Studio chat request failed after %d attempts: %s", max_retries, last_error)
        raise ConnectionError(f"Failed to get LLM response after {max_retries} attempts: {last_error}")

    def chat_stream(self, messages: list[dict], **kwargs) -> Generator[tuple[str, str, LLMResponse], None, None]:
        """
        Stream chat response.
        Yields (clean_chunk, thinking_chunk, None) during generation,
        and (final_clean, final_thinking, LLMResponse) at completion.
        """
        start_time = time.time()
        
        req_temperature = kwargs.get("temperature", self._temperature)
        req_max_tokens = kwargs.get("max_tokens", 2048)
        
        try:
            mapped_messages = []
            for msg in messages:
                mapped_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            stream = self._client.chat.completions.create(
                model=self._model,
                messages=mapped_messages,
                temperature=req_temperature,
                max_tokens=req_max_tokens,
                stream=True
            )

            full_content = ""
            in_think_block = False

            for chunk in stream:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta.content or ""
                
                full_content += delta

                # Naive stream parsing for <think> tokens
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

            # Generation complete
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
                total_tokens=response_tokens
            )
            
            yield "", "", final_resp

        except Exception as e:
            logger.error("LM Studio stream request failed: %s", e)
            raise
