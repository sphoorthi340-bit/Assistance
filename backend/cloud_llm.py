"""
Jarvis Phase 3 — Cloud LLM
============================
Unified cloud API wrapper across OpenAI, Google Gemini, and Anthropic Claude.
Features context compression, token/cost tracking, semantic response caching,
and budget enforcement.
"""

import os
import time
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from backend.provider_manager import ProviderManager
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CloudResponse:
    """Standardized response from any cloud provider."""
    content: str
    provider: str
    model: str
    total_duration_ms: int
    cached: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    response_time_ms: int = 0
    compressed: bool = False


# ---------------------------------------------------------------------------
# Cloud LLM Wrapper
# ---------------------------------------------------------------------------

class CloudLLM:
    def __init__(self, db: DatabaseManager, provider_manager: ProviderManager, vector_store, settings=None):
        self._db = db
        self._pm = provider_manager
        self._vector_store = vector_store
        self._settings = settings or get_settings()
        
        # We need an OllamaClient specifically for context compression (using fast model)
        from backend.llm import OllamaClient
        self._local_compressor = OllamaClient(settings=self._settings)

        logger.info("CloudLLM initialized")

    # -------------------------------------------------------------------
    # Core Interface
    # -------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        messages: list[dict] = None,
        prefer: str = "gemini",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Optional[CloudResponse]:
        """
        Unified completion interface.
        Will try the preferred provider, falling back to others if needed.
        Handles caching, budget checks, and compression automatically.
        """
        messages = messages or []
        
        # 1. Generate query hash for caching
        # We hash the string representation of all messages + prompt
        query_text = json.dumps(messages) + prompt
        query_hash = hashlib.sha256(query_text.encode('utf-8')).hexdigest()

        # 2. Check cache
        if self._settings.cloud.cache_enabled:
            cached_resp = self._check_cache(query_hash)
            if cached_resp:
                logger.info("CloudLLM cache hit for query: %s...", query_hash[:8])
                return CloudResponse(
                    content=cached_resp["response_text"],
                    provider=cached_resp["provider"],
                    model=cached_resp["model"],
                    cached=True
                )

        # 3. Context compression (if enabled and history is large)
        compressed = False
        if self._settings.cloud.compress_before_send and len(messages) > 3:
            try:
                logger.info("Compressing conversation context before cloud call")
                compressed_history = self._compress_context(messages)
                messages = [{"role": "system", "content": f"Previous context summary: {compressed_history}"}]
                compressed = True
            except Exception as e:
                logger.warning("Context compression failed, using full history: %s", e)

        # Determine provider fallback order
        candidates = [prefer]
        for p in self._pm.CLOUD_PROVIDERS:
            if p != prefer:
                candidates.append(p)

        # ── TEMPORARY DEBUG LOGGING (Gemini quota diagnosis) ──────────
        logger.warning(
            "[CLOUD-DEBUG] complete() called: prefer=%s | candidates=%s | "
            "prompt_len=%d | num_messages=%d",
            prefer, candidates, len(prompt), len(messages)
        )
        for _cand in candidates:
            _avail = self._pm.is_available(_cand)
            _budget_ok = self._enforce_budget(_cand) if _avail else "N/A"
            logger.warning(
                "[CLOUD-DEBUG]   candidate=%s | is_available=%s | budget_ok=%s",
                _cand, _avail, _budget_ok
            )
        # ── END DEBUG ─────────────────────────────────────────────────

        start_time = time.time()
        response = None
        selected_provider = None
        selected_model = None

        # 4. Try providers in order
        for provider in candidates:
            if not self._pm.is_available(provider):
                continue
                
            if not self._enforce_budget(provider):
                logger.warning("Daily budget exhausted for %s, skipping", provider)
                continue

            config = self._pm._get_provider_config(provider)
            selected_model = config.model
            
            try:
                logger.info("Calling cloud provider: %s (%s)", provider, selected_model)
                if provider == "openai":
                    response = self._call_openai(prompt, messages, selected_model, max_tokens, temperature)
                elif provider == "gemini":
                    response = self._call_gemini(prompt, messages, selected_model, max_tokens, temperature)
                elif provider == "anthropic":
                    response = self._call_anthropic(prompt, messages, selected_model, max_tokens, temperature)

                if response:
                    selected_provider = provider
                    response.provider = provider
                    response.model = selected_model
                    response.response_time_ms = int((time.time() - start_time) * 1000)
                    response.compressed = compressed
                    cost_in = (response.tokens_in / 1000) * config.cost_per_1k_input
                    cost_out = (response.tokens_out / 1000) * config.cost_per_1k_output
                    response.estimated_cost_usd = cost_in + cost_out
                    self._pm.record_call(provider, response.response_time_ms, success=True)
                    break
            except Exception as e:
                err_str = str(e).lower()
                # ── TEMPORARY DEBUG ───────────────────────────────────
                logger.warning(
                    "[CLOUD-DEBUG] EXCEPTION from %s: type=%s | "
                    "full_error=%s",
                    provider, type(e).__name__, str(e)[:500]
                )
                # ── END DEBUG ─────────────────────────────────────────
                # Detect quota/rate-limit errors and mark the provider accordingly
                if any(kw in err_str for kw in ("429", "quota", "rate_limit", "resource_exhausted",
                                                 "quota_exceeded", "requestsperday", "per_day")):
                    logger.warning("Quota error from %s — marking quota_exhausted: %s", provider, e)
                    self._pm.record_quota_error(provider)
                else:
                    logger.error("Cloud provider %s failed: %s", provider, e)
                    self._pm.record_call(provider, (time.time() - start_time) * 1000, success=False)
                # Try next provider in all cases

        if not response:
            logger.error("All cloud providers failed or are unavailable/exhausted")
            return None

        # 5. Log usage and cache
        self._log_usage(
            provider=selected_provider,
            model=selected_model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost=response.estimated_cost_usd,
            query_hash=query_hash,
            compressed=compressed
        )
        
        if self._settings.cloud.cache_enabled:
            self._cache_response(query_hash, query_text, response.content, selected_provider, selected_model)

        return response

    # -------------------------------------------------------------------
    # Internal Mechanisms
    # -------------------------------------------------------------------

    def _compress_context(self, messages: list[dict]) -> str:
        """Use local fast model to compress conversation history."""
        original_model = self._local_compressor._model
        # Resolve the alias to a plain string for Ollama (compressor always uses Ollama)
        fast_model = self._settings.local_models.get_model_for("fast", "ollama")
        if not fast_model:
            fast_model = "llama3.2:1b"  # safe default
        self._local_compressor._model = fast_model
        
        history_text = ""
        for m in messages:
            history_text += f"{m['role'].upper()}: {m['content']}\n\n"
            
        prompt = (
            "Summarize the following conversation history into a concise, factual summary "
            "of key points, decisions, and facts discussed. Keep it under 150 words.\n\n"
            f"{history_text}"
        )
        
        try:
            resp = self._local_compressor.chat([{"role": "user", "content": prompt}])
            self._local_compressor._model = original_model
            return resp.content
        except Exception:
            self._local_compressor._model = original_model
            raise

    def _check_cache(self, query_hash: str) -> Optional[dict]:
        """Check SQLite for exact hash match."""
        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT response_text, provider, model FROM cloud_cache "
                    "WHERE query_hash = ?", (query_hash,)
                ).fetchone()
                
                if row:
                    # Update hit count
                    conn.execute(
                        "UPDATE cloud_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                        "WHERE query_hash = ?",
                        (datetime.now(timezone.utc).isoformat(), query_hash)
                    )
                    return dict(row)
        except Exception as e:
            logger.error("Cache check failed: %s", e)
        return None

    def _cache_response(self, query_hash: str, query_text: str, response_text: str, provider: str, model: str):
        """Store response in SQLite cache."""
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO cloud_cache (query_hash, query_text, response_text, provider, model, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (query_hash, query_text, response_text, provider, model, datetime.now(timezone.utc).isoformat())
                )
        except Exception as e:
            logger.error("Cache store failed: %s", e)

    def _enforce_budget(self, provider: str) -> bool:
        """Check if daily quota is exceeded."""
        config = self._pm._get_provider_config(provider)
        if config.daily_budget_calls <= 0:
            return True  # Unlimited
            
        usage = self._pm.get_daily_usage(provider)
        return usage["calls"] < config.daily_budget_calls

    def _log_usage(self, provider, model, tokens_in, tokens_out, cost, query_hash, compressed):
        """Log call to cloud_usage table."""
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO cloud_usage (provider, model, tokens_in, tokens_out, estimated_cost_usd, timestamp, query_hash, compressed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (provider, model, tokens_in, tokens_out, cost, datetime.now(timezone.utc).isoformat(), query_hash, 1 if compressed else 0)
                )
        except Exception as e:
            logger.error("Failed to log cloud usage: %s", e)

    # -------------------------------------------------------------------
    # Provider-Specific API Wrappers
    # -------------------------------------------------------------------

    def _call_openai(self, prompt: str, messages: list[dict], model: str, max_tokens: int, temperature: float) -> CloudResponse:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
            
        client = openai.OpenAI(api_key=api_key)
        
        # Format messages
        oai_messages = messages.copy()
        oai_messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return CloudResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens
        )

    def _call_gemini(self, prompt: str, messages: list[dict], model: str, max_tokens: int, temperature: float) -> CloudResponse:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        # ── TEMPORARY DEBUG LOGGING (Gemini quota diagnosis) ──────────
        _key_fingerprint = api_key[-6:] if api_key else "NONE"
        _prompt_len = len(prompt)
        _total_msg_chars = sum(len(m.get("content", "")) for m in messages) + _prompt_len
        _est_tokens = _total_msg_chars // 4
        logger.warning(
            "[GEMINI-DEBUG] provider=gemini | model=%s | prompt_chars=%d | "
            "total_chars=%d | est_tokens=%d | num_messages=%d | "
            "api_key_fingerprint=...%s | max_tokens=%d | temperature=%.2f",
            model, _prompt_len, _total_msg_chars, _est_tokens,
            len(messages), _key_fingerprint, max_tokens, temperature
        )
        logger.warning(
            "[GEMINI-DEBUG] prompt_preview: %.200s",
            prompt[:200]
        )
        logger.warning(
            "[GEMINI-DEBUG] message_roles: %s",
            [m.get("role") for m in messages]
        )
        # Check if key matches the one in test_gemini.py (last 6 chars: "T5l0JA")
        logger.warning(
            "[GEMINI-DEBUG] API key match check: loaded_key[-6:]='%s' | "
            "expected(test_gemini.py)='T5l0JA' | match=%s",
            _key_fingerprint, _key_fingerprint == "T5l0JA"
        )
        # ── END DEBUG ─────────────────────────────────────────────────

        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(model)
        
        # Gemini format mapping
        gemini_messages = []
        system_instructions = ""
        
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_instructions += msg["content"] + "\n\n"
            else:
                gemini_role = "model" if role == "assistant" else "user"
                # If there's a system message, we prepend it to the first user message
                if system_instructions and not gemini_messages and gemini_role == "user":
                    content = system_instructions + msg["content"]
                    system_instructions = ""
                else:
                    content = msg["content"]
                
                gemini_messages.append({"role": gemini_role, "parts": [content]})
                
        # Handle trailing system prompt if no user messages existed
        final_prompt = prompt
        if system_instructions:
            final_prompt = system_instructions + prompt
            
        gemini_messages.append({"role": "user", "parts": [final_prompt]})

        # ── TEMPORARY DEBUG: Log final Gemini payload shape ──────────
        _final_payload_chars = sum(len(str(m.get("parts", ""))) for m in gemini_messages)
        logger.warning(
            "[GEMINI-DEBUG] final_payload: num_gemini_msgs=%d | "
            "total_payload_chars=%d | roles=%s",
            len(gemini_messages), _final_payload_chars,
            [m.get("role") for m in gemini_messages]
        )
        # Compare: test_gemini.py sends a single string "Say hello" directly
        # Jarvis sends a list of {role, parts} dicts — this is valid but different
        # ── END DEBUG ─────────────────────────────────────────────────

        response = genai_model.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        # Fallback estimation if usage metadata isn't available
        tokens_in = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else int(len(str(gemini_messages))/4)
        tokens_out = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else int(len(response.text)/4)
        
        return CloudResponse(
            content=response.text,
            provider="gemini",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out
        )

    def _call_anthropic(self, prompt: str, messages: list[dict], model: str, max_tokens: int, temperature: float) -> CloudResponse:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
            
        client = anthropic.Anthropic(api_key=api_key)
        
        # Anthropic requires system prompt as a top-level parameter
        system_prompt = ""
        ant_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n\n"
            else:
                role = "assistant" if msg["role"] == "assistant" else "user"
                ant_messages.append({"role": role, "content": msg["content"]})
                
        ant_messages.append({"role": "user", "content": prompt})
        
        response = client.messages.create(
            model=model,
            system=system_prompt.strip() if system_prompt else anthropic.NOT_GIVEN,
            messages=ant_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return CloudResponse(
            content=response.content[0].text,
            provider="anthropic",
            model=model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens
        )


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()
    
    settings = get_settings()
    db = DatabaseManager(settings=settings)
    pm = ProviderManager(db=db, settings=settings)
    
    # Mock vector store
    class MockVS:
        pass
        
    cloud = CloudLLM(db=db, provider_manager=pm, vector_store=MockVS(), settings=settings)
    
    # Test call (requires an API key in env)
    print("Testing cloud call...")
    try:
        response = cloud.complete("What is the capital of France?", prefer="openai")
        if response:
            print(f"Response: {response.content}")
            print(f"Provider: {response.provider}")
            print(f"Cost: ${response.estimated_cost_usd:.6f}")
            print(f"Cached: {response.cached}")
        else:
            print("No providers available or all failed.")
    except Exception as e:
        print(f"Test failed: {e}")
