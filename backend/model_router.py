"""
Jarvis Phase 3 — Model Router V2
==================================
Multidimensional routing for LLM requests based on complexity,
privacy, latency requirements, and cost priorities.
"""

import time
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import re
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from backend.provider_manager import ProviderManager
from backend.cloud_llm import CloudLLM, CloudResponse
from backend.llm import OllamaClient, LLMResponse
from backend.observability import get_session_metrics
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    complexity: str       # low | medium | high | extreme
    privacy: str          # public | personal | sensitive
    latency_priority: str # realtime | normal | deep_thinking
    cost_priority: str    # minimize_cost | balanced | quality_first
    selected_provider: str
    selected_model: str
    reason: str
    confidence: float
    conversation_id: str = None


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Intelligently routes LLM requests strictly based on explicitly configured priorities,
    handling domain-specific models without faking unavailability.
    """
    def __init__(
        self,
        provider_manager: ProviderManager,
        cloud_llm: CloudLLM,
        ollama_client: OllamaClient,
        db: DatabaseManager,
        settings=None,
        lm_studio_client=None
    ):
        self._pm = provider_manager
        self._cloud = cloud_llm
        self._local_ollama = ollama_client
        self._local_lm_studio = lm_studio_client
        self._db = db
        self._settings = settings or get_settings()
        
        # Classifier always runs on Ollama (lightest local provider)
        self._classifier_model = self._settings.local_models.get_model_for("classifier", "ollama")
        
        logger.info("ModelRouter initialized")

    # -------------------------------------------------------------------
    # Core Routing Logic
    # -------------------------------------------------------------------

    def complete(self, message: str, conversation_history: list[dict] = None, system_prompt: str = None, conversation_id: str = None) -> LLMResponse:
        import time
        start_time = time.time()
        conversation_history = conversation_history or []
        fallback_triggered = False
        fallback_from = None
        fallback_reason = None
        
        # 1. Classify (with routing timing)
        routing_start = time.time()
        decision = self.route(message, conversation_history)
        routing_time_ms = int((time.time() - routing_start) * 1000)
        
        if conversation_id:
            decision.conversation_id = conversation_id
            
        # 2. Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.extend(conversation_history)
        
        # Derive a classification label
        classification_label = "unknown"
        if hasattr(self, '_derive_classification_label'):
            classification_label = self._derive_classification_label(decision)
        else:
            classification_label = decision.complexity
        
        # 3. Execute
        inference_start = time.time()
        
        if decision.selected_provider == "disabled":
            response = LLMResponse(content=f"Error: Required route disabled. {decision.reason}", model="system")
            self._log_routing(decision, 0)
            inference_time_ms = 0
        else:
            is_cloud = decision.selected_provider in self._pm.CLOUD_PROVIDERS
            
            # Guard against cloud in local-only mode
            if self._settings.mode.local_only_mode and (is_cloud or not decision.selected_provider):
                raise RuntimeError("Both LM Studio and Ollama failed. No cloud fallback available in LOCAL_ONLY_MODE.")
            
            try:
                if is_cloud:
                    cloud_resp = self._cloud.complete(
                        prompt=message,
                        messages=messages,
                        prefer=decision.selected_provider
                    )
                    
                    if cloud_resp:
                        response = LLMResponse(
                            content=cloud_resp.content,
                            thinking="",
                            raw_content=cloud_resp.content,
                            model=cloud_resp.model,
                            total_duration_ms=cloud_resp.response_time_ms,
                            token_count=cloud_resp.tokens_in + cloud_resp.tokens_out
                        )
                    else:
                        logger.warning("Cloud fallback failed, attempting local fallback")
                        fallback_triggered = True
                        fallback_from = decision.selected_provider
                        fallback_reason = "Cloud returned empty response"
                        get_session_metrics().record_fallback(
                            from_provider=decision.selected_provider,
                            to_provider="local",
                            reason="Cloud returned empty response",
                        )
                        # Determine tier from the original classification so the fallback model is correct
                        orig_tier = "reasoning"  # default for cloud-routed requests
                        fallback = self._get_best_local(orig_tier, "Cloud failed")
                        decision.selected_provider = fallback[0]
                        decision.selected_model = fallback[1]
                        decision.reason += f" (Cloud failed, fallback to local: {fallback[2]})"
                        response = self._call_local(decision.selected_provider, message, messages, decision.selected_model)
                else:
                    response = self._call_local(decision.selected_provider, message, messages, decision.selected_model)
                    
            except Exception as e:
                logger.error("Routing execution failed: %s", e)
                fallback_triggered = True
                orig_provider = decision.selected_provider
                fallback_from = orig_provider
                fallback_reason = str(e)
                orig_tier = "reasoning"
                try:
                    fallback = self._get_best_local(orig_tier, f"Error fallback: {e}")
                    get_session_metrics().record_fallback(
                        from_provider=orig_provider,
                        to_provider=fallback[0],
                        reason=str(e),
                    )
                    decision.selected_provider = fallback[0]
                    decision.selected_model = fallback[1]
                    decision.reason = f"Fallback after error: {e} | {fallback[2]}"
                    response = self._call_local(decision.selected_provider, message, messages, decision.selected_model)
                except Exception as fallback_err:
                    if self._settings.mode.local_only_mode:
                        raise RuntimeError("Both LM Studio and Ollama failed. No cloud fallback available in LOCAL_ONLY_MODE.") from fallback_err
                    raise
                    
            inference_time_ms = int((time.time() - inference_start) * 1000)

        # 4. Record metrics
        duration_ms = int((time.time() - start_time) * 1000)
        get_session_metrics().record_request(
            routing_time_ms=routing_time_ms,
            memory_time_ms=0, # Passed separately by memory system in practice, but we mock 0 here for now
            knowledge_time_ms=0,
            inference_time_ms=inference_time_ms,
            total_time_ms=duration_ms,
            provider=decision.selected_provider,
            model=decision.selected_model,
            classification=classification_label,
            fallback_triggered=fallback_triggered,
            fallback_from=fallback_from,
            fallback_reason=fallback_reason,
            prompt_tokens=getattr(response, 'prompt_tokens', 0),
            response_tokens=getattr(response, 'response_tokens', 0),
            total_tokens=getattr(response, 'total_tokens', 0)
        )

        # 5. Attach routing metadata to the response
        setattr(response, 'provider_name', decision.selected_provider)
        setattr(response, 'classification', classification_label)
        setattr(response, 'routing_time_ms', routing_time_ms)
        setattr(response, 'inference_time_ms', inference_time_ms)
        setattr(response, 'fallback_triggered', fallback_triggered)
        setattr(response, 'confidence', decision.confidence)

        # 6. Log routing decision
        self._log_routing(decision, duration_ms)

        
        return response

    def _call_local(self, provider: str, prompt: str, messages: list[dict], model: str) -> LLMResponse:
        client = self._local_lm_studio if provider == "lm_studio" else self._local_ollama
        if not client:
            raise ValueError(f"Local client {provider} is not initialized")
            
        original_model = client._model
        client._model = model
        
        try:
            o_messages = messages.copy()
            o_messages.append({"role": "user", "content": prompt})
            return client.chat(o_messages)
        finally:
            client._model = original_model

    # -------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------

    def route(self, message: str, conversation_history: list[dict] = None) -> RoutingDecision:
        classification = None
        if self._settings.router.keyword_classification:
            classification = self._classify_keywords(message)
            
        if self._settings.router.fallback_to_classifier:
            if not classification or classification.get("confidence", 0) < self._settings.router.confidence_threshold:
                llm_class = self._classify_with_llm(message)
                if llm_class:
                    classification = llm_class
                    
        if not classification:
            classification = {
                "complexity": "medium",
                "privacy": "personal",
                "latency": "normal",
                "cost": "balanced",
                "confidence": 0.3,
                "reason": "Default classification"
            }
            
        provider, model, reason = self._select_model(classification)
        
        return RoutingDecision(
            complexity=classification.get("complexity", "medium"),
            privacy=classification.get("privacy", "personal"),
            latency_priority=classification.get("latency", "normal"),
            cost_priority=classification.get("cost", "balanced"),
            selected_provider=provider,
            selected_model=model,
            reason=reason,
            confidence=classification.get("confidence", 0.5)
        )

    def _classify_keywords(self, message: str) -> Optional[dict]:
        msg_lower = message.lower()
        
        coding_patterns = [r"def ", r"function", r"import ", r"bug", r"error", r"python", r"code", r"script"]
        is_coding = any(re.search(p, msg_lower) for p in coding_patterns)
        
        math_patterns = [r"solve", r"equation", r"math", r"\+", r"-", r"\*", r"/", r"integral", r"derivative"]
        is_math = any(re.search(p, msg_lower) for p in math_patterns)
        
        sensitive_patterns = [r"password", r"secret", r"ssn", r"credit card", r"health", r"medical", r"therapy"]
        personal_patterns = [r"journal", r"diary", r"feeling", r"my life", r"family", r"friend"]
        
        is_sensitive = any(re.search(p, msg_lower) for p in sensitive_patterns)
        is_personal = any(re.search(p, msg_lower) for p in personal_patterns)
        
        length = len(message)
        if length < 50: complexity = "low"
        elif length < 300: complexity = "medium"
        elif length < 1000: complexity = "high"
        else: complexity = "extreme"
            
        if is_coding or is_math:
            complexity = max(complexity, "high")
            
        if is_sensitive: privacy = "sensitive"
        elif is_personal: privacy = "personal"
        else: privacy = "public"
            
        reason = "Keyword heuristic"
        if is_coding: reason += " (Coding detected)"
        if is_math: reason += " (Math detected)"
        
        return {
            "complexity": complexity,
            "privacy": privacy,
            "latency": "normal",
            "cost": "balanced",
            "confidence": 0.7,
            "reason": reason,
            "is_coding": is_coding,
            "is_math": is_math
        }

    def _classify_with_llm(self, message: str) -> Optional[dict]:
        prompt = f"""Analyze this user message and classify it for LLM routing.
Message: "{message[:500]}"

Output JSON strictly matching this schema:
{{
    "complexity": "low" | "medium" | "high" | "extreme",
    "privacy": "public" | "personal" | "sensitive",
    "latency": "realtime" | "normal" | "deep_thinking",
    "cost": "minimize_cost" | "balanced" | "quality_first",
    "is_coding": boolean,
    "is_math": boolean,
    "confidence": float between 0.0 and 1.0
}}
"""
        try:
            original_model = self._local_ollama._model
            self._local_ollama._model = self._classifier_model
            response = self._local_ollama._client.chat(
                model=self._classifier_model,
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            self._local_ollama._model = original_model
            data = json.loads(response["message"]["content"])
            data["reason"] = "LLM Classifier"
            return data
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Route Selection
    # -------------------------------------------------------------------

    def _select_model(self, classification: dict) -> tuple[str, str, str]:
        c_comp = classification.get("complexity", "medium")
        c_priv = classification.get("privacy", "personal")
        is_coding = classification.get("is_coding", False)
        is_math = classification.get("is_math", False)

        # Determine the logical tier
        if is_coding:
            tier = "coding"
        elif is_math:
            tier = "math"
        elif c_comp == "low":
            tier = "fast"
        else:
            tier = "reasoning"

        # 1. Privacy check: sensitive content must stay local
        if c_priv == "sensitive":
            return self._get_best_local(tier, "Privacy: Sensitive")

        # 3. Standard Priority Enforcement via ProviderManager
        optimal_list = self._pm.get_optimal_providers()
        local_only = self._settings.mode.local_only_mode
        
        if not optimal_list:
            return "disabled", "none", "No providers available."
            
        if local_only and all(p.name in self._pm.CLOUD_PROVIDERS for p in optimal_list):
            return "disabled", "none", "Local-only mode active, but no local providers available"

        best = optimal_list[0]

        if best.is_local:
            # Resolve the provider-specific alias for this tier
            return self._resolve_local(best.name, tier)

        # Cloud provider selected — use provider's configured default_model
        cloud_model = self._pm._get_provider_config(best.name).model
        if not cloud_model:
            # No model configured for cloud — fall back to local
            logger.warning("Cloud provider %s has no model configured, falling back to local", best.name)
            return self._get_best_local(tier, f"Cloud {best.name} missing model config")

        return best.name, cloud_model, f"Priority strictly enforced. Selected {best.name} (cloud)"

    def _resolve_local(self, provider: str, tier: str) -> tuple[str, str, str]:
        """
        Resolve a local provider + tier combination to the correct model name alias.

        Checks that the resolved model is actually available in the provider's model list.
        Falls back to other local providers if the primary doesn't have it.
        """
        model_name = self._settings.local_models.get_model_for(tier, provider)

        if not model_name:
            logger.warning("No alias configured for tier '%s' on provider '%s'", tier, provider)
            # Try any available local provider
            return self._get_best_local(tier, f"No alias for {provider}/{tier}")

        # Verify availability
        health = self._pm.check_health(provider)
        if model_name in health.available_models:
            return provider, model_name, f"Alias resolved: {tier} -> {provider}/{model_name}"

        # Model not available on primary provider — try other local providers
        for alt_prov in self._pm.get_optimal_providers(require_local=True):
            if alt_prov.name == provider:
                continue
            alt_model = self._settings.local_models.get_model_for(tier, alt_prov.name)
            if not alt_model:
                continue
            alt_health = self._pm.check_health(alt_prov.name)
            if alt_model in alt_health.available_models:
                return alt_prov.name, alt_model, f"Alias fallback: {tier} -> {alt_prov.name}/{alt_model}"

        # No local provider has this tier's model — fall back to any healthy model
        local_options = self._pm.get_optimal_providers(require_local=True)
        for p in local_options:
            # Try any available model on the provider
            p_health = self._pm._health_cache.get(p.name)
            if p_health and p_health.available_models:
                fallback_model = p_health.available_models[0]
                return p.name, fallback_model, (
                    f"Tier '{tier}' model not found; fallback to available {p.name}/{fallback_model}"
                )

        return "disabled", "none", f"No local provider has a model for tier '{tier}'."

    def _get_best_local(self, tier: str, reason: str) -> tuple[str, str, str]:
        """Get the best available local provider for a given tier."""
        locals_ = self._pm.get_optimal_providers(require_local=True)
        for p in locals_:
            result = self._resolve_local(p.name, tier)
            if result[0] != "disabled":
                return result[0], result[1], f"{reason} ({result[2]})"
        return "disabled", "none", f"{reason} (No local providers available for tier '{tier}')"


    # -------------------------------------------------------------------
    # Audit Logging
    # -------------------------------------------------------------------

    def _log_routing(self, decision: RoutingDecision, response_time_ms: int):
        """Persist routing decision to database."""
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO route_logs (timestamp, complexity, privacy, latency_priority, "
                    "cost_priority, selected_provider, selected_model, reason, response_time_ms, "
                    "confidence, conversation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        decision.complexity, decision.privacy, decision.latency_priority,
                        decision.cost_priority, decision.selected_provider, decision.selected_model,
                        decision.reason, response_time_ms, decision.confidence, decision.conversation_id
                    )
                )
        except Exception as e:
            logger.error("Failed to log routing decision: %s", e)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()
    
    settings = get_settings()
    db = DatabaseManager(settings=settings)
    pm = ProviderManager(db=db, settings=settings)
    
    # Mock Vector Store
    class MockVS: pass
    cloud = CloudLLM(db=db, provider_manager=pm, vector_store=MockVS(), settings=settings)
    local = OllamaClient(settings=settings)
    
    router = ModelRouter(pm, cloud, local, db, settings)
    
    test_queries = [
        "What is 2+2?",
        "Can you write a python script to parse JSON?",
        "I feel really sad today, my journal entry is...",
        "What are the implications of quantum gravity on general relativity?"
    ]
    
    print("=== Testing Model Router ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        decision = router.route(q)
        print(f"  Complexity: {decision.complexity}")
        print(f"  Privacy: {decision.privacy}")
        print(f"  Provider: {decision.selected_provider}")
        print(f"  Model: {decision.selected_model}")
        print(f"  Reason: {decision.reason}")
