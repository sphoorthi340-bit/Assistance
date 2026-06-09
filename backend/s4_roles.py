"""
JARVIS System 4 — Role Definitions & Model Manager
====================================================
Defines the 5 S4 model roles (Chief, Analyst, Engineer, Mentor, Rapid)
and provides a unified interface to call any role against any provider.

Each role encapsulates:
- Display identity
- Physical model binding (provider + model_id)
- System prompt (loaded from configs/s4_prompts/)
- Generation parameters
- Capability tags
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


def resolve_model_id(model_id: str, available_models: list[str]) -> str:
    """Map a configured model ID to an exact provider model name when possible."""
    if not model_id or not available_models:
        return model_id
    if model_id in available_models:
        return model_id

    model_lower = model_id.lower()
    for candidate in available_models:
        if candidate.lower() == model_lower:
            return candidate

    # Suffix match (e.g. phi-4-mini-reasoning -> microsoft/phi-4-mini-reasoning)
    for candidate in available_models:
        if candidate.lower().endswith("/" + model_lower) or candidate.lower().endswith(model_lower):
            return candidate

    norm = model_lower.replace("-", "").replace("/", "").replace("_", "")
    for candidate in available_models:
        cand_norm = candidate.lower().replace("-", "").replace("/", "").replace("_", "")
        if norm in cand_norm or cand_norm.endswith(norm):
            return candidate

    return model_id


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ModelRole:
    """Represents one of the 5 S4 AI roles."""
    name: str                           # chief | analyst | engineer | mentor | rapid
    display_name: str                   # Human-readable label
    provider: str                       # lm_studio | ollama
    model_id: str                       # Exact model identifier string
    system_prompt_file: str             # Relative path from project root
    temperature: float = 0.7
    max_tokens: int = 2048
    capabilities: list = field(default_factory=list)
    _system_prompt_cache: str = field(default="", init=False, repr=False)

    def load_system_prompt(self) -> str:
        """Load and cache the system prompt from disk."""
        if self._system_prompt_cache:
            return self._system_prompt_cache
        try:
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                self.system_prompt_file
            )
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._system_prompt_cache = f.read()
            logger.debug("Loaded system prompt for role '%s'", self.name)
            return self._system_prompt_cache
        except FileNotFoundError:
            logger.warning("System prompt file not found: %s", self.system_prompt_file)
            return f"You are JARVIS {self.display_name}. Be helpful, concise, and accurate."

    def format_system_prompt(self, **kwargs) -> str:
        """Format the system prompt with context variables."""
        template = self.load_system_prompt()
        # Fill placeholders, default empty string if not provided
        placeholders = {
            "state_snapshot": "",
            "memories": "",
            "knowledge": "",
            "conversation_history": ""
        }
        placeholders.update(kwargs)
        try:
            return template.format(**placeholders)
        except KeyError as e:
            logger.warning("Missing placeholder in system prompt for role '%s': %s", self.name, e)
            return template


@dataclass
class RoleCallResult:
    """Result from calling a single role."""
    role_name: str
    content: str
    model_id: str
    provider: str
    duration_ms: int
    success: bool
    error: Optional[str] = None
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Role Registry
# ---------------------------------------------------------------------------

def _build_roles(settings=None) -> dict[str, ModelRole]:
    """Build role registry from settings."""
    s = settings or get_settings()

    # Read model bindings from config
    s4 = getattr(s, "s4", None)
    models_cfg = getattr(s4, "models", None)

    def _get(role_key: str, default_provider: str, default_model: str,
             default_temp: float, default_tokens: int) -> tuple:
        if models_cfg and hasattr(models_cfg, role_key):
            cfg = getattr(models_cfg, role_key)
            if hasattr(cfg, "provider"):  # It's an S4RoleConfig object
                mod_id = getattr(cfg, "model_id", "")
                return (
                    getattr(cfg, "provider", default_provider),
                    mod_id if mod_id else default_model,
                    getattr(cfg, "temperature", default_temp),
                    getattr(cfg, "max_tokens", default_tokens),
                )
            elif isinstance(cfg, dict):
                mod_id = cfg.get("model_id", "")
                return (
                    cfg.get("provider", default_provider),
                    mod_id if mod_id else default_model,
                    cfg.get("temperature", default_temp),
                    cfg.get("max_tokens", default_tokens),
                )
        return default_provider, default_model, default_temp, default_tokens

    chief_p, chief_m, chief_t, chief_tok = _get("chief", "lm_studio", "qwen3-4b", 0.7, 2048)
    analyst_p, analyst_m, analyst_t, analyst_tok = _get(
        "analyst", "lm_studio", "microsoft/phi-4-mini-reasoning", 0.4, 3072
    )
    engineer_p, engineer_m, engineer_t, engineer_tok = _get("engineer", "lm_studio", "qwen2.5-7b-instruct", 0.3, 4096)
    mentor_p, mentor_m, mentor_t, mentor_tok = _get("mentor", "lm_studio", "google/gemma-3-4b", 0.8, 2048)
    rapid_p, rapid_m, rapid_t, rapid_tok = _get("rapid", "ollama", "llama3.2:1b", 0.5, 512)

    return {
        "chief": ModelRole(
            name="chief",
            display_name="Chief of Staff",
            provider=chief_p,
            model_id=chief_m,
            system_prompt_file="configs/s4_prompts/chief.txt",
            temperature=chief_t,
            max_tokens=chief_tok,
            capabilities=["planning", "orchestration", "prioritization",
                          "daily_management", "distraction_guard", "synthesis"]
        ),
        "analyst": ModelRole(
            name="analyst",
            display_name="Strategic Analyst",
            provider=analyst_p,
            model_id=analyst_m,
            system_prompt_file="configs/s4_prompts/analyst.txt",
            temperature=analyst_t,
            max_tokens=analyst_tok,
            capabilities=["deep_reasoning", "research_analysis", "ms_strategy",
                          "career_decisions", "tradeoff_evaluation", "verification"]
        ),
        "engineer": ModelRole(
            name="engineer",
            display_name="Lead Software Engineer",
            provider=engineer_p,
            model_id=engineer_m,
            system_prompt_file="configs/s4_prompts/engineer.txt",
            temperature=engineer_t,
            max_tokens=engineer_tok,
            capabilities=["code_generation", "debugging", "refactoring",
                          "architecture", "code_review", "jarvis_development"]
        ),
        "mentor": ModelRole(
            name="mentor",
            display_name="Learning Mentor",
            provider=mentor_p,
            model_id=mentor_m,
            system_prompt_file="configs/s4_prompts/mentor.txt",
            temperature=mentor_t,
            max_tokens=mentor_tok,
            capabilities=["ece_tutoring", "math_derivations", "exam_prep",
                          "brainstorming", "concept_explanation", "research_simplification"]
        ),
        "rapid": ModelRole(
            name="rapid",
            display_name="Rapid Assistant",
            provider=rapid_p,
            model_id=rapid_m,
            system_prompt_file="configs/s4_prompts/rapid.txt",
            temperature=rapid_t,
            max_tokens=rapid_tok,
            capabilities=["quick_lookup", "summarization", "focus_check",
                          "reminders", "status_check", "triage"]
        ),
    }


# ---------------------------------------------------------------------------
# Role Manager
# ---------------------------------------------------------------------------

class S4RoleManager:
    """
    Manages the 5 S4 model roles and provides a unified calling interface.

    Wraps the existing LM Studio and Ollama clients to route calls
    to the correct physical model per role.
    """

    def __init__(self, ollama_client=None, lm_studio_client=None, settings=None):
        """
        Args:
            ollama_client: OllamaClient instance (for 'rapid' role and fallback)
            lm_studio_client: LMStudioClient instance (for chief/analyst/engineer/mentor)
            settings: Settings instance (optional, uses get_settings() if None)
        """
        self._ollama = ollama_client
        self._lm_studio = lm_studio_client
        self._settings = settings or get_settings()
        self._roles = _build_roles(self._settings)
        self._resolve_role_models()
        self._apply_single_model_mode()
        logger.info("S4RoleManager initialized with %d roles", len(self._roles))

    def _apply_single_model_mode(self) -> None:
        """Collapse all LM Studio roles onto one anchor model to avoid GPU reloads."""
        s4 = getattr(self._settings, "s4", None)
        if not s4 or not getattr(s4, "single_model_mode", True):
            return

        anchor = getattr(s4, "anchor_model", "qwen3-4b")
        lm_models = self._provider_models("lm_studio")
        anchor_resolved = resolve_model_id(anchor, lm_models)

        for role in self._roles.values():
            if role.name == "rapid":
                continue
            if role.provider == "lm_studio":
                if role.model_id != anchor_resolved:
                    logger.info(
                        "Single-model mode: %s %s -> %s",
                        role.name, role.model_id, anchor_resolved,
                    )
                role.model_id = anchor_resolved

    def _provider_models(self, provider: str) -> list[str]:
        try:
            if provider == "ollama" and self._ollama:
                health = self._ollama.check_health()
            elif provider == "lm_studio" and self._lm_studio:
                health = self._lm_studio.check_health()
            else:
                return []
            if health.get("status") == "healthy":
                return health.get("available_models", [])
        except Exception as e:
            logger.warning("Failed to fetch models for provider '%s': %s", provider, e)
        return []

    def _resolve_role_models(self) -> None:
        """Resolve configured model IDs to exact provider names at startup."""
        for role in self._roles.values():
            available = self._provider_models(role.provider)
            resolved = resolve_model_id(role.model_id, available)
            if resolved != role.model_id:
                logger.info(
                    "Resolved S4 role '%s' model: %s -> %s",
                    role.name, role.model_id, resolved,
                )
                role.model_id = resolved

    def _role_model_available(self, role: ModelRole) -> bool:
        available = self._provider_models(role.provider)
        if not available:
            return False
        resolved = resolve_model_id(role.model_id, available)
        return resolved in available

    # -------------------------------------------------------------------
    # Role Access
    # -------------------------------------------------------------------

    def get_role(self, name: str) -> Optional[ModelRole]:
        """Get a role definition by name."""
        return self._roles.get(name)

    def all_roles(self) -> dict[str, ModelRole]:
        """Return all role definitions."""
        return self._roles.copy()

    def is_available(self, role_name: str) -> bool:
        """Check if the provider is reachable and the role's model is loaded."""
        role = self._roles.get(role_name)
        if not role:
            return False
        try:
            if role.provider == "ollama" and self._ollama:
                h = self._ollama.check_health()
                if h.get("status") != "healthy":
                    return False
                return resolve_model_id(role.model_id, h.get("available_models", [])) in h.get(
                    "available_models", []
                )
            elif role.provider == "lm_studio" and self._lm_studio:
                h = self._lm_studio.check_health()
                if h.get("status") != "healthy":
                    return False
                return self._role_model_available(role)
            return False
        except Exception:
            return False

    def health_report(self) -> dict:
        """Return availability status for all 5 roles."""
        return {
            name: {
                "available": self.is_available(name),
                "provider": role.provider,
                "model_id": role.model_id,
                "display_name": role.display_name,
            }
            for name, role in self._roles.items()
        }

    # -------------------------------------------------------------------
    # Core Call Interface
    # -------------------------------------------------------------------

    def call_role(
        self,
        role_name: str,
        message: str,
        conversation_history: list = None,
        context_kwargs: dict = None,
        system_override: str = None,
    ) -> RoleCallResult:
        """
        Call a specific role with a message.

        Args:
            role_name: One of chief|analyst|engineer|mentor|rapid
            message: The user message to send
            conversation_history: Prior conversation turns [{role, content}]
            context_kwargs: Variables to inject into the system prompt
                            (state_snapshot, memories, knowledge, conversation_history)
            system_override: If provided, replaces the role's system prompt entirely

        Returns:
            RoleCallResult with content, timing, and metadata
        """
        role = self._roles.get(role_name)
        if not role:
            return RoleCallResult(
                role_name=role_name, content="", model_id="", provider="",
                duration_ms=0, success=False, error=f"Unknown role: {role_name}"
            )

        history = conversation_history or []
        ctx = context_kwargs or {}
        system_prompt = system_override or role.format_system_prompt(**ctx)

        start = time.time()
        try:
            response_text = self._execute_call(role, message, history, system_prompt)
            duration_ms = int((time.time() - start) * 1000)
            logger.info(
                "Role '%s' responded in %dms via %s/%s",
                role_name, duration_ms, role.provider, role.model_id
            )
            return RoleCallResult(
                role_name=role_name,
                content=response_text,
                model_id=role.model_id,
                provider=role.provider,
                duration_ms=duration_ms,
                success=True
            )
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            logger.error("Role '%s' call failed: %s", role_name, e)
            fallback_text, fallback_meta = self._fallback_call(
                role_name, message, history, system_prompt
            )
            if fallback_text:
                return RoleCallResult(
                    role_name=role_name,
                    content=fallback_text,
                    model_id=fallback_meta.get("model_id", "fallback"),
                    provider=fallback_meta.get("provider", "fallback"),
                    duration_ms=duration_ms,
                    success=True,
                    error=str(e),
                    fallback_used=True,
                )
            return RoleCallResult(
                role_name=role_name, content="", model_id=role.model_id,
                provider=role.provider, duration_ms=duration_ms,
                success=False, error=str(e)
            )

    def _execute_call(
        self,
        role: ModelRole,
        message: str,
        history: list,
        system_prompt: str
    ) -> str:
        """Route the actual LLM call to the correct provider client."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        if role.provider == "lm_studio" and self._lm_studio:
            original_model = self._lm_studio._model
            self._lm_studio._model = role.model_id
            try:
                resp = self._lm_studio.chat(
                    messages,
                    max_tokens=role.max_tokens,
                    temperature=role.temperature,
                    max_retries=3,
                    timeout=60,
                )
                return resp.content
            finally:
                self._lm_studio._model = original_model

        elif role.provider == "ollama" and self._ollama:
            original_model = self._ollama._model
            self._ollama._model = role.model_id
            try:
                resp = self._ollama.chat(
                    messages,
                    options={"temperature": role.temperature, "num_predict": role.max_tokens},
                    max_retries=1,
                )
                return resp.content
            finally:
                self._ollama._model = original_model
        else:
            raise RuntimeError(
                f"No client available for provider '{role.provider}'. "
                "Ensure LM Studio or Ollama is running."
            )

    def _fallback_call(
        self, failed_role_name: str, message: str, history: list, system_prompt: str
    ) -> tuple[Optional[str], dict]:
        """Fallback chain: Ollama fast -> Ollama reasoning -> Chief LM retry.

        Fast model runs first because LM Studio often still holds the GPU;
        loading a 7B Ollama model alongside it typically OOMs (exit code 2).
        """
        attempts: list[tuple[str, ModelRole]] = []
        failed_role = self._roles.get(failed_role_name)
        prompt_file = (
            failed_role.system_prompt_file if failed_role
            else "configs/s4_prompts/rapid.txt"
        )
        fallback_temp = failed_role.temperature if failed_role else 0.7
        fallback_tokens = failed_role.max_tokens if failed_role else 2048
        fast_tokens = 1024 if failed_role_name != "rapid" else 512

        if self._ollama:
            fast_model = self._settings.local_models.get_model_for("fast", "ollama")
            reasoning_model = self._settings.local_models.get_model_for("reasoning", "ollama")

            if fast_model:
                attempts.append((
                    "ollama_fast",
                    ModelRole(
                        name=failed_role_name,
                        display_name=failed_role.display_name if failed_role else "Fallback",
                        provider="ollama",
                        model_id=fast_model,
                        system_prompt_file=prompt_file,
                        temperature=fallback_temp if failed_role_name != "rapid" else 0.5,
                        max_tokens=fast_tokens,
                    ),
                ))

            if reasoning_model and reasoning_model != fast_model and failed_role_name != "rapid":
                attempts.append((
                    "ollama_reasoning",
                    ModelRole(
                        name=failed_role_name,
                        display_name=failed_role.display_name if failed_role else "Fallback",
                        provider="ollama",
                        model_id=reasoning_model,
                        system_prompt_file=prompt_file,
                        temperature=fallback_temp,
                        max_tokens=fallback_tokens,
                    ),
                ))

        if failed_role_name != "chief":
            chief = self._roles.get("chief")
            if chief:
                attempts.append(("chief", chief))

        for label, role in attempts:
            try:
                if role.provider == "ollama":
                    if not self._ollama:
                        continue
                    h = self._ollama.check_health()
                    if h.get("status") != "healthy":
                        continue
                    if resolve_model_id(role.model_id, h.get("available_models", [])) not in h.get(
                        "available_models", []
                    ):
                        continue
                elif role.provider == "lm_studio":
                    if not self._lm_studio or not self._role_model_available(role):
                        continue

                content = self._execute_call(role, message, history, system_prompt)
                logger.warning("S4 fallback via '%s' succeeded after '%s' failed", label, failed_role_name)
                return content, {"model_id": role.model_id, "provider": role.provider}
            except Exception as e:
                logger.error("Fallback '%s' failed: %s", label, e)

        logger.error("All fallbacks exhausted for role '%s'", failed_role_name)
        return None, {}
