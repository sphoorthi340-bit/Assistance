"""
Jarvis V3.0 — Settings Module
==============================
Central configuration management using Pydantic Settings.

Architecture decision:
    Settings are loaded in this priority order (highest wins):
    1. Environment variables (prefixed with JARVIS_)
    2. .env file
    3. configs/config.yaml
    4. Hardcoded defaults in this file

    This layering allows:
    - config.yaml for version-controlled defaults
    - .env for local machine overrides (not committed)
    - env vars for runtime/CI overrides

Usage:
    from configs.settings import get_settings
    settings = get_settings()
    print(settings.llm.model)
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Resolve project root: two levels up from this file (configs/settings.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Phase 3: Mode settings
# ---------------------------------------------------------------------------

class ModeSettings(BaseModel):
    """Operating mode configuration."""
    development_mode: bool = True
    offline_mode: bool = False
    local_only_mode: bool = False


# ---------------------------------------------------------------------------
# Phase 3: Provider settings
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    enabled: bool = True
    base_url: str = ""
    model: str = ""
    timeout: int = 60
    priority_dev: int = 5
    priority_prod: int = 5
    daily_budget_calls: int = 100
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class ProvidersSettings(BaseModel):
    """Configuration for all LLM providers."""
    ollama: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        base_url="http://localhost:11434", timeout=120,
        priority_dev=4, priority_prod=1
    ))
    lm_studio: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=False, base_url="http://localhost:1234/v1", timeout=120,
        priority_dev=4, priority_prod=1
    ))
    openai: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        model="gpt-4o-mini", timeout=60,
        priority_dev=2, priority_prod=3, daily_budget_calls=20,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006
    ))
    gemini: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        model="gemini-2.0-flash", timeout=60,
        priority_dev=1, priority_prod=2, daily_budget_calls=50,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0
    ))
    anthropic: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        model="claude-3-5-haiku-20241022", timeout=60,
        priority_dev=3, priority_prod=4, daily_budget_calls=15,
        cost_per_1k_input=0.0008, cost_per_1k_output=0.004
    ))


# ---------------------------------------------------------------------------
# Phase 3: Local model settings
# ---------------------------------------------------------------------------

class ModelAliasConfig(BaseModel):
    """Provider-specific model name aliases for a single routing tier."""
    ollama: str = ""
    lm_studio: str = ""

    def for_provider(self, provider: str) -> str:
        """Return the provider-specific model name, or empty string if not configured."""
        return getattr(self, provider, "") or ""


class ModelTierSettings(BaseModel):
    """Per-tier provider-specific model alias mapping."""
    fast: ModelAliasConfig = Field(default_factory=lambda: ModelAliasConfig(
        ollama="llama3.2:1b",
        lm_studio="llama-3.2-3b-instruct",
    ))
    reasoning: ModelAliasConfig = Field(default_factory=lambda: ModelAliasConfig(
        ollama="qwen2.5:7b",
        lm_studio="qwen2.5-7b-instruct",
    ))
    coding: ModelAliasConfig = Field(default_factory=lambda: ModelAliasConfig(
        ollama="qwen2.5:7b",
        lm_studio="qwen2.5-7b-instruct",
    ))
    math: ModelAliasConfig = Field(default_factory=lambda: ModelAliasConfig(
        ollama="qwen2.5:7b",
        lm_studio="qwen2.5-7b-instruct",
    ))
    classifier: ModelAliasConfig = Field(default_factory=lambda: ModelAliasConfig(
        ollama="llama3.2:1b",
        lm_studio="llama-3.2-3b-instruct",
    ))

    def get_model_for(self, tier: str, provider: str) -> str:
        """Resolve the correct model name for a given tier and provider."""
        tier_config: ModelAliasConfig = getattr(self, tier, None)
        if tier_config:
            return tier_config.for_provider(provider)
        return ""


# Keep backward-compat alias so any remaining code referencing LocalModelSettings doesn't break
LocalModelSettings = ModelTierSettings


# ---------------------------------------------------------------------------
# Phase 3: Router settings
# ---------------------------------------------------------------------------

class RouterSettings(BaseModel):
    """Model router configuration."""
    keyword_classification: bool = True
    fallback_to_classifier: bool = True
    context_limit_local: int = 4096
    confidence_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Phase 3: Cloud settings
# ---------------------------------------------------------------------------

class CloudSettings(BaseModel):
    """Cloud LLM usage policy."""
    daily_budget_calls: int = 20
    compress_before_send: bool = True
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92


# ---------------------------------------------------------------------------
# Phase 3: Proactive layer settings
# ---------------------------------------------------------------------------

class ProactiveSettings(BaseModel):
    """Proactive layer configuration."""
    morning_briefing_time: str = "07:30"
    evening_nudge_time: str = "20:00"
    stale_goal_days: int = 5
    exam_alert_days: int = 7
    show_inbox_on_startup: bool = True


# ---------------------------------------------------------------------------
# Phase 3: Memory decay settings
# ---------------------------------------------------------------------------

class MemoryDecaySettings(BaseModel):
    """Memory decay and consolidation configuration."""
    permanent_threshold: int = 8
    medium_decay_days: int = 90
    short_decay_days: int = 30
    consolidation_similarity: float = 0.85
    consolidation_day: str = "sunday"


# ---------------------------------------------------------------------------
# Phase 3: Knowledge collection settings
# ---------------------------------------------------------------------------

class KnowledgeCollectionSettings(BaseModel):
    """Multi-collection knowledge store configuration."""
    personal_memory: str = "jarvis_memories"
    academic_knowledge: str = "jarvis_academic"
    project_docs: str = "jarvis_projects"
    reference_material: str = "jarvis_reference"


# ---------------------------------------------------------------------------
# Existing sub-configuration models (preserved from V2.5)
# ---------------------------------------------------------------------------

class LLMSettings(BaseModel):
    """Configuration for the local LLM (Ollama)."""
    model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    strip_thinking_tokens: bool = True
    request_timeout: int = 120
    temperature: float = 0.7
    max_retries: int = 3


class DatabaseSettings(BaseModel):
    """Configuration for SQLite persistent storage."""
    path: str = "data/jarvis.db"
    wal_mode: bool = True


class MemorySettings(BaseModel):
    """Configuration for the memory subsystem (ChromaDB + embeddings)."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chromadb_path: str = "vector_db"
    collection_name: str = "jarvis_memories"
    max_retrieved_memories: int = 3
    max_context_tokens: int = 500
    conversation_history_turns: int = 10
    similarity_threshold: float = 0.3
    importance_threshold: float = 0.4
    decay: MemoryDecaySettings = Field(default_factory=MemoryDecaySettings)


class LoggingSettings(BaseModel):
    """Configuration for the logging subsystem."""
    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 5_242_880  # 5 MB
    backup_count: int = 5
    format: str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"


class SystemSettings(BaseModel):
    """System-level configuration (prompts, behavior)."""
    safe_mode: bool = True
    debug_mode: bool = False
    show_model_debug: bool = False
    system_prompt: str = (
        "You are Jarvis, a persistent personal cognitive assistant.\n"
        "You help the user with projects, goals, academics, routines, and long-term planning.\n"
        "You remember past conversations and important facts about the user.\n"
        "You have access to the user's personal knowledge base.\n"
        "You prioritize continuity, precision, and actionable support.\n\n"
        "CAPABILITIES:\n"
        "- You can create, update, and track goals, habits, and projects.\n"
        "- You can search the user's uploaded documents and notes.\n"
        "- You can access analytics and accountability data.\n\n"
        "RULES:\n"
        "- You are NOT a chatbot. You are cognitive infrastructure.\n"
        "- Be concise and direct. Avoid unnecessary pleasantries.\n"
        "- Reference relevant past context when it aids the conversation.\n"
        "- When you perform actions, briefly confirm what was done.\n"
        "- If you don't know something, say so clearly.\n"
        "- Prioritize the user's stated goals and projects.\n"
        "- Do NOT explicitly mention 'memory', 'context', or 'database'. Just use the information naturally.\n"
        "- NEVER output internal prompt metadata like 'Past context:', 'Available turns:', or 'Knowledge Base'.\n\n"
        "{state_snapshot}\n\n"
        "{memories}\n\n"
        "{knowledge}\n\n"
        "{conversation_history}"
    )


class KnowledgeSettings(BaseModel):
    """Configuration for the knowledge/RAG pipeline."""
    collection_name: str = "jarvis_knowledge"
    documents_dir: str = "documents"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    max_retrieved_chunks: int = 5
    supported_formats: list[str] = [".pdf", ".md", ".txt", ".markdown"]
    collections: KnowledgeCollectionSettings = Field(
        default_factory=KnowledgeCollectionSettings
    )


class ActionEngineSettings(BaseModel):
    """Configuration for the action engine."""
    enabled: bool = True
    auto_execute_threshold: float = 0.8
    low_confidence_threshold: float = 0.4
    confirm_risk_levels: list[str] = ["high"]
    preview_risk_levels: list[str] = ["medium"]
    max_actions_per_message: int = 3


class SchedulerSettings(BaseModel):
    """Configuration for the background scheduler."""
    enabled: bool = True
    morning_summary_time: str = "08:00"
    nightly_reflection_time: str = "22:00"
    stale_project_days: int = 7
    memory_cleanup_interval: int = 24
    backup_interval_hours: int = 24
    max_medium_priority_per_day: int = 2
    max_low_priority_per_day: int = 1


class ContextSettings(BaseModel):
    """Configuration for unified context assembly."""
    total_budget_tokens: int = 1000
    state_budget_tokens: int = 175
    memory_budget_tokens: int = 300
    knowledge_budget_tokens: int = 350
    history_budget_tokens: int = 250


# ---------------------------------------------------------------------------
# Phase 4: S4 Extension
# ---------------------------------------------------------------------------

class S4RoleConfig(BaseModel):
    role_name: str = ""
    provider: str = "lm_studio"
    model_id: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048

class S4ModelsConfig(BaseModel):
    chief: S4RoleConfig = Field(default_factory=S4RoleConfig)
    analyst: S4RoleConfig = Field(default_factory=S4RoleConfig)
    engineer: S4RoleConfig = Field(default_factory=S4RoleConfig)
    mentor: S4RoleConfig = Field(default_factory=S4RoleConfig)
    rapid: S4RoleConfig = Field(default_factory=S4RoleConfig)

class S4UserProfile(BaseModel):
    name: str = "User"
    semester: int = 3
    branch: str = "ECE"
    target_cgpa: float = 8.5
    current_cgpa: Optional[float] = None
    ms_target_year: int = 2028

class S4Settings(BaseModel):
    enabled: bool = True
    user_profile: S4UserProfile = Field(default_factory=S4UserProfile)
    models: S4ModelsConfig = Field(default_factory=S4ModelsConfig)

# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------

class JarvisSettings(BaseModel):
    """Root configuration container for all Jarvis subsystems."""
    # Phase 3 additions
    mode: ModeSettings = Field(default_factory=ModeSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    local_models: ModelTierSettings = Field(default_factory=ModelTierSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    proactive: ProactiveSettings = Field(default_factory=ProactiveSettings)
    s4: S4Settings = Field(default_factory=S4Settings)

    # Existing subsystems (preserved)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    action_engine: ActionEngineSettings = Field(default_factory=ActionEngineSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    project_root: Path = PROJECT_ROOT

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a config-relative path to an absolute path from project root."""
        return self.project_root / relative_path


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def _load_yaml_config() -> dict:
    """Load configuration from configs/config.yaml if it exists."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    return {}


def _load_env_overrides() -> dict:
    """
    Load environment variable overrides.

    Convention: JARVIS_<SECTION>_<KEY> maps to settings.<section>.<key>
    Example: JARVIS_LLM_MODEL -> settings.llm.model
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    overrides: dict = {}
    prefix = "JARVIS_"

    # Map of env var suffixes to their config paths
    env_mapping = {
        # Existing mappings
        "LLM_MODEL": ("llm", "model"),
        "OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
        "DB_PATH": ("database", "path"),
        "CHROMADB_PATH": ("memory", "chromadb_path"),
        "EMBEDDING_MODEL": ("memory", "embedding_model"),
        "MAX_RETRIEVED_MEMORIES": ("memory", "max_retrieved_memories"),
        "MAX_CONTEXT_TOKENS": ("memory", "max_context_tokens"),
        "CONVERSATION_HISTORY_TURNS": ("memory", "conversation_history_turns"),
        "LOG_LEVEL": ("logging", "level"),
        "LOG_DIR": ("logging", "log_dir"),
        # Phase 3 mappings
        "DEVELOPMENT_MODE": ("mode", "development_mode"),
        "OFFLINE_MODE": ("mode", "offline_mode"),
        "CLOUD_DAILY_BUDGET": ("cloud", "daily_budget_calls"),
        "LOCAL_ONLY_MODE": ("mode", "local_only_mode"),
        "SHOW_MODEL_DEBUG": ("system", "show_model_debug"),
    }

    for env_suffix, (section, key) in env_mapping.items():
        value = os.environ.get(f"{prefix}{env_suffix}")
        if value is not None:
            if section not in overrides:
                overrides[section] = {}
            # Attempt type conversion
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            overrides[section][key] = value

    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings() -> JarvisSettings:
    """
    Load settings with full priority chain:
    env vars > .env file > config.yaml > hardcoded defaults.
    """
    # Start with YAML config
    yaml_config = _load_yaml_config()

    # Apply environment variable overrides
    env_overrides = _load_env_overrides()
    merged = _deep_merge(yaml_config, env_overrides)

    # Build the settings object (Pydantic validates everything)
    return JarvisSettings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> JarvisSettings:
    """
    Get the singleton settings instance (cached after first call).

    Use this function throughout the codebase to access configuration.
    Call load_settings() directly if you need a fresh, uncached load.
    """
    return load_settings()
