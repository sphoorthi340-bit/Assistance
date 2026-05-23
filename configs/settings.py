"""
Jarvis V1 — Settings Module
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
# Sub-configuration models
# ---------------------------------------------------------------------------

class LLMSettings(BaseModel):
    """Configuration for the local LLM (Ollama + DeepSeek-R1)."""
    model: str = "deepseek-r1:7b"
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


class LoggingSettings(BaseModel):
    """Configuration for the logging subsystem."""
    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 5_242_880  # 5 MB
    backup_count: int = 5
    format: str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"


class SystemSettings(BaseModel):
    """System-level configuration (prompts, behavior)."""
    system_prompt: str = (
        "You are Jarvis, a persistent personal cognitive assistant.\n"
        "You help the user with projects, goals, academics, routines, and long-term planning.\n"
        "You remember past conversations and important facts about the user.\n"
        "You prioritize continuity, precision, and actionable support.\n\n"
        "IMPORTANT RULES:\n"
        "- You are NOT a chatbot. You are cognitive infrastructure.\n"
        "- Be concise and direct. Avoid unnecessary pleasantries.\n"
        "- Reference relevant past context when it aids the current conversation.\n"
        "- If you don't know something, say so clearly.\n"
        "- Prioritize the user's stated goals and projects.\n\n"
        "{memories}\n\n"
        "{conversation_history}"
    )


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------

class JarvisSettings(BaseModel):
    """Root configuration container for all Jarvis subsystems."""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
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
    }

    for env_suffix, (section, key) in env_mapping.items():
        value = os.environ.get(f"{prefix}{env_suffix}")
        if value is not None:
            if section not in overrides:
                overrides[section] = {}
            # Attempt numeric conversion for int/float fields
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
