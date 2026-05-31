"""
Jarvis Phase 3 — Provider Manager
====================================
Manages health, latency, quota, and availability for all LLM providers.

Supported providers:
    - Ollama (local)
    - LM Studio (local)
    - OpenAI (cloud)
    - Google Gemini (cloud)
    - Anthropic Claude (cloud)

The ProviderManager is queried by the ModelRouter to determine which
provider is optimal for a given request, based on the current operating
mode (development/production/offline) and provider constraints.
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Health check result for a single provider."""
    provider_name: str
    status: str = "unknown"         # healthy | unhealthy | quota_exhausted | disabled | unknown
    latency_ms: float = 0.0
    available_models: list[str] = field(default_factory=list)
    error: str = ""
    last_check: str = ""
    failure_count: int = 0
    quota_remaining: int = -1       # -1 = unlimited/unknown
    daily_calls: int = 0
    daily_cost: float = 0.0


@dataclass
class ProviderInfo:
    """Provider info used for routing decisions."""
    name: str
    status: str
    priority: int
    latency_ms: float
    quota_remaining: int
    is_local: bool
    model: str = ""


# ---------------------------------------------------------------------------
# Provider Manager
# ---------------------------------------------------------------------------

class ProviderManager:
    """
    Manages health monitoring, latency tracking, and availability
    for all LLM providers.

    Responsibilities:
        - Health checks (lightweight ping per provider)
        - Latency tracking (running average)
        - Quota tracking (daily call counts vs. budget)
        - Availability checks (API key presence, health status)
        - Automatic failover (disable unhealthy providers temporarily)
        - Optimal provider selection based on mode and constraints
    """

    # Providers classified by type
    LOCAL_PROVIDERS = {"ollama", "lm_studio"}
    CLOUD_PROVIDERS = {"openai", "gemini", "anthropic"}
    ALL_PROVIDERS = LOCAL_PROVIDERS | CLOUD_PROVIDERS

    # API key environment variable names
    _API_KEY_ENVVARS = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def __init__(self, db: DatabaseManager, settings=None):
        self._db = db
        self._settings = settings or get_settings()

        # In-memory health cache (refreshed by check_health calls)
        self._health_cache: dict[str, HealthStatus] = {}

        # Temporary disable tracking
        self._disabled_until: dict[str, datetime] = {}

        logger.info("ProviderManager initialized")

    # -------------------------------------------------------------------
    # Health checks
    # -------------------------------------------------------------------

    def check_health(self, provider_name: str) -> HealthStatus:
        """
        Run a health check for a specific provider.

        Returns:
            HealthStatus with current availability information.
        """
        if provider_name not in self.ALL_PROVIDERS:
            return HealthStatus(
                provider_name=provider_name,
                status="unknown",
                error=f"Unknown provider: {provider_name}",
            )

        provider_config = self._get_provider_config(provider_name)
        if not provider_config.enabled:
            status = HealthStatus(
                provider_name=provider_name,
                status="disabled",
                last_check=datetime.now(timezone.utc).isoformat(),
            )
            self._health_cache[provider_name] = status
            return status

        # If it's temporarily disabled (like for quota exhaustion), don't ping, just return the cached status
        if provider_name in self._disabled_until:
            if datetime.now(timezone.utc) < self._disabled_until[provider_name]:
                return self._health_cache.get(provider_name, HealthStatus(
                    provider_name=provider_name,
                    status="disabled",
                    error="Temporarily disabled"
                ))

        start_time = time.time()

        try:
            if provider_name == "ollama":
                status = self._check_ollama(provider_config)
            elif provider_name == "lm_studio":
                status = self._check_lm_studio(provider_config)
            elif provider_name in ("openai", "gemini", "anthropic"):
                status = self._check_cloud_provider(provider_name)
            else:
                status = HealthStatus(
                    provider_name=provider_name, status="unknown"
                )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            old_status = self._health_cache.get(provider_name)
            failure_count = (old_status.failure_count + 1) if old_status else 1

            status = HealthStatus(
                provider_name=provider_name,
                status="unhealthy",
                latency_ms=latency,
                error=str(e),
                failure_count=failure_count,
                last_check=datetime.now(timezone.utc).isoformat(),
            )
            logger.warning(
                "Health check failed for %s (failures: %d): %s",
                provider_name, failure_count, e
            )

        # Compute latency
        if status.latency_ms == 0:
            status.latency_ms = (time.time() - start_time) * 1000
        status.last_check = datetime.now(timezone.utc).isoformat()

        # Enrich with daily usage stats
        daily = self._get_daily_usage_from_db(provider_name)
        status.daily_calls = daily.get("calls", 0)
        status.daily_cost = daily.get("cost", 0.0)

        # Compute remaining quota
        budget = provider_config.daily_budget_calls
        status.quota_remaining = max(0, budget - status.daily_calls)

        # Cache and persist
        self._health_cache[provider_name] = status
        self._persist_health(status)

        return status

    def check_all_health(self) -> dict[str, HealthStatus]:
        """Run health checks for all enabled providers."""
        results = {}
        for name in self.ALL_PROVIDERS:
            config = self._get_provider_config(name)
            if config.enabled:
                results[name] = self.check_health(name)
        return results

    def _check_ollama(self, config) -> HealthStatus:
        """Check Ollama server health."""
        try:
            import ollama as ollama_lib
            client = ollama_lib.Client(host=config.base_url)
            models_response = client.list()
            available = [
                m.model for m in models_response.models
            ] if hasattr(models_response, 'models') else []

            return HealthStatus(
                provider_name="ollama",
                status="healthy",
                available_models=available,
            )
        except Exception as e:
            return HealthStatus(
                provider_name="ollama",
                status="unhealthy",
                error=str(e),
            )

    def _check_lm_studio(self, config) -> HealthStatus:
        """Check LM Studio server health (OpenAI-compatible API)."""
        try:
            import openai
            client = openai.OpenAI(
                base_url=config.base_url,
                api_key="lm-studio",  # LM Studio doesn't require a real key
            )
            models = client.models.list()
            available = [m.id for m in models.data]

            return HealthStatus(
                provider_name="lm_studio",
                status="healthy",
                available_models=available,
            )
        except Exception as e:
            return HealthStatus(
                provider_name="lm_studio",
                status="unhealthy",
                error=str(e),
            )

    def _check_cloud_provider(self, provider_name: str) -> HealthStatus:
        """
        Authenticate and verify a cloud provider is truly reachable.

        Healthy    = SDK imported + API key present + lightweight listing call succeeds.
        Quota      = 429 / resource_exhausted / quota_exceeded response.
        Unhealthy  = Auth failure, SDK missing, or network error.
        """
        env_var = self._API_KEY_ENVVARS.get(provider_name, "")
        api_key = os.environ.get(env_var, "")

        if not api_key:
            return HealthStatus(
                provider_name=provider_name,
                status="unhealthy",
                error=f"API key not set ({env_var})",
            )

        try:
            if provider_name == "openai":
                import openai
                client = openai.OpenAI(api_key=api_key)
                models = client.models.list()
                available = [m.id for m in models.data][:5]
                return HealthStatus(
                    provider_name="openai",
                    status="healthy",
                    available_models=available,
                )

            elif provider_name == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                models = list(genai.list_models())
                available = [m.name for m in models if "generateContent" in m.supported_generation_methods][:5]
                return HealthStatus(
                    provider_name="gemini",
                    status="healthy",
                    available_models=available,
                )

            elif provider_name == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=self._get_provider_config("anthropic").model or "claude-3-5-haiku-20241022",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}],
                )
                return HealthStatus(
                    provider_name="anthropic",
                    status="healthy",
                    available_models=[resp.model] if resp.model else [],
                )

        except ImportError as e:
            return HealthStatus(
                provider_name=provider_name,
                status="unhealthy",
                error=f"SDK missing: {e}",
            )
        except Exception as e:
            err_str = str(e).lower()
            # Quota / rate limit detection
            if any(kw in err_str for kw in ("429", "quota", "rate_limit", "resource_exhausted",
                                             "quota_exceeded", "requestsperday", "per_day")):
                logger.warning("Provider %s quota exhausted: %s", provider_name, e)
                return HealthStatus(
                    provider_name=provider_name,
                    status="quota_exhausted",
                    error=f"Quota exhausted: {e}",
                )
            # Auth / access errors
            if any(kw in err_str for kw in ("api key", "authentication", "invalid_api_key",
                                             "unauthorized", "permission_denied")):
                reason = "Authentication failed (invalid API key)"
            elif any(kw in err_str for kw in ("connect", "timeout", "network", "connection")):
                reason = "Network error (cannot reach provider)"
            else:
                reason = str(e)
            return HealthStatus(
                provider_name=provider_name,
                status="unhealthy",
                error=reason,
            )


    # -------------------------------------------------------------------
    # Availability & status
    # -------------------------------------------------------------------

    def is_available(self, provider_name: str) -> bool:
        """Check if a provider is currently available for use."""
        # Check temporary disable
        if provider_name in self._disabled_until:
            if datetime.now(timezone.utc) < self._disabled_until[provider_name]:
                return False
            else:
                del self._disabled_until[provider_name]

        # Check config
        config = self._get_provider_config(provider_name)
        if not config.enabled:
            return False

        # Check cached health — quota_exhausted is treated as unavailable
        cached = self._health_cache.get(provider_name)
        if cached:
            if cached.status in ("unhealthy", "disabled", "quota_exhausted"):
                return False
            if cached.status == "healthy":
                if cached.quota_remaining is not None and cached.quota_remaining <= 0:
                    return False
                return True

        # No cached status — run a quick check
        status = self.check_health(provider_name)
        return status.status == "healthy"

    def get_latency(self, provider_name: str) -> float:
        """Get last known latency in ms for a provider."""
        cached = self._health_cache.get(provider_name)
        return cached.latency_ms if cached else 0.0

    def get_remaining_quota(self, provider_name: str) -> int:
        """Get remaining daily call quota for a provider."""
        cached = self._health_cache.get(provider_name)
        if cached and cached.quota_remaining >= 0:
            return cached.quota_remaining

        config = self._get_provider_config(provider_name)
        daily = self._get_daily_usage_from_db(provider_name)
        return max(0, config.daily_budget_calls - daily.get("calls", 0))

    def get_daily_usage(self, provider_name: str) -> dict:
        """Get today's usage statistics for a provider."""
        return self._get_daily_usage_from_db(provider_name)

    def disable_temporarily(self, provider_name: str, duration_s: int = 300):
        """
        Temporarily disable a provider for the given duration.

        Args:
            provider_name: Provider to disable.
            duration_s: Seconds to disable for (default: 5 minutes).
        """
        self._disabled_until[provider_name] = (
            datetime.now(timezone.utc) + timedelta(seconds=duration_s)
        )
        logger.warning(
            "Provider %s temporarily disabled for %ds",
            provider_name, duration_s
        )

    def record_quota_error(self, provider_name: str):
        """
        Record a runtime quota/rate-limit error (HTTP 429) from a provider.
        Updates the health cache to quota_exhausted and temporarily disables the provider.
        """
        cached = self._health_cache.get(provider_name)
        if cached:
            cached.status = "quota_exhausted"
            cached.error = "Quota exhausted (HTTP 429 received during inference)"
            self._health_cache[provider_name] = cached
            self._persist_health(cached)
        else:
            status = HealthStatus(
                provider_name=provider_name,
                status="quota_exhausted",
                error="Quota exhausted (HTTP 429 received during inference)",
                last_check=datetime.now(timezone.utc).isoformat(),
            )
            self._health_cache[provider_name] = status
            self._persist_health(status)

        # Disable for remainder of the day (86400s) so router bypasses it
        self.disable_temporarily(provider_name, duration_s=86400)
        logger.warning("Provider %s marked quota_exhausted and disabled for 24h", provider_name)


    # -------------------------------------------------------------------
    # Optimal provider selection
    # -------------------------------------------------------------------

    def get_optimal_providers(
        self,
        require_local: bool = False,
        exclude: list[str] = None,
    ) -> list[ProviderInfo]:
        """
        Get a prioritized list of available providers for the current mode.

        Args:
            require_local: If True, only return local providers.
            exclude: Provider names to exclude.

        Returns:
            Sorted list of ProviderInfo (best first).
        """
        exclude = set(exclude or [])
        is_dev = self._settings.mode.development_mode
        is_offline = self._settings.mode.offline_mode

        candidates = []

        for name in self.ALL_PROVIDERS:
            if name in exclude:
                continue

            if is_offline and name in self.CLOUD_PROVIDERS:
                continue

            if require_local and name in self.CLOUD_PROVIDERS:
                continue

            config = self._get_provider_config(name)
            if not config.enabled:
                continue

            if not self.is_available(name):
                continue

            priority = config.priority_dev if is_dev else config.priority_prod
            cached = self._health_cache.get(name)

            info = ProviderInfo(
                name=name,
                status=cached.status if cached else "unknown",
                priority=priority,
                latency_ms=cached.latency_ms if cached else 0.0,
                quota_remaining=self.get_remaining_quota(name),
                is_local=name in self.LOCAL_PROVIDERS,
                model=config.model,
            )
            candidates.append(info)

        # Sort by priority (lower = better)
        candidates.sort(key=lambda p: p.priority)

        return candidates

    def record_call(
        self,
        provider_name: str,
        latency_ms: float,
        success: bool = True,
    ):
        """
        Record a completed call to update provider metrics.

        Args:
            provider_name: Which provider was called.
            latency_ms: Call duration in milliseconds.
            success: Whether the call succeeded.
        """
        cached = self._health_cache.get(provider_name)
        if cached:
            if success:
                # Running average latency
                cached.latency_ms = (cached.latency_ms + latency_ms) / 2
                cached.failure_count = 0
            else:
                cached.failure_count += 1
                # Auto-disable after 3 consecutive failures
                if cached.failure_count >= 3:
                    self.disable_temporarily(provider_name, duration_s=300)
                    logger.warning(
                        "Auto-disabled %s after %d consecutive failures",
                        provider_name, cached.failure_count
                    )

            cached.daily_calls += 1
            self._health_cache[provider_name] = cached

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_provider_config(self, provider_name: str):
        """Get the ProviderConfig for a named provider."""
        return getattr(self._settings.providers, provider_name, None) or \
               type(self._settings.providers).model_fields[provider_name].default

    def _get_daily_usage_from_db(self, provider_name: str) -> dict:
        """Query cloud_usage table for today's stats."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as calls, "
                    "COALESCE(SUM(estimated_cost_usd), 0) as cost, "
                    "COALESCE(SUM(tokens_in), 0) as tokens_in, "
                    "COALESCE(SUM(tokens_out), 0) as tokens_out "
                    "FROM cloud_usage "
                    "WHERE provider = ? AND date(timestamp) = ?",
                    (provider_name, today),
                ).fetchone()

            if row:
                return {
                    "calls": row["calls"],
                    "cost": row["cost"],
                    "tokens_in": row["tokens_in"],
                    "tokens_out": row["tokens_out"],
                }
        except Exception as e:
            logger.debug("Could not query daily usage for %s: %s", provider_name, e)

        return {"calls": 0, "cost": 0.0, "tokens_in": 0, "tokens_out": 0}

    def _persist_health(self, status: HealthStatus):
        """Persist health status to provider_health table."""
        try:
            with self._db._connect() as conn:
                # Check if row exists
                existing = conn.execute(
                    "SELECT id FROM provider_health WHERE provider_name = ?",
                    (status.provider_name,),
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE provider_health SET "
                        "status=?, latency_ms=?, last_check=?, failure_count=?, "
                        "quota_remaining=?, daily_cost_estimate=?, daily_calls=? "
                        "WHERE provider_name=?",
                        (
                            status.status, status.latency_ms, status.last_check,
                            status.failure_count, status.quota_remaining,
                            status.daily_cost, status.daily_calls,
                            status.provider_name,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO provider_health "
                        "(provider_name, status, latency_ms, last_check, "
                        "failure_count, quota_remaining, daily_cost_estimate, daily_calls) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            status.provider_name, status.status, status.latency_ms,
                            status.last_check, status.failure_count,
                            status.quota_remaining, status.daily_cost,
                            status.daily_calls,
                        ),
                    )
        except Exception as e:
            logger.debug("Could not persist health for %s: %s", status.provider_name, e)

    def get_all_status(self) -> dict:
        """Get a summary of all provider statuses for display."""
        result = {}
        for name in self.ALL_PROVIDERS:
            config = self._get_provider_config(name)
            cached = self._health_cache.get(name)

            result[name] = {
                "enabled": config.enabled,
                "status": cached.status if cached else "unchecked",
                "latency_ms": round(cached.latency_ms, 1) if cached else 0,
                "daily_calls": cached.daily_calls if cached else 0,
                "daily_cost": round(cached.daily_cost, 4) if cached else 0,
                "quota_remaining": self.get_remaining_quota(name),
                "is_local": name in self.LOCAL_PROVIDERS,
                "model": config.model,
                "temporarily_disabled": name in self._disabled_until,
            }

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()

    print("=== JARVIS Provider Manager ===\n")

    settings = get_settings()
    db = DatabaseManager(settings=settings)

    # Run Phase 3 migrations first
    from backend.db_migrations import run_migrations
    run_migrations()

    pm = ProviderManager(db=db, settings=settings)

    # Check all providers
    print("Checking all providers...\n")
    results = pm.check_all_health()
    for name, status in results.items():
        icon = "✓" if status.status == "healthy" else "✗"
        print(f"  {icon} {name}: {status.status} ({status.latency_ms:.0f}ms)")
        if status.error:
            print(f"    Error: {status.error}")
        if status.available_models:
            print(f"    Models: {', '.join(status.available_models[:5])}")

    # Get optimal providers
    print("\n--- Optimal providers (current mode) ---")
    optimal = pm.get_optimal_providers()
    for i, p in enumerate(optimal, 1):
        print(f"  {i}. {p.name} (priority={p.priority}, latency={p.latency_ms:.0f}ms)")

    print("\n--- All status ---")
    import json
    print(json.dumps(pm.get_all_status(), indent=2))
