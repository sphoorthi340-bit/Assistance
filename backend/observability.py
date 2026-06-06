"""
Jarvis Phase 3 — Session Observability Module
==============================================
Central observability module for session-level metrics tracking
during a 7-day local-only testing phase.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from backend.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RequestRecord:
    timestamp: str
    routing_time_ms: int
    memory_time_ms: int
    knowledge_time_ms: int
    inference_time_ms: int
    total_time_ms: int
    provider: str
    model: str
    classification: str
    fallback_triggered: bool
    fallback_from: Optional[str]
    fallback_reason: Optional[str]
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FallbackEvent:
    timestamp: str
    from_provider: str
    to_provider: str
    reason: str


class SessionMetrics:
    """Singleton for tracking session-level metrics."""
    
    def __init__(self):
        self.session_start = datetime.now(timezone.utc)
        self.requests: list[RequestRecord] = []
        self.model_usage: dict[tuple[str, str], dict] = {}
        self.fallback_events: list[FallbackEvent] = []
        self.total_requests = 0

    def record_request(
        self,
        routing_time_ms: int,
        memory_time_ms: int,
        knowledge_time_ms: int,
        inference_time_ms: int,
        total_time_ms: int,
        provider: str,
        model: str,
        classification: str,
        fallback_triggered: bool = False,
        fallback_from: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        total_tokens: int = 0
    ):
        """Record a single completed request."""
        record = RequestRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            routing_time_ms=routing_time_ms,
            memory_time_ms=memory_time_ms,
            knowledge_time_ms=knowledge_time_ms,
            inference_time_ms=inference_time_ms,
            total_time_ms=total_time_ms,
            provider=provider,
            model=model,
            classification=classification,
            fallback_triggered=fallback_triggered,
            fallback_from=fallback_from,
            fallback_reason=fallback_reason,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens
        )
        
        self.requests.append(record)
        # Keep only last 100
        if len(self.requests) > 100:
            self.requests.pop(0)
            
        self.total_requests += 1
        
        # Update model usage
        key = (provider, model)
        if key not in self.model_usage:
            self.model_usage[key] = {"count": 0, "total_inference_ms": 0}
        
        self.model_usage[key]["count"] += 1
        self.model_usage[key]["total_inference_ms"] += inference_time_ms

    def record_fallback(self, from_provider: str, to_provider: str, reason: str):
        """Record a fallback event."""
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason
        )
        self.fallback_events.append(event)
        
    def get_perf_summary(self) -> dict:
        """Get summary for /perf command."""
        if not self.requests:
            return {"status": "error", "error": "No requests in current session"}
            
        last = self.requests[-1]
        
        # Calculate averages
        count = len(self.requests)
        avg_routing = sum(r.routing_time_ms for r in self.requests) / count
        avg_memory = sum(r.memory_time_ms for r in self.requests) / count
        avg_knowledge = sum(r.knowledge_time_ms for r in self.requests) / count
        avg_inference = sum(r.inference_time_ms for r in self.requests) / count
        avg_total = sum(r.total_time_ms for r in self.requests) / count
        
        return {
            "status": "success",
            "last_request": {
                "routing_time_ms": last.routing_time_ms,
                "memory_time_ms": last.memory_time_ms,
                "knowledge_time_ms": last.knowledge_time_ms,
                "inference_time_ms": last.inference_time_ms,
                "total_time_ms": last.total_time_ms
            },
            "session_average": {
                "routing_time_ms": int(avg_routing),
                "memory_time_ms": int(avg_memory),
                "knowledge_time_ms": int(avg_knowledge),
                "inference_time_ms": int(avg_inference),
                "total_time_ms": int(avg_total)
            }
        }

    def get_model_stats(self) -> dict:
        """Get stats for /model_stats command."""
        stats = {}
        for (provider, model), data in self.model_usage.items():
            avg_inf = data["total_inference_ms"] / data["count"] if data["count"] > 0 else 0
            
            if provider not in stats:
                stats[provider] = {}
                
            stats[provider][model] = {
                "requests": data["count"],
                "avg_inference_s": round(avg_inf / 1000, 2)
            }
            
        return {"status": "success", "session_usage": stats}

    def get_fallback_log(self) -> list[dict]:
        """Get events for /fallbacks command."""
        return [asdict(e) for e in reversed(self.fallback_events)]

    def export_session_metrics(self) -> dict:
        """Export full session data for /perf_export."""
        # Calculate session duration
        now = datetime.now(timezone.utc)
        duration = now - self.session_start
        
        # Formatting timedelta nicely
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        avg_inference = 0
        if self.requests:
            avg_inference = sum(r.inference_time_ms for r in self.requests) / len(self.requests) / 1000
            
        return {
            "session_start": self.session_start.isoformat(),
            "session_duration": duration_str,
            "total_requests": self.total_requests,
            "average_inference_time_s": round(avg_inference, 2),
            "model_usage": self.get_model_stats().get("session_usage", {}),
            "fallback_events": self.get_fallback_log()
        }

    def get_health_dashboard_metrics(self) -> dict:
        """Get metrics for /health dashboard."""
        avg_resp = 0
        if self.requests:
            avg_resp = sum(r.total_time_ms for r in self.requests) / len(self.requests) / 1000
            
        return {
            "total_requests_today": self.total_requests, # Using session total as proxy for today in testing
            "average_response_s": round(avg_resp, 2),
            "fallback_events": len(self.fallback_events)
        }


# Singleton instance
_session_metrics = None

def get_session_metrics() -> SessionMetrics:
    """Get the global SessionMetrics singleton."""
    global _session_metrics
    if _session_metrics is None:
        _session_metrics = SessionMetrics()
    return _session_metrics
