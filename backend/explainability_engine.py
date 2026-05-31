"""
Jarvis Phase 3 — Explainability Engine
========================================
Provides transparency into system decisions.
Implements the backend logic for /why, /context, /router, and /provider_trace commands.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


class ExplainabilityEngine:
    """
    Records and explains the system's internal decisions.
    Records are stored in the decision_trace SQLite table.
    """
    
    def __init__(self, db: DatabaseManager, settings=None):
        self._db = db
        self._settings = settings or get_settings()
        logger.info("ExplainabilityEngine initialized")

    # -------------------------------------------------------------------
    # Trace Recording
    # -------------------------------------------------------------------

    def record_decision(
        self,
        decision_type: str,
        input_summary: str,
        result: dict,
        trace: dict,
        response_time_ms: int = 0,
        conversation_id: str = None
    ) -> str:
        """
        Records an internal decision to the trace table.
        
        Args:
            decision_type: routing, provider_selection, memory_retrieval, etc.
            input_summary: Brief description of the input that triggered this decision
            result: The outcome (JSON serializable)
            trace: The internal logic/scores that led to the outcome (JSON serializable)
            response_time_ms: How long the decision took
            conversation_id: Optional link to a conversation
            
        Returns:
            The generated UUID for this trace.
        """
        trace_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO decision_trace "
                    "(id, timestamp, decision_type, input_summary, result_json, trace_json, "
                    "response_time_ms, conversation_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        trace_id, timestamp, decision_type, input_summary,
                        json.dumps(result), json.dumps(trace), response_time_ms,
                        conversation_id
                    )
                )
            logger.debug("Recorded decision trace: %s (%s)", trace_id, decision_type)
            return trace_id
        except Exception as e:
            logger.error("Failed to record decision trace: %s", e)
            return ""

    # -------------------------------------------------------------------
    # Explainers (for Slash Commands)
    # -------------------------------------------------------------------

    def explain_last_decision(self) -> dict:
        """
        Provides data for the /why command.
        Retrieves the most recent routing and context assembly decisions.
        """
        data = {
            "status": "success",
            "provider": "unknown",
            "model": "unknown",
            "reason": "No recent routing data found",
            "complexity": "unknown",
            "privacy": "unknown",
            "response_time_ms": 0,
            "confidence": 0.0,
            "context": {}
        }
        
        try:
            with self._db._connect() as conn:
                # Get the last route log
                route_row = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                if route_row:
                    data.update({
                        "provider": route_row["selected_provider"],
                        "model": route_row["selected_model"],
                        "reason": route_row["reason"],
                        "complexity": route_row["complexity"],
                        "privacy": route_row["privacy"],
                        "response_time_ms": route_row["response_time_ms"],
                        "confidence": route_row["confidence"],
                        "timestamp": route_row["timestamp"]
                    })
                    
                # Get the last context assembly trace
                context_row = conn.execute(
                    "SELECT result_json, trace_json FROM decision_trace "
                    "WHERE decision_type = 'context_assembly' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                if context_row:
                    try:
                        res = json.loads(context_row["result_json"])
                        data["context"] = {
                            "memories_retrieved": res.get("memories_count", 0),
                            "knowledge_chunks": res.get("knowledge_count", 0),
                            "state_items": res.get("state_count", 0),
                            "history_turns": res.get("history_count", 0),
                            "total_tokens": res.get("total_tokens", 0)
                        }
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error("Failed to fetch /why data: %s", e)
            data["status"] = "error"
            data["error"] = str(e)
            
        return data

    def explain_context(self) -> dict:
        """
        Provides data for the /context command.
        """
        data = {"status": "error", "error": "No recent context assembly traces found"}
        
        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM decision_trace "
                    "WHERE decision_type = 'context_assembly' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                if row:
                    data = {
                        "status": "success",
                        "timestamp": row["timestamp"],
                        "input_summary": row["input_summary"],
                        "result": json.loads(row["result_json"]),
                        "trace": json.loads(row["trace_json"])
                    }
        except Exception as e:
            logger.error("Failed to fetch /context data: %s", e)
            
        return data

    def explain_routing(self) -> list[dict]:
        """
        Provides data for the /router command (history of routing decisions).
        """
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT 5"
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Failed to fetch /router data: %s", e)
            return []

    def explain_provider_trace(self) -> dict:
        """
        Provides data for the /provider_trace command.
        Combines health, quota, and daily usage data.
        """
        data = {"providers": {}}
        
        try:
            with self._db._connect() as conn:
                # Get provider health
                health_rows = conn.execute(
                    "SELECT * FROM provider_health"
                ).fetchall()
                
                for r in health_rows:
                    data["providers"][r["provider_name"]] = {
                        "status": r["status"],
                        "latency_ms": r["latency_ms"],
                        "quota_remaining": r["quota_remaining"],
                        "daily_calls": r["daily_calls"],
                        "daily_cost": r["daily_cost_estimate"]
                    }
                    
                # Cloud stats summary
                cloud = conn.execute(
                    "SELECT COUNT(*) as calls, SUM(estimated_cost_usd) as cost "
                    "FROM cloud_usage WHERE date(timestamp) = date('now')"
                ).fetchone()
                
                data["cloud_summary_today"] = {
                    "total_calls": cloud["calls"] if cloud else 0,
                    "total_cost": cloud["cost"] if cloud and cloud["cost"] else 0.0
                }
                
        except Exception as e:
            logger.error("Failed to fetch /provider_trace data: %s", e)
            
        return data

    def explain_memory_trace(self, query: str = None) -> dict:
        """
        Provides data for the /memory_trace command.
        """
        data = {"status": "error", "error": "No recent memory retrieval traces found"}
        
        try:
            with self._db._connect() as conn:
                if query:
                    # Search by query
                    row = conn.execute(
                        "SELECT * FROM decision_trace "
                        "WHERE decision_type = 'memory_retrieval' "
                        "AND input_summary LIKE ? "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (f"%{query}%",)
                    ).fetchone()
                else:
                    # Last retrieval
                    row = conn.execute(
                        "SELECT * FROM decision_trace "
                        "WHERE decision_type = 'memory_retrieval' "
                        "ORDER BY timestamp DESC LIMIT 1"
                    ).fetchone()
                    
                if row:
                    data = {
                        "status": "success",
                        "timestamp": row["timestamp"],
                        "query": row["input_summary"],
                        "result": json.loads(row["result_json"]),
                        "trace": json.loads(row["trace_json"]),
                        "response_time_ms": row["response_time_ms"]
                    }
        except Exception as e:
            logger.error("Failed to fetch /memory_trace data: %s", e)
            
        return data

    def get_decision_history(self, limit: int = 20) -> list[dict]:
        """Get raw decision trace history."""
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM decision_trace ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    # -------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------

    def format_for_terminal(self, data: dict, command: str) -> str:
        """
        Formats dict data into a readable string for the terminal.
        """
        if data.get("status") == "error":
            return f"Error: {data.get('error', 'Unknown error')}"
            
        if command == "/why":
            ctx = data.get("context", {})
            return (
                f"Selected Provider: {data.get('provider', 'N/A')}\n"
                f"Selected Model:    {data.get('model', 'N/A')}\n"
                f"Reason:            {data.get('reason', 'N/A')}\n"
                f"Classification:    {data.get('complexity', 'N/A')} complexity, {data.get('privacy', 'N/A')} privacy\n"
                f"Response Time:     {data.get('response_time_ms', 0)} ms\n"
                f"Confidence:        {data.get('confidence', 0.0):.2f}\n"
                f"Memories Included: {ctx.get('memories_retrieved', 0)}\n"
                f"Knowledge Chunks:  {ctx.get('knowledge_chunks', 0)}\n"
                f"State Items:       {ctx.get('state_items', 0)}\n"
                f"Context Tokens:    {ctx.get('total_tokens', 0)}"
            )
            
        elif command == "/provider_trace":
            lines = ["Provider Health & Quota:\n"]
            for name, info in data.get("providers", {}).items():
                lines.append(f"  {name.upper()}:")
                lines.append(f"    Status:  {info['status']}")
                lines.append(f"    Latency: {info['latency_ms']:.1f} ms")
                lines.append(f"    Calls Today: {info['daily_calls']}")
                if info.get("daily_cost", 0) > 0:
                    lines.append(f"    Cost Today:  ${info['daily_cost']:.4f}")
                if info.get("quota_remaining", -1) >= 0:
                    lines.append(f"    Remaining:   {info['quota_remaining']} calls")
                lines.append("")
                
            summary = data.get("cloud_summary_today", {})
            lines.append(f"Cloud Summary: {summary.get('total_calls', 0)} calls, ${summary.get('total_cost', 0):.4f}")
            return "\n".join(lines)
            
        elif command == "/memory_debug":
            res = data.get("result", {})
            trace = data.get("trace", {})
            return (
                f"Memory Retrieval Trace:\n"
                f"Query: {data.get('query', 'N/A')}\n"
                f"Time:  {data.get('timestamp', 'N/A')}\n"
                f"Found: {res.get('memories_count', 0)} items\n"
                f"Time:  {data.get('response_time_ms', 0)} ms\n\n"
                f"Trace Details:\n{json.dumps(trace, indent=2)}"
            )
            
        # Default JSON fallback
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()
    
    settings = get_settings()
    db = DatabaseManager(settings=settings)
    engine = ExplainabilityEngine(db=db, settings=settings)
    
    # Test record
    trace_id = engine.record_decision(
        decision_type="test_decision",
        input_summary="Test input",
        result={"action": "pass"},
        trace={"score": 0.99},
        response_time_ms=10
    )
    print(f"Recorded trace: {trace_id}")
    
    print("\n--- /why format ---")
    print(engine.format_for_terminal(engine.explain_last_decision(), "/why"))
    
    print("\n--- /provider_trace format ---")
    print(engine.format_for_terminal(engine.explain_provider_trace(), "/provider_trace"))
