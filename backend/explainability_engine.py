"""
Jarvis Phase 3 — Explainability Engine
========================================
Provides transparency into system decisions.
Implements the backend logic for /why, /context, /router, /provider_trace,
/route, /trace, /last_context, /decision_history, /knowledge_debug, and /memory_debug commands.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Classification label mapping
# ---------------------------------------------------------------------------

_CLASSIFICATION_MAP = {
    # complexity → label
    "low": "Fast",
    "medium": "Reasoning",
    "high": "Reasoning",
    "extreme": "Reasoning",
}

_REASON_KEYWORD_LABELS = {
    "coding": "Coding",
    "math": "Math",
}


def _derive_classification_label(complexity: str, reason: str) -> str:
    """
    Derive a human-friendly classification label from the complexity level
    and the free-text reason field logged by the router.
    """
    reason_lower = (reason or "").lower()
    for keyword, label in _REASON_KEYWORD_LABELS.items():
        if keyword in reason_lower:
            return label
    return _CLASSIFICATION_MAP.get(complexity, "Reasoning")


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
            "context": {},
            # New fields
            "question": "",
            "classification_label": "unknown",
            "memories_injected": 0,
            "knowledge_chunks": 0,
            "context_tokens": 0,
            "inference_time_ms": 0,
        }
        
        try:
            with self._db._connect() as conn:
                # Get the last route log
                route_row = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                if route_row:
                    complexity = route_row["complexity"]
                    reason = route_row["reason"]
                    data.update({
                        "provider": route_row["selected_provider"],
                        "model": route_row["selected_model"],
                        "reason": reason,
                        "complexity": complexity,
                        "privacy": route_row["privacy"],
                        "response_time_ms": route_row["response_time_ms"],
                        "confidence": route_row["confidence"],
                        "timestamp": route_row["timestamp"],
                        # New fields from route_logs
                        "question": route_row["query_summary"] or "",
                        "classification_label": _derive_classification_label(complexity, reason),
                        "inference_time_ms": route_row["response_time_ms"] or 0,
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
                        memories_count = res.get("memories_count", 0)
                        knowledge_count = res.get("knowledge_count", 0)
                        total_tokens = res.get("total_tokens", 0)
                        data["context"] = {
                            "memories_retrieved": memories_count,
                            "knowledge_chunks": knowledge_count,
                            "state_items": res.get("state_count", 0),
                            "history_turns": res.get("history_count", 0),
                            "total_tokens": total_tokens,
                        }
                        # Populate top-level convenience fields
                        data["memories_injected"] = memories_count
                        data["knowledge_chunks"] = knowledge_count
                        data["context_tokens"] = total_tokens
                    except json.JSONDecodeError:
                        pass

                # If question is still empty, try the routing decision_trace
                if not data["question"]:
                    routing_trace = conn.execute(
                        "SELECT input_summary FROM decision_trace "
                        "WHERE decision_type = 'routing' "
                        "ORDER BY timestamp DESC LIMIT 1"
                    ).fetchone()
                    if routing_trace:
                        data["question"] = routing_trace["input_summary"] or ""

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
        Provides data for the /memory_debug command.
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
                    result = json.loads(row["result_json"])
                    trace = json.loads(row["trace_json"])
                    data = {
                        "status": "success",
                        "timestamp": row["timestamp"],
                        "query": row["input_summary"],
                        "result": result,
                        "trace": trace,
                        "response_time_ms": row["response_time_ms"],
                        # New fields
                        "importance_scores": trace.get("importance_scores", []),
                        "injection_tokens": result.get("injection_tokens", 0),
                        "used_in_context": result.get("used_in_context", False),
                    }
        except Exception as e:
            logger.error("Failed to fetch /memory_debug data: %s", e)
            
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
    # New Explainers (Phase 3 extensions)
    # -------------------------------------------------------------------

    def explain_route_diagnostics(self) -> dict:
        """
        Provides data for the /route diagnostics command.
        Returns classification, confidence, primary/fallback/selected provider info
        from the most recent route_log entry.
        """
        data = {
            "status": "error",
            "error": "No recent routing data found",
        }

        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if row:
                    complexity = row["complexity"]
                    reason = row["reason"] or ""
                    classification = _derive_classification_label(complexity, reason)

                    # Try to extract fallback info from reason text
                    fallback_provider = None
                    fallback_model = None
                    if row["fallback_used"]:
                        # Reason often contains "(Cloud failed, fallback to local: ...)"
                        # or "Fallback after error: ... | ..."
                        import re
                        fb_match = re.search(
                            r"fallback.*?(\w+)/(\S+)", reason, re.IGNORECASE
                        )
                        if fb_match:
                            fallback_provider = fb_match.group(1)
                            fallback_model = fb_match.group(2)

                    data = {
                        "status": "success",
                        "classification": classification,
                        "complexity": complexity,
                        "privacy": row["privacy"],
                        "latency_priority": row["latency_priority"],
                        "cost_priority": row["cost_priority"],
                        "confidence": row["confidence"],
                        "primary_provider": row["selected_provider"],
                        "primary_model": row["selected_model"],
                        "fallback_provider": fallback_provider,
                        "fallback_model": fallback_model,
                        "fallback_used": bool(row["fallback_used"]),
                        "selected_provider": row["selected_provider"],
                        "reason": reason,
                        "timestamp": row["timestamp"],
                    }

        except Exception as e:
            logger.error("Failed to fetch /route diagnostics: %s", e)
            data["error"] = str(e)

        return data

    def explain_knowledge_trace(self) -> dict:
        """
        Provides data for the /knowledge_debug command.
        Queries the decision_trace table for decision_type='knowledge_retrieval'.

        Note: 'knowledge_retrieval' may not exist in the CHECK constraint yet.
        The method handles this gracefully by returning an informative empty result.
        """
        data = {
            "status": "error",
            "error": "No recent knowledge retrieval traces found",
        }

        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM decision_trace "
                    "WHERE decision_type = 'knowledge_retrieval' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if row:
                    result = json.loads(row["result_json"])
                    trace = json.loads(row["trace_json"])
                    data = {
                        "status": "success",
                        "timestamp": row["timestamp"],
                        "query": row["input_summary"],
                        "chunks_retrieved": result.get("chunks_count", 0),
                        "source_names": result.get("source_names", []),
                        "similarity_scores": trace.get("similarity_scores", []),
                        "injection_tokens": result.get("injection_tokens", 0),
                        "result": result,
                        "trace": trace,
                        "response_time_ms": row["response_time_ms"],
                    }

        except Exception as e:
            logger.error("Failed to fetch /knowledge_debug data: %s", e)
            data["error"] = str(e)

        return data

    def explain_unified_trace(self) -> dict:
        """
        Provides data for the /trace (unified) command.
        Combines the last routing decision, memory retrieval, knowledge
        retrieval, and context assembly into a single diagnostic dict.
        """
        data: dict = {
            "status": "success",
            "routing": None,
            "memory_retrieval": None,
            "knowledge_retrieval": None,
            "context_assembly": None,
        }

        try:
            with self._db._connect() as conn:
                # --- Last routing decision (from route_logs) ---
                route_row = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if route_row:
                    data["routing"] = {
                        "provider": route_row["selected_provider"],
                        "model": route_row["selected_model"],
                        "complexity": route_row["complexity"],
                        "privacy": route_row["privacy"],
                        "confidence": route_row["confidence"],
                        "reason": route_row["reason"],
                        "response_time_ms": route_row["response_time_ms"],
                        "timestamp": route_row["timestamp"],
                    }

                # --- Helper to fetch latest decision_trace by type ---
                def _last_trace(dtype: str) -> Optional[dict]:
                    row = conn.execute(
                        "SELECT * FROM decision_trace "
                        "WHERE decision_type = ? "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (dtype,)
                    ).fetchone()
                    if row:
                        return {
                            "input_summary": row["input_summary"],
                            "result": json.loads(row["result_json"]),
                            "trace": json.loads(row["trace_json"]),
                            "response_time_ms": row["response_time_ms"],
                            "timestamp": row["timestamp"],
                        }
                    return None

                data["memory_retrieval"] = _last_trace("memory_retrieval")
                data["knowledge_retrieval"] = _last_trace("knowledge_retrieval")
                data["context_assembly"] = _last_trace("context_assembly")

        except Exception as e:
            logger.error("Failed to fetch /trace unified data: %s", e)
            data["status"] = "error"
            data["error"] = str(e)

        return data

    def explain_last_context(self) -> dict:
        """
        Provides data for the /last_context command.
        Returns the last context_assembly decision trace with
        conversation tokens, memory items, knowledge items, and totals.
        """
        data = {
            "status": "error",
            "error": "No recent context assembly traces found",
        }

        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM decision_trace "
                    "WHERE decision_type = 'context_assembly' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if row:
                    result = json.loads(row["result_json"])
                    trace = json.loads(row["trace_json"])
                    
                    conv_tokens = result.get("conversation_tokens", 0)
                    mem_tokens = result.get("memory_tokens", 0)
                    know_tokens = result.get("knowledge_tokens", 0)
                    state_tokens = result.get("state_tokens", 0)
                    total_tokens = result.get("total_tokens", 0)
                    system_tokens = total_tokens - (conv_tokens + mem_tokens + know_tokens + state_tokens)
                    
                    data = {
                        "status": "success",
                        "timestamp": row["timestamp"],
                        "conversation_tokens": conv_tokens,
                        "memory_tokens": mem_tokens,
                        "knowledge_tokens": know_tokens,
                        "state_tokens": state_tokens,
                        "system_tokens": system_tokens,
                        "memory_items": trace.get("memory_items", []),
                        "knowledge_items": trace.get("knowledge_items", []),
                        "total_context_tokens": total_tokens,
                        "result": result,
                        "trace": trace,
                    }

        except Exception as e:
            logger.error("Failed to fetch /last_context data: %s", e)
            data["error"] = str(e)

        return data

    def get_routing_history(self, limit: int = 10) -> list[dict]:
        """
        Provides data for the /decision_history command.
        Returns recent route_logs ordered by timestamp DESC.
        """
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM route_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()

                history = []
                for r in rows:
                    history.append({
                        "input_summary": r["query_summary"] or "",
                        "route": _derive_classification_label(
                            r["complexity"], r["reason"] or ""
                        ),
                        "provider": r["selected_provider"],
                        "model": r["selected_model"],
                        "timestamp": r["timestamp"],
                    })
                return history

        except Exception as e:
            logger.error("Failed to fetch /decision_history data: %s", e)
            return []

    # -------------------------------------------------------------------
    # Response Footer Formatter
    # -------------------------------------------------------------------

    def format_response_footer(
        self,
        provider: str,
        model: str,
        classification: str,
        memory_count: int,
        knowledge_count: int,
        fallback_triggered: bool,
        inference_time_ms: float,
        total_time_ms: float,
    ) -> str:
        """
        Formats a compact response footer summarising the execution.
        """
        inference_s = inference_time_ms / 1000.0
        total_s = total_time_ms / 1000.0
        fallback_str = "Yes" if fallback_triggered else "No"

        return (
            "━━━━━━━━━━━━━━━━━━\n"
            f"Provider: {provider}\n"
            f"Model: {model}\n"
            f"Route: {classification}\n"
            "\n"
            f"Memory Used: {memory_count}\n"
            f"Knowledge Chunks: {knowledge_count}\n"
            f"Fallback Triggered: {fallback_str}\n"
            "\n"
            f"Inference Time: {inference_s:.1f}s\n"
            f"Total Time: {total_s:.1f}s\n"
            "━━━━━━━━━━━━━━━━━━"
        )

    # -------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------

    def format_for_terminal(self, data: dict | list, command: str = "") -> str:
        """
        Formats dict data into a readable string for the terminal.
        """
        # Handle error status for dict payloads
        if isinstance(data, dict) and data.get("status") == "error":
            return f"Error: {data.get('error', 'Unknown error')}"
            
        if command == "/why":
            ctx = data.get("context", {})
            return (
                f"Selected Provider: {data.get('provider', 'N/A')}\n"
                f"Selected Model:    {data.get('model', 'N/A')}\n"
                f"Reason:            {data.get('reason', 'N/A')}\n"
                f"Classification:    {data.get('complexity', 'N/A')} complexity, {data.get('privacy', 'N/A')} privacy\n"
                f"Label:             {data.get('classification_label', 'N/A')}\n"
                f"Question:          {data.get('question', 'N/A')}\n"
                f"Response Time:     {data.get('response_time_ms', 0)} ms\n"
                f"Confidence:        {data.get('confidence', 0.0):.2f}\n"
                f"Memories Included: {ctx.get('memories_retrieved', 0)}\n"
                f"Knowledge Chunks:  {ctx.get('knowledge_chunks', 0)}\n"
                f"State Items:       {ctx.get('state_items', 0)}\n"
                f"Context Tokens:    {ctx.get('total_tokens', 0)}\n"
                f"Inference Time:    {data.get('inference_time_ms', 0)} ms"
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
            importance = data.get("importance_scores", [])
            importance_str = ", ".join(f"{s:.3f}" for s in importance) if importance else "N/A"
            return (
                f"Memory Retrieval Trace:\n"
                f"Query: {data.get('query', 'N/A')}\n"
                f"Time:  {data.get('timestamp', 'N/A')}\n"
                f"Found: {res.get('memories_count', 0)} items\n"
                f"Time:  {data.get('response_time_ms', 0)} ms\n"
                f"Importance Scores: {importance_str}\n"
                f"Injection Tokens:  {data.get('injection_tokens', 0)}\n"
                f"Used in Context:   {'Yes' if data.get('used_in_context') else 'No'}\n\n"
                f"Trace Details:\n{json.dumps(trace, indent=2)}"
            )

        elif command == "/route":
            return (
                f"Route Diagnostics:\n"
                f"Classification:    {data.get('classification', 'N/A')}\n"
                f"Complexity:        {data.get('complexity', 'N/A')}\n"
                f"Privacy:           {data.get('privacy', 'N/A')}\n"
                f"Latency Priority:  {data.get('latency_priority', 'N/A')}\n"
                f"Cost Priority:     {data.get('cost_priority', 'N/A')}\n"
                f"Confidence:        {data.get('confidence', 0.0):.2f}\n"
                f"Primary Provider:  {data.get('primary_provider', 'N/A')}\n"
                f"Primary Model:     {data.get('primary_model', 'N/A')}\n"
                f"Fallback Provider: {data.get('fallback_provider') or 'N/A'}\n"
                f"Fallback Model:    {data.get('fallback_model') or 'N/A'}\n"
                f"Fallback Used:     {'Yes' if data.get('fallback_used') else 'No'}\n"
                f"Selected Provider: {data.get('selected_provider', 'N/A')}\n"
                f"Reason:            {data.get('reason', 'N/A')}"
            )

        elif command == "/trace":
            lines = ["Unified Trace:\n"]

            routing = data.get("routing")
            if routing:
                lines.append("  ROUTING:")
                lines.append(f"    Provider:  {routing.get('provider', 'N/A')}")
                lines.append(f"    Model:     {routing.get('model', 'N/A')}")
                lines.append(f"    Complexity:{routing.get('complexity', 'N/A')}")
                lines.append(f"    Confidence:{routing.get('confidence', 0.0):.2f}")
                lines.append(f"    Reason:    {routing.get('reason', 'N/A')}")
                lines.append(f"    Time:      {routing.get('response_time_ms', 0)} ms")
                lines.append(f"    Timestamp: {routing.get('timestamp', 'N/A')}")
            else:
                lines.append("  ROUTING: No data")

            lines.append("")

            mem = data.get("memory_retrieval")
            if mem:
                lines.append("  MEMORY RETRIEVAL:")
                lines.append(f"    Input:     {mem.get('input_summary', 'N/A')}")
                mem_res = mem.get("result", {})
                lines.append(f"    Count:     {mem_res.get('memories_count', 0)}")
                lines.append(f"    Time:      {mem.get('response_time_ms', 0)} ms")
                lines.append(f"    Timestamp: {mem.get('timestamp', 'N/A')}")
            else:
                lines.append("  MEMORY RETRIEVAL: No data")

            lines.append("")

            know = data.get("knowledge_retrieval")
            if know:
                lines.append("  KNOWLEDGE RETRIEVAL:")
                lines.append(f"    Input:     {know.get('input_summary', 'N/A')}")
                know_res = know.get("result", {})
                lines.append(f"    Chunks:    {know_res.get('chunks_count', 0)}")
                lines.append(f"    Time:      {know.get('response_time_ms', 0)} ms")
                lines.append(f"    Timestamp: {know.get('timestamp', 'N/A')}")
            else:
                lines.append("  KNOWLEDGE RETRIEVAL: No data")

            lines.append("")

            ctx = data.get("context_assembly")
            if ctx:
                lines.append("  CONTEXT ASSEMBLY:")
                lines.append(f"    Input:     {ctx.get('input_summary', 'N/A')}")
                ctx_res = ctx.get("result", {})
                lines.append(f"    Tokens:    {ctx_res.get('total_tokens', 0)}")
                lines.append(f"    Time:      {ctx.get('response_time_ms', 0)} ms")
                lines.append(f"    Timestamp: {ctx.get('timestamp', 'N/A')}")
            else:
                lines.append("  CONTEXT ASSEMBLY: No data")

            return "\n".join(lines)

        elif command == "/last_context":
            mem_items = data.get("memory_items", [])
            know_items = data.get("knowledge_items", [])

            lines = [
                "Last Context Assembly:\n",
                f"  Conversation Tokens: {data.get('conversation_tokens', 0)}",
                f"  Memory Tokens:       {data.get('memory_tokens', 0)}",
                f"  Knowledge Tokens:    {data.get('knowledge_tokens', 0)}",
                f"  State Tokens:        {data.get('state_tokens', 0)}",
                f"  System Tokens:       {data.get('system_tokens', 0)}",
                "",
                f"  Total Context Tokens:{data.get('total_context_tokens', 0)}",
                "",
                f"  Memory Items ({len(mem_items)}):",
            ]
            for i, item in enumerate(mem_items, 1):
                snippet = item if isinstance(item, str) else json.dumps(item)
                lines.append(f"    {i}. {snippet[:120]}")
            if not mem_items:
                lines.append("    (none)")

            lines.append("")
            lines.append(f"  Knowledge Items ({len(know_items)}):")
            for i, item in enumerate(know_items, 1):
                snippet = item if isinstance(item, str) else json.dumps(item)
                lines.append(f"    {i}. {snippet[:120]}")
            if not know_items:
                lines.append("    (none)")

            return "\n".join(lines)

        elif command == "/decision_history":
            if not data:
                return "No routing history found."
            lines = ["Recent Routing History:\n"]
            for i, entry in enumerate(data, 1):
                summary = entry.get("input_summary", "")
                summary_display = (summary[:60] + "…") if len(summary) > 60 else summary
                lines.append(
                    f"  {i}. [{entry.get('route', '?')}] "
                    f"{entry.get('provider', '?')}/{entry.get('model', '?')}  "
                    f"\"{summary_display}\"  "
                    f"({entry.get('timestamp', 'N/A')})"
                )
            return "\n".join(lines)

        elif command == "/knowledge_debug":
            scores = data.get("similarity_scores", [])
            scores_str = ", ".join(f"{s:.3f}" for s in scores) if scores else "N/A"
            sources = data.get("source_names", [])
            sources_str = ", ".join(sources) if sources else "N/A"
            return (
                f"Knowledge Retrieval Trace:\n"
                f"Query:             {data.get('query', 'N/A')}\n"
                f"Time:              {data.get('timestamp', 'N/A')}\n"
                f"Chunks Retrieved:  {data.get('chunks_retrieved', 0)}\n"
                f"Sources:           {sources_str}\n"
                f"Similarity Scores: {scores_str}\n"
                f"Injection Tokens:  {data.get('injection_tokens', 0)}\n"
                f"Response Time:     {data.get('response_time_ms', 0)} ms"
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

    print("\n--- /route format ---")
    print(engine.format_for_terminal(engine.explain_route_diagnostics(), "/route"))

    print("\n--- /trace format ---")
    print(engine.format_for_terminal(engine.explain_unified_trace(), "/trace"))

    print("\n--- /last_context format ---")
    print(engine.format_for_terminal(engine.explain_last_context(), "/last_context"))

    print("\n--- /decision_history format ---")
    print(engine.format_for_terminal(engine.get_routing_history(), "/decision_history"))

    print("\n--- /knowledge_debug format ---")
    print(engine.format_for_terminal(engine.explain_knowledge_trace(), "/knowledge_debug"))

    print("\n--- Response footer ---")
    print(engine.format_response_footer(
        provider="LM Studio", model="qwen2.5-7b-instruct",
        classification="Reasoning", memory_count=2, knowledge_count=3,
        fallback_triggered=False, inference_time_ms=2800, total_time_ms=3100
    ))
