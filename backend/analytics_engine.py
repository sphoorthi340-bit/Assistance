"""
Jarvis Phase 3 — Analytics Engine
===================================
Extends the existing AnalyticsManager with complex reporting,
LLM-generated insights, and cross-domain correlation detection.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from state.analytics_manager import AnalyticsManager
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class WeeklyReport:
    week_start: str
    week_end: str
    habit_summary: dict
    academic_summary: dict
    project_summary: dict
    behavioral_insight: str
    correlation_alerts: list[str]
    generated_at: str


# ---------------------------------------------------------------------------
# Analytics Engine
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """
    Advanced analytics that wraps the deterministic AnalyticsManager
    to provide LLM-powered insights and complex correlations.
    """
    
    def __init__(self, db: DatabaseManager, analytics_manager: AnalyticsManager, settings=None):
        self._db = db
        self._am = analytics_manager
        self._settings = settings or get_settings()
        
        # We use a direct ollama client for insights to avoid routing overhead
        from backend.llm import OllamaClient
        self._llm = OllamaClient(settings=self._settings)
        
        logger.info("AnalyticsEngine initialized")

    # -------------------------------------------------------------------
    # Advanced Metrics
    # -------------------------------------------------------------------

    def get_habit_streaks_and_failures(self) -> list[dict]:
        """
        Enhances basic habit stats with specific dates missed in the last 30 days.
        """
        stats = self._am.get_all_habit_stats()
        
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        for habit_stat in stats:
            habit_id = habit_stat["habit_id"]
            
            # Find missed dates
            try:
                with self._db._connect() as conn:
                    # Get all dates the habit was completed
                    completed = conn.execute(
                        "SELECT date FROM habit_logs WHERE habit_id = ? AND completed = 1 AND date >= ?",
                        (habit_id, start_date)
                    ).fetchall()
                    completed_dates = {row["date"] for row in completed}
                    
                    # Generate all dates in last 30 days
                    missed = []
                    for i in range(30):
                        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                        if d not in completed_dates:
                            missed.append(d)
                            
                    habit_stat["failure_days"] = missed
            except Exception as e:
                logger.error("Failed to get failure days for habit %s: %s", habit_id, e)
                habit_stat["failure_days"] = []
                
        return stats

    def get_study_session_frequency(self, days: int = 30) -> dict:
        """
        Analyzes conversation messages for academic/study sessions.
        """
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        keywords = ["study", "exam", "lecture", "homework", "assignment", "revision", "chapter", "textbook", "read"]
        
        result = {
            "sessions_by_date": {},
            "topics": {}
        }
        
        try:
            with self._db._connect() as conn:
                # SQLite doesn't have great regex, so we'll fetch all recent user messages and filter in python
                messages = conn.execute(
                    "SELECT timestamp, content FROM messages "
                    "WHERE role = 'user' AND timestamp >= ?",
                    (start_date,)
                ).fetchall()
                
                for msg in messages:
                    content_lower = msg["content"].lower()
                    date_str = msg["timestamp"][:10]  # Just YYYY-MM-DD
                    
                    # Check for study session
                    for kw in keywords:
                        if kw in content_lower:
                            # Record session count
                            result["sessions_by_date"][date_str] = result["sessions_by_date"].get(date_str, 0) + 1
                            # Record topic
                            result["topics"][kw] = result["topics"].get(kw, 0) + 1
                            
        except Exception as e:
            logger.error("Failed to get study frequency: %s", e)
            
        return result

    def get_goal_completion_time(self) -> list[dict]:
        """
        Analyzes how long it takes to complete goals.
        """
        results = []
        try:
            with self._db._connect() as conn:
                goals = conn.execute(
                    "SELECT id, title, created_at, updated_at "
                    "FROM goals WHERE status = 'completed'"
                ).fetchall()
                
                for g in goals:
                    try:
                        # Clean up timestamps for parsing
                        c_str = g["created_at"].replace('Z', '+00:00')
                        u_str = g["updated_at"].replace('Z', '+00:00')
                        
                        created = datetime.fromisoformat(c_str)
                        completed = datetime.fromisoformat(u_str)
                        
                        days = (completed - created).days
                        
                        results.append({
                            "title": g["title"],
                            "created_at": g["created_at"],
                            "completed_at": g["updated_at"],
                            "days_to_complete": max(0, days)
                        })
                    except ValueError:
                        continue # Skip malformed dates
        except Exception as e:
            logger.error("Failed to get goal completion times: %s", e)
            
        return results

    def get_daily_conversation_volume(self, days: int = 30) -> list[dict]:
        """
        Analyzes Jarvis usage volume over time.
        """
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        results = []
        
        try:
            with self._db._connect() as conn:
                # Use substr to extract YYYY-MM-DD from ISO format timestamp
                rows = conn.execute(
                    "SELECT substr(timestamp, 1, 10) as day, COUNT(*) as count "
                    "FROM messages "
                    "WHERE timestamp >= ? "
                    "GROUP BY day ORDER BY day DESC",
                    (start_date,)
                ).fetchall()
                
                results = [{"date": r["day"], "message_count": r["count"]} for r in rows]
        except Exception as e:
            logger.error("Failed to get conversation volume: %s", e)
            
        return results

    def detect_correlations(self) -> list[str]:
        """
        Heuristic-based correlation detection across domains.
        """
        alerts = []
        
        # 1. Burnout detection: High JARVIS usage but low habit completion
        vol = self.get_daily_conversation_volume(days=7)
        habits = self._am.get_all_habit_stats()
        
        if vol and habits:
            avg_msgs_7d = sum(d["message_count"] for d in vol) / len(vol)
            
            # If habit completion is dropping but conversation volume is high
            declining = [h for h in habits if h.get("completion_rate_30d", 0) > 20 and self._am.get_habit_completion_rate(h["habit_id"], 7) < h.get("completion_rate_30d", 0) * 0.5]
            
            if len(declining) > 0 and avg_msgs_7d > 10:
                alerts.append(f"Burnout Risk: JARVIS usage is high (avg {avg_msgs_7d:.1f} msgs/day) but {len(declining)} habits are dropping off.")

        # 2. Academic drop-off: Missed study sessions correlate with approaching exams
        study_freq = self.get_study_session_frequency(14)
        sessions_last_7 = sum(count for d, count in study_freq["sessions_by_date"].items() if (datetime.now() - datetime.strptime(d, "%Y-%m-%d")).days <= 7)
        sessions_prev_7 = sum(count for d, count in study_freq["sessions_by_date"].items() if 7 < (datetime.now() - datetime.strptime(d, "%Y-%m-%d")).days <= 14)
        
        if sessions_prev_7 > 0 and sessions_last_7 == 0:
            alerts.append("Academic Drop-off: You had study sessions last week, but none in the past 7 days.")
            
        return alerts

    # -------------------------------------------------------------------
    # Weekly Reporting
    # -------------------------------------------------------------------

    def generate_weekly_report(self) -> WeeklyReport:
        """
        Compiles the weekly report and generates an LLM insight.
        """
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = now.strftime("%Y-%m-%d")
        
        logger.info("Generating weekly report for %s to %s", week_start, week_end)
        
        # Gather data
        habit_stats = self._am.get_all_habit_stats()
        study_stats = self.get_study_session_frequency(7)
        project_health = self._am.get_project_health()
        correlations = self.detect_correlations()
        
        # Prepare summaries
        habit_summary = {
            h["name"]: {
                "rate": self._am.get_habit_completion_rate(h["habit_id"], 7),
                "streak": h["current_streak"]
            } for h in habit_stats
        }
        
        academic_summary = {
            "total_sessions": sum(study_stats["sessions_by_date"].values()),
            "topics": study_stats["topics"]
        }
        
        project_summary = {
            p["name"]: {
                "status": p["health_status"],
                "overdue": p["overdue_task_count"]
            } for p in project_health
        }
        
        # Generate insight
        data_for_llm = {
            "habits": habit_summary,
            "academics": academic_summary,
            "projects": project_summary,
            "correlations": correlations
        }
        
        insight = self._generate_behavioral_insight(data_for_llm)
        
        report = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            habit_summary=habit_summary,
            academic_summary=academic_summary,
            project_summary=project_summary,
            behavioral_insight=insight,
            correlation_alerts=correlations,
            generated_at=now.isoformat()
        )
        
        # Persist
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO weekly_reports (week_start, week_end, report_json, insight, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (week_start, week_end, json.dumps(asdict(report)), insight, now.isoformat())
                )
            logger.info("Weekly report saved successfully")
        except Exception as e:
            logger.error("Failed to save weekly report: %s", e)
            
        return report

    def _generate_behavioral_insight(self, data: dict) -> str:
        """Uses the local fast model to generate an insight."""
        prompt = f"""As a cognitive analytics engine, provide ONE key behavioral insight based on this week's data.
Keep it strictly under 3 sentences. Be analytical and supportive, identifying patterns. Do NOT use pleasantries.

Data:
{json.dumps(data, indent=2)}
"""
        try:
            original_model = self._llm._model
            # Resolve alias to plain string for Ollama
            fast_model = self._settings.local_models.get_model_for("fast", "ollama") or "llama3.2:1b"
            self._llm._model = fast_model
            
            resp = self._llm.chat([{"role": "user", "content": prompt}])
            
            self._llm._model = original_model
            return resp.content.strip()
        except Exception as e:
            logger.warning("LLM insight generation failed, using deterministic fallback: %s", e)
            
            # Deterministic fallback
            alerts = data.get("correlations", [])
            if alerts:
                return f"Notice: {alerts[0]}"
                
            habits = data.get("habits", {})
            good_habits = sum(1 for h in habits.values() if h["rate"] > 70)
            
            return f"You maintained >70% completion on {good_habits} out of {len(habits)} habits this week. Keep an eye on your active projects to maintain momentum."

    def get_latest_report(self) -> Optional[WeeklyReport]:
        """Fetch the most recent weekly report."""
        try:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT report_json FROM weekly_reports ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                
                if row:
                    data = json.loads(row["report_json"])
                    return WeeklyReport(**data)
        except Exception as e:
            logger.error("Failed to get latest report: %s", e)
        return None
        
    def get_report_history(self, limit: int = 10) -> list[dict]:
        """Fetch historical reports."""
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM weekly_reports ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Failed to get report history: %s", e)
            return []


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()
    
    settings = get_settings()
    db = DatabaseManager(settings=settings)
    am = AnalyticsManager(db=db)
    
    engine = AnalyticsEngine(db=db, analytics_manager=am, settings=settings)
    
    print("=== Testing Analytics Engine ===")
    
    print("\nDetecting correlations...")
    correlations = engine.detect_correlations()
    for c in correlations:
        print(f"  - {c}")
        
    print("\nGenerating weekly report...")
    report = engine.generate_weekly_report()
    print(f"\nWeek: {report.week_start} to {report.week_end}")
    print(f"Insight: {report.behavioral_insight}")
    
    print("\nLatest Report from DB:")
    latest = engine.get_latest_report()
    if latest:
        print(f"Loaded report from {latest.generated_at}")
    else:
        print("No report found in DB.")
