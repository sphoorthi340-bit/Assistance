"""
Jarvis Phase 3 — Proactive Layer
==================================
Manages proactive notifications, morning briefings, evening nudges,
and the persistent inbox system.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

from backend.database import DatabaseManager
from backend.logger import get_logger
from state.analytics_manager import AnalyticsManager
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class InboxItem:
    id: str
    type: str               # morning_briefing, evening_nudge, reminder, alert, notification, weekly_report
    title: str
    content: str
    timestamp: str          # ISO 8601
    read_status: bool = False
    priority: str = 'normal'  # low, normal, high, urgent
    source: str = 'system'


# ---------------------------------------------------------------------------
# Proactive Layer
# ---------------------------------------------------------------------------

class ProactiveLayer:
    """
    Generates intelligent proactive messages and manages the inbox.
    """
    
    def __init__(self, db: DatabaseManager, analytics_manager: AnalyticsManager, settings=None):
        self._db = db
        self._am = analytics_manager
        self._settings = settings or get_settings()
        
        # Local fast model for summaries
        from backend.llm import OllamaClient
        self._llm = OllamaClient(settings=self._settings)
        
        logger.info("ProactiveLayer initialized")

    # -------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------

    def morning_briefing(self) -> InboxItem:
        """
        Generates the daily morning briefing.
        Includes habits due today, upcoming deadlines, stale goals, and misses from yesterday.
        """
        logger.info("Generating morning briefing")
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        seven_days = (now + timedelta(days=7)).strftime("%Y-%m-%d")
        five_days_ago = (now - timedelta(days=self._settings.proactive.stale_goal_days)).isoformat()
        
        data_blocks = []
        
        try:
            with self._db._connect() as conn:
                # 1. Habits due today
                habits = conn.execute("SELECT name FROM habits WHERE active = 1").fetchall()
                if habits:
                    names = [h["name"] for h in habits]
                    data_blocks.append(f"Habits due today: {', '.join(names)}")
                
                # 2. Upcoming deadlines
                deadlines = conn.execute(
                    "SELECT title, deadline FROM goals "
                    "WHERE status = 'active' AND deadline IS NOT NULL "
                    "AND deadline >= ? AND deadline <= ?",
                    (today, seven_days)
                ).fetchall()
                if deadlines:
                    data_blocks.append("Upcoming goal deadlines:")
                    for d in deadlines:
                        data_blocks.append(f"  - {d['title']} ({d['deadline'][:10]})")
                        
                tasks = conn.execute(
                    "SELECT title, due_date FROM project_tasks "
                    "WHERE status != 'completed' AND due_date IS NOT NULL "
                    "AND due_date >= ? AND due_date <= ?",
                    (today, seven_days)
                ).fetchall()
                if tasks:
                    data_blocks.append("Upcoming task deadlines:")
                    for t in tasks:
                        data_blocks.append(f"  - {t['title']} ({t['due_date'][:10]})")
                
                # 3. Stale goals
                stale = conn.execute(
                    "SELECT title FROM goals "
                    "WHERE status = 'active' AND updated_at < ?",
                    (five_days_ago,)
                ).fetchall()
                if stale:
                    names = [s["title"] for s in stale]
                    data_blocks.append(f"Stale goals (no activity in {self._settings.proactive.stale_goal_days}+ days): {', '.join(names)}")
                
                # 4. Missed yesterday
                logged_yesterday = conn.execute(
                    "SELECT h.name FROM habit_logs l "
                    "JOIN habits h ON l.habit_id = h.id "
                    "WHERE l.date = ? AND l.completed = 1",
                    (yesterday,)
                ).fetchall()
                logged_names = {l["name"] for l in logged_yesterday}
                
                if habits:
                    missed = [h["name"] for h in habits if h["name"] not in logged_names]
                    if missed:
                        data_blocks.append(f"Habits missed yesterday: {', '.join(missed)}")
                        
        except Exception as e:
            logger.error("Failed to gather morning briefing data: %s", e)
            data_blocks.append("Error gathering briefing data.")

        # Combine data
        raw_text = "\n".join(data_blocks) if data_blocks else "No pressing items for today."
        
        # Try to summarize
        final_content = self._try_llm_summary(raw_text, "morning briefing")
        
        item = InboxItem(
            id=str(uuid.uuid4()),
            type="morning_briefing",
            title=f"Morning Briefing: {now.strftime('%A, %B %d')}",
            content=final_content,
            timestamp=now.isoformat(),
            priority="high"
        )
        
        self._store_inbox_item(item)
        return item

    def evening_nudge(self) -> list[InboxItem]:
        """
        Generates evening nudges for missed habits and lack of study activity.
        """
        logger.info("Generating evening nudges")
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        
        items = []
        
        try:
            with self._db._connect() as conn:
                # 1. Missed habits today
                all_habits = conn.execute("SELECT id, name FROM habits WHERE active = 1").fetchall()
                logged = conn.execute(
                    "SELECT habit_id FROM habit_logs WHERE date = ? AND completed = 1",
                    (today,)
                ).fetchall()
                logged_ids = {l["habit_id"] for l in logged}
                
                missed = [h for h in all_habits if h["id"] not in logged_ids]
                if missed:
                    names = [h["name"] for h in missed]
                    item = InboxItem(
                        id=str(uuid.uuid4()),
                        type="evening_nudge",
                        title="Evening Habit Reminder",
                        content=f"You haven't logged these habits today: {', '.join(names)}",
                        timestamp=now.isoformat(),
                        priority="normal"
                    )
                    self._store_inbox_item(item)
                    items.append(item)
                
                # 2. Exam study alerts
                # Find goals with 'exam' or 'test' within 7 days
                seven_days = (now + timedelta(days=self._settings.proactive.exam_alert_days)).strftime("%Y-%m-%d")
                exams = conn.execute(
                    "SELECT title, deadline FROM goals "
                    "WHERE status = 'active' AND deadline IS NOT NULL "
                    "AND deadline >= ? AND deadline <= ? "
                    "AND (title LIKE '%exam%' OR title LIKE '%test%' OR description LIKE '%exam%')",
                    (today, seven_days)
                ).fetchall()
                
                if exams:
                    # Check recent study activity (last 3 days)
                    three_days_ago = (now - timedelta(days=3)).isoformat()
                    recent_study = conn.execute(
                        "SELECT COUNT(*) as count FROM messages "
                        "WHERE role = 'user' AND timestamp >= ? "
                        "AND (content LIKE '%study%' OR content LIKE '%revision%' OR content LIKE '%read%')",
                        (three_days_ago,)
                    ).fetchone()
                    
                    if recent_study and recent_study["count"] == 0:
                        exam_names = [e["title"] for e in exams]
                        item = InboxItem(
                            id=str(uuid.uuid4()),
                            type="alert",
                            title="Study Alert",
                            content=f"You have upcoming exams ({', '.join(exam_names)}) but no study sessions recorded in the last 3 days.",
                            timestamp=now.isoformat(),
                            priority="high"
                        )
                        self._store_inbox_item(item)
                        items.append(item)
                        
        except Exception as e:
            logger.error("Failed to generate evening nudges: %s", e)
            
        return items

    def _try_llm_summary(self, data: str, context: str) -> str:
        """Use local LLM to format the proactive message into natural language."""
        if not data or data == "No pressing items for today.":
            return data
            
        prompt = f"""Format the following raw data into a friendly, concise {context} for the user.
Use bullet points. Keep the tone professional but supportive.

Raw Data:
{data}
"""
        try:
            original_model = self._llm._model
            self._llm._model = self._settings.local_models.fast
            resp = self._llm.chat([{"role": "user", "content": prompt}])
            self._llm._model = original_model
            return resp.content.strip()
        except Exception as e:
            logger.warning("LLM formatting failed, using raw data: %s", e)
            return f"Raw Data:\n{data}"

    # -------------------------------------------------------------------
    # Inbox Management
    # -------------------------------------------------------------------

    def _store_inbox_item(self, item: InboxItem) -> None:
        """Persist to database."""
        try:
            with self._db._connect() as conn:
                conn.execute(
                    "INSERT INTO inbox (id, type, title, content, timestamp, read_status, priority, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (item.id, item.type, item.title, item.content, item.timestamp,
                     1 if item.read_status else 0, item.priority, item.source)
                )
        except Exception as e:
            logger.error("Failed to store inbox item: %s", e)

    def get_unread_inbox(self) -> list[InboxItem]:
        """Fetch all unread messages."""
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM inbox WHERE read_status = 0 ORDER BY timestamp DESC"
                ).fetchall()
                return [InboxItem(**dict(r)) for r in rows]
        except Exception as e:
            logger.error("Failed to get unread inbox: %s", e)
            return []

    def get_all_inbox(self, limit: int = 50) -> list[InboxItem]:
        """Fetch message history."""
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM inbox ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [InboxItem(**dict(r)) for r in rows]
        except Exception as e:
            logger.error("Failed to get inbox: %s", e)
            return []

    def mark_read(self, item_id: str) -> bool:
        """Mark single message as read."""
        try:
            with self._db._connect() as conn:
                conn.execute("UPDATE inbox SET read_status = 1 WHERE id = ?", (item_id,))
            return True
        except Exception as e:
            logger.error("Failed to mark inbox read: %s", e)
            return False

    def mark_all_read(self) -> int:
        """Mark all messages as read."""
        try:
            with self._db._connect() as conn:
                cursor = conn.execute("UPDATE inbox SET read_status = 1 WHERE read_status = 0")
                return cursor.rowcount
        except Exception as e:
            logger.error("Failed to mark all inbox read: %s", e)
            return 0

    def clear_old_inbox(self, days: int = 30) -> int:
        """Delete old read messages."""
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with self._db._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM inbox WHERE read_status = 1 AND timestamp < ?",
                    (cutoff,)
                )
                return cursor.rowcount
        except Exception as e:
            logger.error("Failed to clear old inbox: %s", e)
            return 0

    def on_startup(self) -> list[InboxItem]:
        """
        Called when JARVIS terminal starts.
        Returns unread items for immediate display.
        """
        if self._settings.proactive.show_inbox_on_startup:
            return self.get_unread_inbox()
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
    
    layer = ProactiveLayer(db=db, analytics_manager=am, settings=settings)
    
    print("=== Testing Proactive Layer ===")
    
    # Generate briefing
    briefing = layer.morning_briefing()
    print(f"\n{briefing.title}")
    print("-" * 40)
    print(briefing.content)
    
    # Generate nudges
    nudges = layer.evening_nudge()
    for n in nudges:
        print(f"\n[{n.priority.upper()}] {n.title}")
        print("-" * 40)
        print(n.content)
        
    # Check inbox
    print("\nUnread Inbox Items:")
    unread = layer.get_unread_inbox()
    for item in unread:
        print(f"  - {item.title}")
        
    # Mark read
    count = layer.mark_all_read()
    print(f"\nMarked {count} items as read.")
