"""
Jarvis V2 — Habit Manager
=============================
Manages recurring habits, daily logging, and streak computation.

Architecture decisions:
    - Habits are recurring behaviors (daily/weekly) with per-period targets.
    - Habit logs are keyed by (habit_id, date) — logging the same habit
      on the same date updates the existing entry (upsert semantics).
    - Dates use LOCAL time (not UTC) because habits are tied to the user's
      calendar day, not absolute time.
    - Streak calculation is pure Python over sorted log dates — simple,
      correct, and avoids complex SQL window functions on SQLite.
    - The active flag supports soft-deactivation without deleting history.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger(__name__)


class HabitManager:
    """
    Manages recurring habit definitions, daily logging, and streak tracking.

    Provides CRUD for habits, upsert-based log entries, and pure-Python
    streak computation over sorted date sequences.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db
        logger.info("HabitManager initialized")

    # -------------------------------------------------------------------
    # ID resolution
    # -------------------------------------------------------------------

    def _resolve_habit_id(self, habit_id_or_prefix: str) -> Optional[str]:
        """
        Resolve a full habit ID or an unambiguous prefix to the full ID.

        Args:
            habit_id_or_prefix: Full UUID or unique prefix.

        Returns:
            The full habit ID, or None if unresolvable.
        """
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT id FROM habits WHERE id = ?",
                (habit_id_or_prefix,),
            ).fetchone()
            if row:
                return row["id"]

            rows = conn.execute(
                "SELECT id FROM habits WHERE id LIKE ?",
                (f"{habit_id_or_prefix}%",),
            ).fetchall()

        if len(rows) == 1:
            return rows[0]["id"]

        if len(rows) > 1:
            logger.warning(
                "Ambiguous habit prefix '%s' matched %d habits",
                habit_id_or_prefix, len(rows),
            )
        return None

    # -------------------------------------------------------------------
    # Habit CRUD
    # -------------------------------------------------------------------

    def add_habit(
        self,
        name: str,
        description: Optional[str] = None,
        frequency: str = "daily",
        category: str = "personal",
        target_per_period: float = 1,
    ) -> dict:
        """
        Create a new habit to track.

        Args:
            name: Short name (e.g., "Morning run", "Read 30 min").
            description: Optional longer description.
            frequency: One of 'daily', 'weekly', 'custom'.
            category: Grouping label.
            target_per_period: How many times per period to target.

        Returns:
            The full habit dict as stored.
        """
        habit_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        habit = {
            "id": habit_id,
            "name": name,
            "description": description,
            "frequency": frequency,
            "category": category,
            "target_per_period": target_per_period,
            "active": 1,
            "created_at": now,
        }

        with self._db._connect() as conn:
            conn.execute(
                "INSERT INTO habits "
                "(id, name, description, frequency, category, "
                "target_per_period, active, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    habit_id, name, description, frequency, category,
                    target_per_period, 1, now,
                ),
            )

        logger.info("Created habit '%s' [%s] (frequency=%s)", name, habit_id[:8], frequency)
        return habit

    def list_habits(self, active_only: bool = True) -> list[dict]:
        """
        List all habits, optionally filtered to active ones.

        Args:
            active_only: If True, only return active habits.

        Returns:
            List of habit dicts.
        """
        if active_only:
            query = "SELECT * FROM habits WHERE active = 1 ORDER BY created_at DESC"
        else:
            query = "SELECT * FROM habits ORDER BY created_at DESC"

        with self._db._connect() as conn:
            rows = conn.execute(query).fetchall()

        logger.debug("Listed %d habits (active_only=%s)", len(rows), active_only)
        return [dict(row) for row in rows]

    def get_habit(self, habit_id: str) -> Optional[dict]:
        """
        Get a single habit by ID or ID prefix.

        Args:
            habit_id: Full UUID or unique prefix.

        Returns:
            Habit dict, or None if not found.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM habits WHERE id = ?",
                (resolved,),
            ).fetchone()

        return dict(row) if row else None

    def deactivate_habit(self, habit_id: str) -> Optional[dict]:
        """
        Soft-deactivate a habit (preserves history).

        Args:
            habit_id: Full UUID or unique prefix.

        Returns:
            The updated habit dict, or None if not found.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            conn.execute(
                "UPDATE habits SET active = 0 WHERE id = ?",
                (resolved,),
            )

        logger.info("Deactivated habit: %s", resolved[:8])
        return self.get_habit(resolved)

    def activate_habit(self, habit_id: str) -> Optional[dict]:
        """
        Re-activate a deactivated habit.

        Args:
            habit_id: Full UUID or unique prefix.

        Returns:
            The updated habit dict, or None if not found.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            conn.execute(
                "UPDATE habits SET active = 1 WHERE id = ?",
                (resolved,),
            )

        logger.info("Activated habit: %s", resolved[:8])
        return self.get_habit(resolved)

    def delete_habit(self, habit_id: str) -> bool:
        """
        Permanently delete a habit and all its logs (CASCADE).

        Args:
            habit_id: Full UUID or unique prefix.

        Returns:
            True if the habit was deleted.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return False

        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM habits WHERE id = ?",
                (resolved,),
            )

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted habit: %s", resolved[:8])
        return deleted

    # -------------------------------------------------------------------
    # Habit logging
    # -------------------------------------------------------------------

    def log_habit(
        self,
        habit_id: str,
        date: Optional[str] = None,
        value: Optional[float] = None,
        notes: Optional[str] = None,
        completed: bool = True,
    ) -> dict:
        """
        Log a habit completion for a given date.

        Uses upsert semantics: if a log already exists for this habit+date,
        it is updated instead of creating a duplicate.

        Args:
            habit_id: Full UUID or unique prefix.
            date: Date string (YYYY-MM-DD). Defaults to today (local time).
            value: Optional numeric value (e.g., minutes, reps).
            notes: Optional text notes.
            completed: Whether the habit was completed.

        Returns:
            The log entry dict.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            raise ValueError(f"Habit not found: {habit_id}")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        now = datetime.now(timezone.utc).isoformat()
        completed_int = 1 if completed else 0

        with self._db._connect() as conn:
            # Check for existing log on this date
            existing = conn.execute(
                "SELECT id FROM habit_logs WHERE habit_id = ? AND date = ?",
                (resolved, date),
            ).fetchone()

            if existing:
                # Update existing log
                log_id = existing["id"]
                conn.execute(
                    "UPDATE habit_logs SET completed = ?, value = ?, "
                    "notes = ? WHERE id = ?",
                    (completed_int, value, notes, log_id),
                )
                logger.info(
                    "Updated habit log for %s on %s (completed=%s)",
                    resolved[:8], date, completed,
                )
            else:
                # Create new log
                log_id = str(uuid4())
                conn.execute(
                    "INSERT INTO habit_logs "
                    "(id, habit_id, date, completed, value, notes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (log_id, resolved, date, completed_int, value, notes, now),
                )
                logger.info(
                    "Logged habit %s on %s (completed=%s)",
                    resolved[:8], date, completed,
                )

        return {
            "id": log_id,
            "habit_id": resolved,
            "date": date,
            "completed": completed_int,
            "value": value,
            "notes": notes,
            "created_at": now,
        }

    def get_logs(
        self,
        habit_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict]:
        """
        Get habit logs ordered by date descending.

        Args:
            habit_id: Full UUID or unique prefix.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).
            limit: Maximum number of logs to return.

        Returns:
            List of log dicts, newest first.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return []

        query = "SELECT * FROM habit_logs WHERE habit_id = ?"
        params: list = [resolved]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        with self._db._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        logger.debug("Retrieved %d logs for habit %s", len(rows), resolved[:8])
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------
    # Streak computation
    # -------------------------------------------------------------------

    def get_streak(self, habit_id: str) -> dict:
        """
        Compute current and longest streaks for a habit.

        Uses pure Python iteration over sorted log dates. A streak is a
        sequence of consecutive calendar days with completed=1 entries.

        Args:
            habit_id: Full UUID or unique prefix.

        Returns:
            Dict with current_streak, longest_streak, and last_logged date.
        """
        resolved = self._resolve_habit_id(habit_id)
        if not resolved:
            return {"current_streak": 0, "longest_streak": 0, "last_logged": None}

        with self._db._connect() as conn:
            rows = conn.execute(
                "SELECT date FROM habit_logs "
                "WHERE habit_id = ? AND completed = 1 "
                "ORDER BY date DESC",
                (resolved,),
            ).fetchall()

        if not rows:
            return {"current_streak": 0, "longest_streak": 0, "last_logged": None}

        # Parse dates and deduplicate (shouldn't have dupes, but safety)
        dates = sorted(
            {datetime.strptime(row["date"], "%Y-%m-%d").date() for row in rows},
            reverse=True,
        )

        last_logged = dates[0].isoformat()
        today = datetime.now().date()

        # --- Current streak (consecutive days backwards from today) ---
        current_streak = 0
        # The streak can start from today or yesterday
        expected = today
        if dates[0] < today:
            # Last log wasn't today — streak may still be alive if it was yesterday
            if dates[0] == today - timedelta(days=1):
                expected = today - timedelta(days=1)
            else:
                # Gap of 2+ days — no current streak
                current_streak = 0
                expected = None

        if expected is not None:
            for d in dates:
                if d == expected:
                    current_streak += 1
                    expected -= timedelta(days=1)
                elif d < expected:
                    break

        # --- Longest streak (scan all dates) ---
        dates_asc = sorted(dates)
        longest_streak = 1
        run = 1
        for i in range(1, len(dates_asc)):
            if dates_asc[i] - dates_asc[i - 1] == timedelta(days=1):
                run += 1
                longest_streak = max(longest_streak, run)
            else:
                run = 1

        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "last_logged": last_logged,
        }
