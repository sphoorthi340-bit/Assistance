"""
Jarvis V2 — Goal Manager
============================
CRUD and lifecycle management for personal goals.

Architecture decisions:
    - Goals support four target types: streak, count, completion, progress.
      This covers everything from "meditate daily for 30 days" (streak) to
      "read 12 books this year" (count) to "finish thesis" (completion).
    - Priority 1-5 (1=highest) for ordering; defaults to 3 (medium).
    - ID prefix matching allows short-hand references in conversation
      (e.g., "update goal a3f" instead of the full UUID).
    - All mutations are logged at INFO level for accountability auditing.
    - Status lifecycle: active → paused → active (resume) or → completed/abandoned.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger(__name__)


class GoalManager:
    """
    Manages the full lifecycle of personal goals.

    Provides CRUD operations, status transitions, and ID prefix resolution
    for convenient conversational references.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db
        logger.info("GoalManager initialized")

    # -------------------------------------------------------------------
    # ID resolution
    # -------------------------------------------------------------------

    def _resolve_goal_id(self, goal_id_or_prefix: str) -> Optional[str]:
        """
        Resolve a full goal ID or an unambiguous prefix to the full ID.

        Tries exact match first, then prefix match. Returns None if no match
        or if the prefix is ambiguous (matches multiple goals).

        Args:
            goal_id_or_prefix: Full UUID or unique prefix.

        Returns:
            The full goal ID, or None if unresolvable.
        """
        with self._db._connect() as conn:
            # Exact match
            row = conn.execute(
                "SELECT id FROM goals WHERE id = ?",
                (goal_id_or_prefix,),
            ).fetchone()
            if row:
                return row["id"]

            # Prefix match
            rows = conn.execute(
                "SELECT id FROM goals WHERE id LIKE ?",
                (f"{goal_id_or_prefix}%",),
            ).fetchall()

        if len(rows) == 1:
            return rows[0]["id"]

        if len(rows) > 1:
            logger.warning(
                "Ambiguous goal prefix '%s' matched %d goals",
                goal_id_or_prefix, len(rows),
            )
        return None

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_goal(
        self,
        title: str,
        description: Optional[str] = None,
        category: str = "personal",
        target_type: str = "completion",
        target_value: Optional[float] = None,
        priority: int = 3,
        start_date: Optional[str] = None,
        deadline: Optional[str] = None,
    ) -> dict:
        """
        Create a new goal.

        Args:
            title: Short name for the goal.
            description: Optional detailed description.
            category: Grouping label (e.g., 'personal', 'health', 'career').
            target_type: One of 'streak', 'count', 'completion', 'progress'.
            target_value: Numeric target (e.g., 30 for a 30-day streak).
            priority: 1 (highest) to 5 (lowest), default 3.
            start_date: ISO date string for when the goal begins.
            deadline: ISO date string for the target completion date.

        Returns:
            The full goal dict as stored.
        """
        goal_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        goal = {
            "id": goal_id,
            "title": title,
            "description": description,
            "category": category,
            "target_type": target_type,
            "target_value": target_value,
            "current_value": 0,
            "status": "active",
            "priority": priority,
            "start_date": start_date,
            "deadline": deadline,
            "created_at": now,
            "updated_at": now,
        }

        with self._db._connect() as conn:
            conn.execute(
                "INSERT INTO goals "
                "(id, title, description, category, target_type, target_value, "
                "current_value, status, priority, start_date, deadline, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    goal_id, title, description, category, target_type,
                    target_value, 0, "active", priority, start_date,
                    deadline, now, now,
                ),
            )

        logger.info("Created goal '%s' [%s] (priority=%d)", title, goal_id[:8], priority)
        return goal

    def list_goals(
        self,
        status: str = "active",
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        List goals filtered by status and optional category.

        Args:
            status: Filter by goal status (e.g., 'active', 'completed').
            category: Optional category filter.

        Returns:
            List of goal dicts ordered by priority ASC, then created_at DESC.
        """
        query = "SELECT * FROM goals WHERE status = ?"
        params: list = [status]

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY priority ASC, created_at DESC"

        with self._db._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        logger.debug("Listed %d goals (status=%s, category=%s)", len(rows), status, category)
        return [dict(row) for row in rows]

    def get_goal(self, goal_id: str) -> Optional[dict]:
        """
        Get a single goal by ID or ID prefix.

        Args:
            goal_id: Full UUID or unique prefix.

        Returns:
            Goal dict, or None if not found / ambiguous prefix.
        """
        resolved = self._resolve_goal_id(goal_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?",
                (resolved,),
            ).fetchone()

        return dict(row) if row else None

    def update_goal(self, goal_id: str, **fields) -> Optional[dict]:
        """
        Update one or more fields on a goal.

        Allowed fields: title, description, category, target_type,
        target_value, current_value, status, priority, start_date, deadline.

        Args:
            goal_id: Full UUID or unique prefix.
            **fields: Key-value pairs of fields to update.

        Returns:
            The updated goal dict, or None if not found.
        """
        resolved = self._resolve_goal_id(goal_id)
        if not resolved:
            return None

        allowed = {
            "title", "description", "category", "target_type",
            "target_value", "current_value", "status", "priority",
            "start_date", "deadline",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_goal(resolved)

        now = datetime.now(timezone.utc).isoformat()
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [resolved]

        with self._db._connect() as conn:
            conn.execute(
                f"UPDATE goals SET {set_clause} WHERE id = ?",
                values,
            )

        logger.info("Updated goal %s: %s", resolved[:8], list(updates.keys()))
        return self.get_goal(resolved)

    def complete_goal(self, goal_id: str) -> Optional[dict]:
        """
        Mark a goal as completed.

        Sets status to 'completed' and current_value to target_value
        (if target_value is set).

        Args:
            goal_id: Full UUID or unique prefix.

        Returns:
            The updated goal dict, or None if not found.
        """
        resolved = self._resolve_goal_id(goal_id)
        if not resolved:
            return None

        # Fetch target_value to set current_value
        goal = self.get_goal(resolved)
        if not goal:
            return None

        update_fields = {"status": "completed"}
        if goal.get("target_value") is not None:
            update_fields["current_value"] = goal["target_value"]

        result = self.update_goal(resolved, **update_fields)
        logger.info("Completed goal '%s' [%s]", goal["title"], resolved[:8])
        return result

    def pause_goal(self, goal_id: str) -> Optional[dict]:
        """
        Pause an active goal.

        Args:
            goal_id: Full UUID or unique prefix.

        Returns:
            The updated goal dict, or None if not found.
        """
        result = self.update_goal(goal_id, status="paused")
        if result:
            logger.info("Paused goal '%s' [%s]", result["title"], result["id"][:8])
        return result

    def resume_goal(self, goal_id: str) -> Optional[dict]:
        """
        Resume a paused goal.

        Args:
            goal_id: Full UUID or unique prefix.

        Returns:
            The updated goal dict, or None if not found.
        """
        result = self.update_goal(goal_id, status="active")
        if result:
            logger.info("Resumed goal '%s' [%s]", result["title"], result["id"][:8])
        return result

    def delete_goal(self, goal_id: str) -> bool:
        """
        Permanently delete a goal.

        Args:
            goal_id: Full UUID or unique prefix.

        Returns:
            True if a goal was deleted, False otherwise.
        """
        resolved = self._resolve_goal_id(goal_id)
        if not resolved:
            return False

        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM goals WHERE id = ?",
                (resolved,),
            )

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted goal: %s", resolved[:8])
        return deleted
