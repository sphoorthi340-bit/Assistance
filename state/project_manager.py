"""
Jarvis V2 — Project Manager
================================
Manages projects and their nested tasks with automatic progress tracking.

Architecture decisions:
    - Projects contain tasks (1:N via project_tasks table with CASCADE delete).
    - Progress percentage is auto-calculated from completed/total task ratio
      when a task is marked complete. Manual override is still possible via
      update_project().
    - last_worked_at is updated whenever a task is completed or the project
      is directly updated — this powers "inactive project" detection in
      analytics.
    - Tasks have their own status lifecycle: pending → in_progress → completed.
    - Project summaries aggregate task counts for quick dashboard views.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger(__name__)


class ProjectManager:
    """
    Manages projects and their associated tasks.

    Provides CRUD for projects and tasks, auto-progress tracking,
    and summary aggregation for dashboard views.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db
        logger.info("ProjectManager initialized")

    # -------------------------------------------------------------------
    # ID resolution
    # -------------------------------------------------------------------

    def _resolve_project_id(self, project_id_or_prefix: str) -> Optional[str]:
        """
        Resolve a full project ID or unambiguous prefix to the full ID.

        Args:
            project_id_or_prefix: Full UUID or unique prefix.

        Returns:
            The full project ID, or None if unresolvable.
        """
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT id FROM projects WHERE id = ?",
                (project_id_or_prefix,),
            ).fetchone()
            if row:
                return row["id"]

            rows = conn.execute(
                "SELECT id FROM projects WHERE id LIKE ?",
                (f"{project_id_or_prefix}%",),
            ).fetchall()

        if len(rows) == 1:
            return rows[0]["id"]

        if len(rows) > 1:
            logger.warning(
                "Ambiguous project prefix '%s' matched %d projects",
                project_id_or_prefix, len(rows),
            )
        return None

    def _resolve_task_id(self, task_id_or_prefix: str) -> Optional[str]:
        """
        Resolve a full task ID or unambiguous prefix to the full ID.

        Args:
            task_id_or_prefix: Full UUID or unique prefix.

        Returns:
            The full task ID, or None if unresolvable.
        """
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT id FROM project_tasks WHERE id = ?",
                (task_id_or_prefix,),
            ).fetchone()
            if row:
                return row["id"]

            rows = conn.execute(
                "SELECT id FROM project_tasks WHERE id LIKE ?",
                (f"{task_id_or_prefix}%",),
            ).fetchall()

        if len(rows) == 1:
            return rows[0]["id"]

        if len(rows) > 1:
            logger.warning(
                "Ambiguous task prefix '%s' matched %d tasks",
                task_id_or_prefix, len(rows),
            )
        return None

    # -------------------------------------------------------------------
    # Project CRUD
    # -------------------------------------------------------------------

    def add_project(
        self,
        name: str,
        description: Optional[str] = None,
        priority: int = 3,
    ) -> dict:
        """
        Create a new project.

        Args:
            name: Project name.
            description: Optional detailed description.
            priority: 1 (highest) to 5 (lowest), default 3.

        Returns:
            The full project dict as stored.
        """
        project_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "status": "active",
            "progress_percentage": 0,
            "current_blocker": None,
            "next_step": None,
            "priority": priority,
            "created_at": now,
            "updated_at": now,
            "last_worked_at": None,
        }

        with self._db._connect() as conn:
            conn.execute(
                "INSERT INTO projects "
                "(id, name, description, status, progress_percentage, "
                "priority, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, name, description, "active", 0, priority, now, now),
            )

        logger.info("Created project '%s' [%s]", name, project_id[:8])
        return project

    def list_projects(self, status: str = "active") -> list[dict]:
        """
        List projects filtered by status.

        Args:
            status: One of 'active', 'paused', 'completed', 'archived'.

        Returns:
            List of project dicts ordered by priority ASC, then updated_at DESC.
        """
        with self._db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects WHERE status = ? "
                "ORDER BY priority ASC, updated_at DESC",
                (status,),
            ).fetchall()

        logger.debug("Listed %d projects (status=%s)", len(rows), status)
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> Optional[dict]:
        """
        Get a single project by ID or ID prefix.

        Args:
            project_id: Full UUID or unique prefix.

        Returns:
            Project dict, or None if not found.
        """
        resolved = self._resolve_project_id(project_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (resolved,),
            ).fetchone()

        return dict(row) if row else None

    def update_project(self, project_id: str, **fields) -> Optional[dict]:
        """
        Update one or more fields on a project.

        Allowed fields: name, description, status, progress_percentage,
        current_blocker, next_step, priority.

        Args:
            project_id: Full UUID or unique prefix.
            **fields: Key-value pairs to update.

        Returns:
            The updated project dict, or None if not found.
        """
        resolved = self._resolve_project_id(project_id)
        if not resolved:
            return None

        allowed = {
            "name", "description", "status", "progress_percentage",
            "current_blocker", "next_step", "priority",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_project(resolved)

        now = datetime.now(timezone.utc).isoformat()
        updates["updated_at"] = now
        updates["last_worked_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [resolved]

        with self._db._connect() as conn:
            conn.execute(
                f"UPDATE projects SET {set_clause} WHERE id = ?",
                values,
            )

        logger.info("Updated project %s: %s", resolved[:8], list(updates.keys()))
        return self.get_project(resolved)

    def complete_project(self, project_id: str) -> Optional[dict]:
        """
        Mark a project as completed with 100% progress.

        Args:
            project_id: Full UUID or unique prefix.

        Returns:
            The updated project dict, or None if not found.
        """
        result = self.update_project(
            project_id,
            status="completed",
            progress_percentage=100,
        )
        if result:
            logger.info("Completed project '%s' [%s]", result["name"], result["id"][:8])
        return result

    def delete_project(self, project_id: str) -> bool:
        """
        Permanently delete a project and all its tasks (CASCADE).

        Args:
            project_id: Full UUID or unique prefix.

        Returns:
            True if the project was deleted.
        """
        resolved = self._resolve_project_id(project_id)
        if not resolved:
            return False

        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM projects WHERE id = ?",
                (resolved,),
            )

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted project: %s", resolved[:8])
        return deleted

    # -------------------------------------------------------------------
    # Task CRUD
    # -------------------------------------------------------------------

    def add_task(
        self,
        project_id: str,
        title: str,
        description: Optional[str] = None,
        due_date: Optional[str] = None,
        priority: int = 3,
    ) -> dict:
        """
        Add a task to a project.

        Args:
            project_id: Parent project ID or prefix.
            title: Task title.
            description: Optional task description.
            due_date: Optional due date (YYYY-MM-DD).
            priority: 1 (highest) to 5 (lowest).

        Returns:
            The full task dict as stored.
        """
        resolved_project = self._resolve_project_id(project_id)
        if not resolved_project:
            raise ValueError(f"Project not found: {project_id}")

        task_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "id": task_id,
            "project_id": resolved_project,
            "title": title,
            "description": description,
            "status": "pending",
            "due_date": due_date,
            "priority": priority,
            "created_at": now,
            "completed_at": None,
        }

        with self._db._connect() as conn:
            conn.execute(
                "INSERT INTO project_tasks "
                "(id, project_id, title, description, status, due_date, "
                "priority, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id, resolved_project, title, description,
                    "pending", due_date, priority, now,
                ),
            )

        logger.info(
            "Added task '%s' to project %s [%s]",
            title, resolved_project[:8], task_id[:8],
        )
        return task

    def list_tasks(
        self,
        project_id: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        """
        List tasks for a project, optionally filtered by status.

        Args:
            project_id: Parent project ID or prefix.
            status: Optional status filter ('pending', 'in_progress', 'completed').

        Returns:
            List of task dicts ordered by priority ASC, then created_at ASC.
        """
        resolved = self._resolve_project_id(project_id)
        if not resolved:
            return []

        query = "SELECT * FROM project_tasks WHERE project_id = ?"
        params: list = [resolved]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY priority ASC, created_at ASC"

        with self._db._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        logger.debug("Listed %d tasks for project %s", len(rows), resolved[:8])
        return [dict(row) for row in rows]

    def update_task(self, task_id: str, **fields) -> Optional[dict]:
        """
        Update one or more fields on a task.

        Allowed fields: title, description, status, due_date, priority.

        Args:
            task_id: Full UUID or unique prefix.
            **fields: Key-value pairs to update.

        Returns:
            The updated task dict, or None if not found.
        """
        resolved = self._resolve_task_id(task_id)
        if not resolved:
            return None

        allowed = {"title", "description", "status", "due_date", "priority"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            with self._db._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM project_tasks WHERE id = ?",
                    (resolved,),
                ).fetchone()
            return dict(row) if row else None

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [resolved]

        with self._db._connect() as conn:
            conn.execute(
                f"UPDATE project_tasks SET {set_clause} WHERE id = ?",
                values,
            )

        logger.info("Updated task %s: %s", resolved[:8], list(updates.keys()))

        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_tasks WHERE id = ?",
                (resolved,),
            ).fetchone()
        return dict(row) if row else None

    def complete_task(self, task_id: str) -> Optional[dict]:
        """
        Mark a task as completed and recalculate parent project progress.

        Sets task status to 'completed' and completed_at to now, then
        updates the parent project's progress_percentage based on the
        ratio of completed to total tasks.

        Args:
            task_id: Full UUID or unique prefix.

        Returns:
            The updated task dict, or None if not found.
        """
        resolved = self._resolve_task_id(task_id)
        if not resolved:
            return None

        now = datetime.now(timezone.utc).isoformat()

        with self._db._connect() as conn:
            # Mark task completed
            conn.execute(
                "UPDATE project_tasks SET status = 'completed', "
                "completed_at = ? WHERE id = ?",
                (now, resolved),
            )

            # Get parent project ID
            task_row = conn.execute(
                "SELECT * FROM project_tasks WHERE id = ?",
                (resolved,),
            ).fetchone()

            if task_row:
                project_id = task_row["project_id"]

                # Recalculate project progress
                total = conn.execute(
                    "SELECT COUNT(*) FROM project_tasks WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0]

                completed = conn.execute(
                    "SELECT COUNT(*) FROM project_tasks "
                    "WHERE project_id = ? AND status = 'completed'",
                    (project_id,),
                ).fetchone()[0]

                progress = round((completed / total) * 100, 1) if total > 0 else 0

                conn.execute(
                    "UPDATE projects SET progress_percentage = ?, "
                    "updated_at = ?, last_worked_at = ? WHERE id = ?",
                    (progress, now, now, project_id),
                )

                logger.info(
                    "Completed task %s — project %s progress: %.1f%%",
                    resolved[:8], project_id[:8], progress,
                )

        return dict(task_row) if task_row else None

    def delete_task(self, task_id: str) -> bool:
        """
        Permanently delete a task.

        Args:
            task_id: Full UUID or unique prefix.

        Returns:
            True if the task was deleted.
        """
        resolved = self._resolve_task_id(task_id)
        if not resolved:
            return False

        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM project_tasks WHERE id = ?",
                (resolved,),
            )

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted task: %s", resolved[:8])
        return deleted

    # -------------------------------------------------------------------
    # Project summary
    # -------------------------------------------------------------------

    def get_project_summary(self, project_id: str) -> Optional[dict]:
        """
        Get a comprehensive summary of a project and its tasks.

        Args:
            project_id: Full UUID or unique prefix.

        Returns:
            Dict with project info plus task count breakdowns, or None.
        """
        resolved = self._resolve_project_id(project_id)
        if not resolved:
            return None

        with self._db._connect() as conn:
            project_row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (resolved,),
            ).fetchone()

            if not project_row:
                return None

            project = dict(project_row)

            today = datetime.now().strftime("%Y-%m-%d")

            total = conn.execute(
                "SELECT COUNT(*) FROM project_tasks WHERE project_id = ?",
                (resolved,),
            ).fetchone()[0]

            completed = conn.execute(
                "SELECT COUNT(*) FROM project_tasks "
                "WHERE project_id = ? AND status = 'completed'",
                (resolved,),
            ).fetchone()[0]

            pending = conn.execute(
                "SELECT COUNT(*) FROM project_tasks "
                "WHERE project_id = ? AND status = 'pending'",
                (resolved,),
            ).fetchone()[0]

            in_progress = conn.execute(
                "SELECT COUNT(*) FROM project_tasks "
                "WHERE project_id = ? AND status = 'in_progress'",
                (resolved,),
            ).fetchone()[0]

            overdue = conn.execute(
                "SELECT COUNT(*) FROM project_tasks "
                "WHERE project_id = ? AND status != 'completed' "
                "AND due_date IS NOT NULL AND due_date < ?",
                (resolved, today),
            ).fetchone()[0]

        progress = round((completed / total) * 100, 1) if total > 0 else 0

        project.update({
            "total_tasks": total,
            "completed_tasks": completed,
            "pending_tasks": pending,
            "in_progress_tasks": in_progress,
            "overdue_tasks": overdue,
            "progress_percentage": progress,
        })

        logger.debug(
            "Project summary for %s: %d/%d tasks completed",
            resolved[:8], completed, total,
        )
        return project
