"""
Jarvis V2 — Analytics Manager
=================================
Deterministic, SQL-driven analytics across goals, habits, and projects.

Architecture decisions:
    - ALL analytics are computed from deterministic SQL queries and Python
      aggregation — no LLM reasoning, no probabilistic inference.
    - The accountability report generates supportive, analytical observations
      rather than guilt-driven messages. Tone: reflective, observant.
    - Consistency score is the average completion rate across all active
      habits, providing a single number for longitudinal tracking.
    - Dashboard stats are designed for a single-call overview, aggregating
      counts across all three domains (goals, habits, projects).
    - Methods are intentionally stateless — each call queries fresh data.
"""

from datetime import datetime, timedelta
from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger(__name__)


class AnalyticsManager:
    """
    Provides deterministic analytics across goals, habits, and projects.

    All computations are SQL-based with Python aggregation. No LLM calls.
    Designed for dashboard views, accountability reports, and trend analysis.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db
        logger.info("AnalyticsManager initialized")

    # -------------------------------------------------------------------
    # Habit analytics
    # -------------------------------------------------------------------

    def get_habit_completion_rate(self, habit_id: str, days: int = 30) -> float:
        """
        Calculate the completion rate for a habit over the last N days.

        Args:
            habit_id: The habit UUID.
            days: Number of days to look back.

        Returns:
            Completion rate as a percentage (0.0 to 100.0).
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._db._connect() as conn:
            completed_days = conn.execute(
                "SELECT COUNT(DISTINCT date) FROM habit_logs "
                "WHERE habit_id = ? AND completed = 1 AND date >= ?",
                (habit_id, start_date),
            ).fetchone()[0]

        rate = round((completed_days / days) * 100, 1) if days > 0 else 0.0
        logger.debug("Habit %s completion rate: %.1f%% over %d days", habit_id[:8], rate, days)
        return rate

    def get_all_habit_stats(self) -> list[dict]:
        """
        Get comprehensive stats for all active habits.

        Returns:
            List of dicts, each containing: name, current_streak,
            longest_streak, completion_rate_30d, last_logged, total_logs.
        """
        with self._db._connect() as conn:
            habits = conn.execute(
                "SELECT * FROM habits WHERE active = 1"
            ).fetchall()

        stats = []
        for habit in habits:
            habit_id = habit["id"]

            # Import locally to avoid circular dependency
            from state.habit_manager import HabitManager
            # Build a temporary manager for streak computation
            hm = HabitManager(self._db)
            streak_info = hm.get_streak(habit_id)

            completion_rate = self.get_habit_completion_rate(habit_id, days=30)

            with self._db._connect() as conn:
                total_logs = conn.execute(
                    "SELECT COUNT(*) FROM habit_logs WHERE habit_id = ?",
                    (habit_id,),
                ).fetchone()[0]

            stats.append({
                "name": habit["name"],
                "habit_id": habit_id,
                "current_streak": streak_info["current_streak"],
                "longest_streak": streak_info["longest_streak"],
                "completion_rate_30d": completion_rate,
                "last_logged": streak_info["last_logged"],
                "total_logs": total_logs,
            })

        logger.debug("Computed stats for %d active habits", len(stats))
        return stats

    # -------------------------------------------------------------------
    # Goal analytics
    # -------------------------------------------------------------------

    def get_goal_progress_summary(self) -> list[dict]:
        """
        Get progress info for all active goals.

        Returns:
            List of active goal dicts with their progress fields.
        """
        with self._db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE status = 'active' "
                "ORDER BY priority ASC, created_at DESC"
            ).fetchall()

        goals = []
        for row in rows:
            goal = dict(row)
            # Calculate progress percentage for numeric targets
            if goal.get("target_value") and goal["target_value"] > 0:
                current = goal.get("current_value") or 0
                goal["progress_percentage"] = round(
                    (current / goal["target_value"]) * 100, 1
                )
            else:
                goal["progress_percentage"] = 0.0
            goals.append(goal)

        logger.debug("Retrieved progress summary for %d active goals", len(goals))
        return goals

    def get_overdue_goals(self) -> list[dict]:
        """
        Get active goals that are past their deadline.

        Returns:
            List of overdue goal dicts.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        with self._db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals "
                "WHERE status = 'active' AND deadline IS NOT NULL "
                "AND deadline < ? "
                "ORDER BY deadline ASC",
                (today,),
            ).fetchall()

        logger.debug("Found %d overdue goals", len(rows))
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------
    # Project analytics
    # -------------------------------------------------------------------

    def get_project_health(self) -> list[dict]:
        """
        Assess health of all active projects.

        Returns:
            List of dicts with project info, days_since_worked,
            overdue_task_count, and a status assessment string.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()

        with self._db._connect() as conn:
            projects = conn.execute(
                "SELECT * FROM projects WHERE status = 'active' "
                "ORDER BY priority ASC"
            ).fetchall()

        results = []
        for proj in projects:
            project = dict(proj)
            project_id = project["id"]

            # Days since last worked
            last_worked = project.get("last_worked_at")
            if last_worked:
                try:
                    last_dt = datetime.fromisoformat(last_worked)
                    # Handle timezone-aware vs naive comparison
                    if last_dt.tzinfo is not None:
                        last_dt = last_dt.replace(tzinfo=None)
                    days_since = (now - last_dt).days
                except (ValueError, TypeError):
                    days_since = None
            else:
                days_since = None

            # Overdue task count
            with self._db._connect() as conn:
                overdue_count = conn.execute(
                    "SELECT COUNT(*) FROM project_tasks "
                    "WHERE project_id = ? AND status != 'completed' "
                    "AND due_date IS NOT NULL AND due_date < ?",
                    (project_id, today),
                ).fetchone()[0]

            # Status assessment
            if days_since is not None and days_since > 14:
                assessment = "stale"
            elif days_since is not None and days_since > 7:
                assessment = "inactive"
            elif overdue_count > 0:
                assessment = "at_risk"
            else:
                assessment = "on_track"

            project["days_since_worked"] = days_since
            project["overdue_task_count"] = overdue_count
            project["health_status"] = assessment
            results.append(project)

        logger.debug("Assessed health for %d active projects", len(results))
        return results

    def get_overdue_tasks(self) -> list[dict]:
        """
        Get all tasks past their due date that aren't completed.

        Returns:
            List of overdue task dicts with project_name included.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        with self._db._connect() as conn:
            rows = conn.execute(
                "SELECT t.*, p.name as project_name "
                "FROM project_tasks t "
                "JOIN projects p ON t.project_id = p.id "
                "WHERE t.status != 'completed' "
                "AND t.due_date IS NOT NULL AND t.due_date < ? "
                "ORDER BY t.due_date ASC",
                (today,),
            ).fetchall()

        logger.debug("Found %d overdue tasks", len(rows))
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------
    # Cross-cutting analytics
    # -------------------------------------------------------------------

    def get_dashboard_stats(self) -> dict:
        """
        Get aggregate counts across all domains for a dashboard view.

        Returns:
            Dict with counts: active_goals, completed_goals, active_habits,
            active_projects, total_tasks, completed_tasks, overdue_tasks,
            overdue_goals, overall_consistency_score.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        with self._db._connect() as conn:
            active_goals = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'"
            ).fetchone()[0]

            completed_goals = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'completed'"
            ).fetchone()[0]

            active_habits = conn.execute(
                "SELECT COUNT(*) FROM habits WHERE active = 1"
            ).fetchone()[0]

            active_projects = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE status = 'active'"
            ).fetchone()[0]

            total_tasks = conn.execute(
                "SELECT COUNT(*) FROM project_tasks"
            ).fetchone()[0]

            completed_tasks = conn.execute(
                "SELECT COUNT(*) FROM project_tasks WHERE status = 'completed'"
            ).fetchone()[0]

            overdue_tasks = conn.execute(
                "SELECT COUNT(*) FROM project_tasks "
                "WHERE status != 'completed' "
                "AND due_date IS NOT NULL AND due_date < ?",
                (today,),
            ).fetchone()[0]

            overdue_goals = conn.execute(
                "SELECT COUNT(*) FROM goals "
                "WHERE status = 'active' AND deadline IS NOT NULL "
                "AND deadline < ?",
                (today,),
            ).fetchone()[0]

        consistency = self.get_consistency_score(days=30)

        stats = {
            "active_goals": active_goals,
            "completed_goals": completed_goals,
            "active_habits": active_habits,
            "active_projects": active_projects,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "overdue_tasks": overdue_tasks,
            "overdue_goals": overdue_goals,
            "overall_consistency_score": consistency,
        }

        logger.debug("Dashboard stats: %s", stats)
        return stats

    def get_consistency_score(self, days: int = 30) -> float:
        """
        Calculate the average completion rate across all active habits.

        This single metric represents overall habit consistency over N days.

        Args:
            days: Number of days to look back.

        Returns:
            Average completion rate as a percentage (0.0 to 100.0).
        """
        with self._db._connect() as conn:
            habits = conn.execute(
                "SELECT id FROM habits WHERE active = 1"
            ).fetchall()

        if not habits:
            return 0.0

        total_rate = 0.0
        for habit in habits:
            total_rate += self.get_habit_completion_rate(habit["id"], days=days)

        score = round(total_rate / len(habits), 1)
        logger.debug("Consistency score (over %d days): %.1f%%", days, score)
        return score

    # -------------------------------------------------------------------
    # Accountability report
    # -------------------------------------------------------------------

    def get_accountability_report(self) -> dict:
        """
        Generate the main accountability report.

        Compiles missed habits, broken streaks, declining habits, inactive
        projects, overdue goals/tasks, and human-readable observations.

        The tone is reflective, analytical, and supportive — never guilt-driven.

        Returns:
            Dict with missed_habits, broken_streaks, declining_habits,
            inactive_projects, overdue_goals, overdue_tasks, observations,
            and consistency_score.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        observations: list[str] = []

        # --- Missed habits (not logged today) ---
        with self._db._connect() as conn:
            all_active_habits = conn.execute(
                "SELECT * FROM habits WHERE active = 1"
            ).fetchall()

            logged_today_ids = {
                row["habit_id"] for row in conn.execute(
                    "SELECT DISTINCT habit_id FROM habit_logs WHERE date = ?",
                    (today,),
                ).fetchall()
            }

        missed_habits = []
        for habit in all_active_habits:
            if habit["id"] not in logged_today_ids:
                missed_habits.append(dict(habit))

        completed_today = len(all_active_habits) - len(missed_habits)
        total_habits = len(all_active_habits)

        if total_habits > 0:
            if completed_today == total_habits:
                observations.append(
                    f"All {total_habits} habits completed today. Excellent consistency."
                )
            elif completed_today > 0:
                observations.append(
                    f"{completed_today} out of {total_habits} habits were completed today. "
                    f"{'Strong' if completed_today >= total_habits * 0.6 else 'Moderate'} consistency."
                )

        # --- Broken streaks (had a streak but broke recently) ---
        from state.habit_manager import HabitManager
        hm = HabitManager(self._db)

        broken_streaks = []
        for habit in all_active_habits:
            streak_info = hm.get_streak(habit["id"])

            # A streak is "broken" if the longest streak > 2 but current streak is 0
            if streak_info["longest_streak"] > 2 and streak_info["current_streak"] == 0:
                last_logged = streak_info["last_logged"]
                if last_logged:
                    days_ago = (datetime.now().date() - datetime.strptime(
                        last_logged, "%Y-%m-%d"
                    ).date()).days
                    if days_ago <= 7:  # Only report recently broken streaks
                        broken_streaks.append({
                            "name": habit["name"],
                            "habit_id": habit["id"],
                            "longest_streak": streak_info["longest_streak"],
                            "days_since_last": days_ago,
                        })
                        observations.append(
                            f"Your {habit['name']} streak broke {days_ago} day(s) ago "
                            f"after reaching {streak_info['longest_streak']} days."
                        )

        # --- Declining habits (recent rate < overall rate) ---
        declining_habits = []
        for habit in all_active_habits:
            rate_7d = self.get_habit_completion_rate(habit["id"], days=7)
            rate_30d = self.get_habit_completion_rate(habit["id"], days=30)

            # Declining if 7-day rate is significantly lower than 30-day rate
            if rate_30d > 20 and rate_7d < rate_30d * 0.6:
                declining_habits.append({
                    "name": habit["name"],
                    "habit_id": habit["id"],
                    "rate_7d": rate_7d,
                    "rate_30d": rate_30d,
                })
                observations.append(
                    f"{habit['name']} completion has dropped from "
                    f"{rate_30d:.0f}% (30-day) to {rate_7d:.0f}% (7-day)."
                )

        # --- Habit completion observations ---
        for habit in all_active_habits:
            rate_30d = self.get_habit_completion_rate(habit["id"], days=30)
            if rate_30d > 0:
                completed_count = round(rate_30d * 30 / 100)
                observations.append(
                    f"You've completed {habit['name']} {completed_count} out of "
                    f"the last 30 days ({rate_30d:.0f}%)."
                )

        # --- Inactive projects (not worked on in 7+ days) ---
        project_health = self.get_project_health()
        inactive_projects = [
            p for p in project_health
            if p.get("days_since_worked") is not None and p["days_since_worked"] >= 7
        ]

        for proj in inactive_projects:
            observations.append(
                f"{proj['name']} has not been updated in {proj['days_since_worked']} days."
            )

        # --- Overdue goals ---
        overdue_goals = self.get_overdue_goals()

        # --- Overdue tasks ---
        overdue_tasks = self.get_overdue_tasks()

        # --- Deadline proximity observations ---
        with self._db._connect() as conn:
            week_from_now = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            approaching = conn.execute(
                "SELECT COUNT(*) FROM goals "
                "WHERE status = 'active' AND deadline IS NOT NULL "
                "AND deadline >= ? AND deadline <= ?",
                (today, week_from_now),
            ).fetchone()[0]

        if approaching > 0:
            observations.append(
                f"You have {approaching} goal{'s' if approaching > 1 else ''} "
                f"approaching {'their' if approaching > 1 else 'its'} "
                f"deadline this week."
            )

        if overdue_goals:
            observations.append(
                f"{len(overdue_goals)} goal{'s are' if len(overdue_goals) > 1 else ' is'} "
                f"past {'their' if len(overdue_goals) > 1 else 'its'} deadline."
            )

        if overdue_tasks:
            observations.append(
                f"{len(overdue_tasks)} task{'s are' if len(overdue_tasks) > 1 else ' is'} "
                f"overdue across your projects."
            )

        # --- Consistency score ---
        consistency = self.get_consistency_score(days=30)

        report = {
            "missed_habits": missed_habits,
            "broken_streaks": broken_streaks,
            "declining_habits": declining_habits,
            "inactive_projects": inactive_projects,
            "overdue_goals": overdue_goals,
            "overdue_tasks": overdue_tasks,
            "observations": observations,
            "consistency_score": consistency,
        }

        logger.info(
            "Generated accountability report — %d missed habits, %d broken streaks, "
            "%d declining, %d inactive projects, consistency=%.1f%%",
            len(missed_habits), len(broken_streaks), len(declining_habits),
            len(inactive_projects), consistency,
        )
        return report
