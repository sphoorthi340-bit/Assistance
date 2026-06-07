"""
JARVIS System 4 — Structured Memory Manager
=============================================
Manages the three-tier S4 memory system:

  HOT   (in-session dict)           — current task stack, today's context
  WARM  (daily JSON + weekly JSON)  — daily log, weekly state
  COLD  (long-term profile JSONs)   — user profile, academic map, skill graph, MS roadmap

This layer sits ABOVE the existing ChromaDB/SQLite memory system.
It tracks structured, schema-defined state that the vector store cannot represent.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------

@dataclass
class DistractionEvent:
    time: str           # HH:MM
    type: str           # youtube | reddit | phone | side_project | other
    duration_min: int = 0


@dataclass
class DailyLog:
    date: str
    session_start: Optional[str] = None
    session_end: Optional[str] = None
    tasks_planned: list = field(default_factory=list)
    tasks_completed: list = field(default_factory=list)
    tasks_deferred: list = field(default_factory=list)
    study_hours_by_subject: dict = field(default_factory=dict)
    pomodoros_completed: int = 0
    pomodoros_broken: int = 0
    total_focused_minutes: int = 0
    distraction_incidents: list = field(default_factory=list)
    papers_read: list = field(default_factory=list)
    coding_minutes: int = 0
    jarvis_commits: int = 0
    wins: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    morning_brief_generated: bool = False
    evening_wrap_generated: bool = False
    model_interactions: dict = field(default_factory=lambda: {
        "chief": 0, "analyst": 0, "engineer": 0, "mentor": 0, "rapid": 0
    })

    @property
    def total_study_minutes(self) -> int:
        return sum(self.study_hours_by_subject.values())

    @property
    def total_study_hours(self) -> float:
        return self.total_study_minutes / 60.0


@dataclass
class WeeklyState:
    week_number: int
    year: int
    start_date: str
    end_date: str
    goals: list = field(default_factory=list)
    goal_progress: dict = field(default_factory=dict)
    study_hours_total: float = 0.0
    study_target_hours: float = 42.0
    tasks_planned: int = 0
    tasks_completed: int = 0
    distraction_incidents: int = 0
    papers_read: int = 0
    coding_hours: float = 0.0
    pomodoros_completed: int = 0
    subjects_covered: list = field(default_factory=list)
    at_risk: list = field(default_factory=list)
    analyst_report: Optional[str] = None
    next_week_goals: list = field(default_factory=list)
    review_completed: bool = False
    review_date: Optional[str] = None


# ---------------------------------------------------------------------------
# S4 Memory Manager
# ---------------------------------------------------------------------------

class S4MemoryManager:
    """
    Manages all structured S4 memory (hot, warm, cold tiers).

    Directory layout (relative to project root):
      data/s4_daily/         — one JSON per day
      data/s4_weekly/        — current_week.json + archive/
      data/s4_profile/       — user_profile.json, academic_map.json,
                                skill_graph.json, ms_roadmap.json
    """

    def __init__(self, base_dir: str = None):
        """
        Args:
            base_dir: Project root directory. Defaults to two levels above this file.
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._base = base_dir
        self._daily_dir = os.path.join(base_dir, "data", "s4_daily")
        self._weekly_dir = os.path.join(base_dir, "data", "s4_weekly")
        self._archive_dir = os.path.join(base_dir, "data", "s4_weekly", "archive")
        self._profile_dir = os.path.join(base_dir, "data", "s4_profile")
        self._papers_dir = os.path.join(base_dir, "knowledge", "papers")

        # Hot memory (session-scoped)
        self._hot: dict = {
            "current_task": None,
            "session_start": None,
            "distraction_flag": False,
            "distraction_count_today": 0,
        }

        self._ensure_dirs()
        logger.info("S4MemoryManager initialized at %s", base_dir)

    def _ensure_dirs(self):
        """Create directory structure if it doesn't exist."""
        for d in [self._daily_dir, self._weekly_dir, self._archive_dir,
                  self._profile_dir, self._papers_dir]:
            os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------
    # Hot Memory
    # -------------------------------------------------------------------

    def set_current_task(self, task: str):
        """Record the user's current active task."""
        self._hot["current_task"] = task
        logger.debug("Hot memory: current task = '%s'", task)

    def get_current_task(self) -> Optional[str]:
        return self._hot.get("current_task")

    def flag_distraction(self):
        """Record a distraction event in hot memory."""
        self._hot["distraction_flag"] = True
        self._hot["distraction_count_today"] = self._hot.get("distraction_count_today", 0) + 1

    def clear_distraction_flag(self):
        self._hot["distraction_flag"] = False

    def is_distracted(self) -> bool:
        return self._hot.get("distraction_flag", False)

    def get_distraction_count(self) -> int:
        return self._hot.get("distraction_count_today", 0)

    # -------------------------------------------------------------------
    # Daily Log (Warm Memory)
    # -------------------------------------------------------------------

    def _daily_path(self, date_str: str = None) -> str:
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self._daily_dir, f"{date_str}.json")

    def get_daily_log(self, date_str: str = None) -> DailyLog:
        """Load today's (or specified date's) daily log, creating if absent."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        path = self._daily_path(date_str)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DailyLog(**{k: v for k, v in data.items()
                                   if k in DailyLog.__dataclass_fields__})
            except Exception as e:
                logger.warning("Failed to load daily log %s: %s", path, e)
        return DailyLog(date=date_str)

    def save_daily_log(self, log: DailyLog):
        """Persist a DailyLog to disk."""
        path = self._daily_path(log.date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(log), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save daily log: %s", e)

    def log_session_start(self):
        """Record session start time in today's log."""
        log = self.get_daily_log()
        if not log.session_start:
            log.session_start = datetime.now().strftime("%H:%M:%S")
        self._hot["session_start"] = log.session_start
        self.save_daily_log(log)

    def log_session_end(self):
        """Record session end time in today's log."""
        log = self.get_daily_log()
        log.session_end = datetime.now().strftime("%H:%M:%S")
        self.save_daily_log(log)

    def record_task_complete(self, task: str):
        """Mark a task as completed in today's log."""
        log = self.get_daily_log()
        if task not in log.tasks_completed:
            log.tasks_completed.append(task)
        self.save_daily_log(log)
        logger.debug("Task completed: %s", task)

    def record_task_planned(self, task: str):
        """Add a planned task to today's log."""
        log = self.get_daily_log()
        if task not in log.tasks_planned:
            log.tasks_planned.append(task)
        self.save_daily_log(log)

    def record_study(self, subject: str, duration_minutes: int):
        """Log study time for a subject."""
        log = self.get_daily_log()
        current = log.study_hours_by_subject.get(subject, 0)
        log.study_hours_by_subject[subject] = current + duration_minutes
        if subject not in log.subjects_covered if hasattr(log, "subjects_covered") else True:
            pass  # tracked at weekly level
        self.save_daily_log(log)

    def record_pomodoro(self, completed: bool = True):
        """Record a Pomodoro session."""
        log = self.get_daily_log()
        if completed:
            log.pomodoros_completed += 1
            log.total_focused_minutes += 25
        else:
            log.pomodoros_broken += 1
        self.save_daily_log(log)

    def record_distraction_event(self, distraction_type: str, duration_min: int = 0):
        """Log a distraction event."""
        log = self.get_daily_log()
        event = {
            "time": datetime.now().strftime("%H:%M"),
            "type": distraction_type,
            "duration_min": duration_min,
        }
        log.distraction_incidents.append(event)
        self.save_daily_log(log)
        self.flag_distraction()

    def record_win(self, win: str):
        """Log a daily win."""
        log = self.get_daily_log()
        log.wins.append(win)
        self.save_daily_log(log)

    def record_blocker(self, blocker: str):
        """Log a blocker."""
        log = self.get_daily_log()
        log.blockers.append(blocker)
        self.save_daily_log(log)

    def record_role_interaction(self, role_name: str):
        """Increment interaction count for a role."""
        log = self.get_daily_log()
        log.model_interactions[role_name] = log.model_interactions.get(role_name, 0) + 1
        self.save_daily_log(log)

    def record_paper_read(self, title: str):
        """Log a research paper as read today."""
        log = self.get_daily_log()
        if title not in log.papers_read:
            log.papers_read.append(title)
        self.save_daily_log(log)

    def get_last_n_days(self, n: int = 7) -> list[DailyLog]:
        """Load the last N days of logs for weekly review."""
        logs = []
        for i in range(n):
            date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            log = self.get_daily_log(date_str)
            logs.append(log)
        return logs

    # -------------------------------------------------------------------
    # Weekly State (Warm Memory)
    # -------------------------------------------------------------------

    def _weekly_path(self) -> str:
        return os.path.join(self._weekly_dir, "current_week.json")

    def get_weekly_state(self) -> WeeklyState:
        """Load current weekly state."""
        path = self._weekly_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return WeeklyState(**{k: v for k, v in data.items()
                                      if k in WeeklyState.__dataclass_fields__})
            except Exception as e:
                logger.warning("Failed to load weekly state: %s", e)

        # Create new weekly state
        now = datetime.now()
        iso_cal = now.isocalendar()
        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)
        return WeeklyState(
            week_number=iso_cal[1],
            year=iso_cal[0],
            start_date=week_start.strftime("%Y-%m-%d"),
            end_date=week_end.strftime("%Y-%m-%d"),
        )

    def save_weekly_state(self, state: WeeklyState):
        """Persist current weekly state."""
        try:
            with open(self._weekly_path(), "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save weekly state: %s", e)

    def aggregate_week_from_daily(self) -> WeeklyState:
        """Recompute weekly aggregates from daily logs."""
        state = self.get_weekly_state()
        daily_logs = self.get_last_n_days(7)

        state.study_hours_total = sum(
            log.total_study_hours for log in daily_logs
        )
        state.tasks_planned = sum(len(log.tasks_planned) for log in daily_logs)
        state.tasks_completed = sum(len(log.tasks_completed) for log in daily_logs)
        state.distraction_incidents = sum(
            len(log.distraction_incidents) for log in daily_logs
        )
        state.papers_read = sum(len(log.papers_read) for log in daily_logs)
        state.coding_hours = sum(log.coding_minutes for log in daily_logs) / 60.0
        state.pomodoros_completed = sum(log.pomodoros_completed for log in daily_logs)

        covered = set()
        for log in daily_logs:
            covered.update(log.study_hours_by_subject.keys())
        state.subjects_covered = list(covered)

        self.save_weekly_state(state)
        return state

    def archive_week(self) -> str:
        """Archive current week and reset. Returns archive filename."""
        state = self.aggregate_week_from_daily()
        state.review_completed = True
        state.review_date = datetime.now().strftime("%Y-%m-%d")

        archive_name = f"week_{state.year}_W{state.week_number:02d}.json"
        archive_path = os.path.join(self._archive_dir, archive_name)
        try:
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2, ensure_ascii=False)
            logger.info("Archived week to %s", archive_path)
        except Exception as e:
            logger.error("Failed to archive week: %s", e)

        # Reset current_week.json for next week
        now = datetime.now() + timedelta(days=1)
        iso_cal = now.isocalendar()
        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)
        new_state = WeeklyState(
            week_number=iso_cal[1],
            year=iso_cal[0],
            start_date=week_start.strftime("%Y-%m-%d"),
            end_date=week_end.strftime("%Y-%m-%d"),
            goals=state.next_week_goals,
        )
        self.save_weekly_state(new_state)
        return archive_name

    # -------------------------------------------------------------------
    # Cold Memory — Profile Files
    # -------------------------------------------------------------------

    def _load_profile(self, filename: str) -> dict:
        path = os.path.join(self._profile_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", filename, e)
        return {}

    def _save_profile(self, filename: str, data: dict):
        path = os.path.join(self._profile_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save profile %s: %s", filename, e)

    def get_user_profile(self) -> dict:
        return self._load_profile("user_profile.json")

    def update_user_profile(self, updates: dict):
        profile = self.get_user_profile()
        profile.update(updates)
        profile["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self._save_profile("user_profile.json", profile)

    def get_academic_map(self) -> dict:
        return self._load_profile("academic_map.json")

    def update_academic_map(self, updates: dict):
        data = self.get_academic_map()
        data.update(updates)
        self._save_profile("academic_map.json", data)

    def get_skill_graph(self) -> dict:
        return self._load_profile("skill_graph.json")

    def update_skill(self, skill_name: str, level: int, notes: str = ""):
        data = self.get_skill_graph()
        skills = data.get("skills", {})
        skills[skill_name] = {"level": level, "max": 5, "notes": notes}
        data["skills"] = skills
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self._save_profile("skill_graph.json", data)

    def get_ms_roadmap(self) -> dict:
        return self._load_profile("ms_roadmap.json")

    def update_ms_milestone(self, milestone_id: str, done: bool = True):
        data = self.get_ms_roadmap()
        for phase_data in data.get("phases", {}).values():
            for m in phase_data.get("milestones", []):
                if m.get("id") == milestone_id:
                    m["done"] = done
                    m["done_date"] = datetime.now().strftime("%Y-%m-%d") if done else None
        self._save_profile("ms_roadmap.json", data)
        logger.info("MS milestone %s marked done=%s", milestone_id, done)

    def add_target_university(self, name: str, details: dict = None):
        data = self.get_ms_roadmap()
        universities = data.get("universities", [])
        entry = {"name": name, "added": datetime.now().strftime("%Y-%m-%d")}
        if details:
            entry.update(details)
        universities.append(entry)
        data["universities"] = universities
        self._save_profile("ms_roadmap.json", data)

    def update_gre_score(self, score: int):
        data = self.get_ms_roadmap()
        data.setdefault("gre", {})["score"] = score
        data["gre"]["status"] = "taken"
        self._save_profile("ms_roadmap.json", data)

    def update_toefl_score(self, score: int):
        data = self.get_ms_roadmap()
        data.setdefault("toefl", {})["score"] = score
        data["toefl"]["status"] = "taken"
        self._save_profile("ms_roadmap.json", data)

    # -------------------------------------------------------------------
    # Quick Context Snapshot for System Prompts
    # -------------------------------------------------------------------

    def get_s4_context_snapshot(self) -> str:
        """
        Returns a compact text summary for injection into system prompts.
        Designed to be short (<200 tokens).
        """
        today = self.get_daily_log()
        weekly = self.get_weekly_state()
        profile = self.get_user_profile()

        lines = [
            "=== S4 CONTEXT SNAPSHOT ===",
            f"Date: {today.date}",
            f"Current Task: {self._hot.get('current_task', 'None')}",
            f"Study Today: {today.total_study_hours:.1f}h",
            f"Distractions Today: {len(today.distraction_incidents)}",
            f"Pomodoros Completed: {today.pomodoros_completed}",
            f"Tasks Done: {len(today.tasks_completed)}/{len(today.tasks_planned)}",
        ]

        if today.papers_read:
            lines.append(f"Papers Read Today: {len(today.papers_read)}")

        # Weekly summary
        lines.append(f"Week Study Total: {weekly.study_hours_total:.1f}h "
                     f"(target {weekly.study_target_hours:.0f}h)")

        # Current CGPA
        cgpa = profile.get("goals", {}).get("current_cgpa")
        target = profile.get("goals", {}).get("target_cgpa", 8.5)
        if cgpa:
            lines.append(f"CGPA: {cgpa} → target {target}")
        else:
            lines.append(f"CGPA target: {target}")

        return "\n".join(lines)
