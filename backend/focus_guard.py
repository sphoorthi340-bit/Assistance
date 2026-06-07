"""
JARVIS System 4 — Focus Guard
================================
Pomodoro-based focus enforcement with distraction detection.

Features:
  - 25/5 min Pomodoro timer (configurable)
  - Distraction event logging
  - Lockdown mode after N distractions
  - Session stats (focus score, productivity rating)

Integrates with S4MemoryManager for persistence.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORK_MINUTES = 25
DEFAULT_BREAK_MINUTES = 5
DEFAULT_MAX_DISTRACTIONS = 2
DEFAULT_LOCKDOWN_MINUTES = 30

DISTRACTION_TYPES = {
    "youtube": "YouTube",
    "reddit": "Reddit / Social Media",
    "phone": "Phone",
    "side_project": "Side Project",
    "other": "Other",
}


# ---------------------------------------------------------------------------
# Session Data
# ---------------------------------------------------------------------------

@dataclass
class FocusSession:
    task_description: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    work_minutes: int = DEFAULT_WORK_MINUTES
    break_minutes: int = DEFAULT_BREAK_MINUTES
    pomodoros_completed: int = 0
    pomodoros_broken: int = 0
    distractions: list = field(default_factory=list)
    lockdown_active: bool = False
    lockdown_until: Optional[float] = None
    current_phase: str = "work"     # work | break | complete
    phase_start: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60.0

    @property
    def total_focused_minutes(self) -> int:
        return self.pomodoros_completed * self.work_minutes

    @property
    def focus_score(self) -> float:
        """0–100 productivity score based on completed vs. broken pomodoros."""
        total = self.pomodoros_completed + self.pomodoros_broken
        if total == 0:
            return 100.0
        return round((self.pomodoros_completed / total) * 100, 1)

    @property
    def phase_elapsed_minutes(self) -> float:
        return (time.time() - self.phase_start) / 60.0

    @property
    def phase_remaining_minutes(self) -> float:
        target = self.work_minutes if self.current_phase == "work" else self.break_minutes
        return max(0.0, target - self.phase_elapsed_minutes)


# ---------------------------------------------------------------------------
# Focus Guard
# ---------------------------------------------------------------------------

class FocusGuard:
    """
    Manages Pomodoro focus sessions and distraction intervention.

    Usage:
        guard = FocusGuard(s4_memory)
        guard.start_session("Study DSP Chapter 3")
        # ... after 25 min ...
        guard.complete_pomodoro()
        guard.record_distraction("youtube")
    """

    def __init__(
        self,
        s4_memory: S4MemoryManager,
        work_minutes: int = DEFAULT_WORK_MINUTES,
        break_minutes: int = DEFAULT_BREAK_MINUTES,
        max_distractions: int = DEFAULT_MAX_DISTRACTIONS,
        lockdown_minutes: int = DEFAULT_LOCKDOWN_MINUTES,
    ):
        self._mem = s4_memory
        self._work_min = work_minutes
        self._break_min = break_minutes
        self._max_distractions = max_distractions
        self._lockdown_min = lockdown_minutes
        self._session: Optional[FocusSession] = None
        logger.info("FocusGuard initialized (work=%dmin, break=%dmin)", work_minutes, break_minutes)

    # -------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------

    def start_session(self, task_description: str) -> FocusSession:
        """Start a new focus session."""
        self._session = FocusSession(
            task_description=task_description,
            work_minutes=self._work_min,
            break_minutes=self._break_min,
        )
        self._mem.set_current_task(task_description)
        logger.info("Focus session started: '%s'", task_description)
        return self._session

    def stop_session(self) -> Optional[FocusSession]:
        """End the current session and persist stats."""
        if not self._session:
            return None
        self._session.end_time = time.time()
        self._session.current_phase = "complete"

        # Persist to daily log
        self._mem.record_pomodoro(completed=True)  # Count last incomplete as done if stopped
        self._mem.set_current_task(None)

        logger.info(
            "Focus session ended: %d pomodoros, focus score %.0f%%",
            self._session.pomodoros_completed,
            self._session.focus_score
        )
        completed = self._session
        self._session = None
        return completed

    def has_active_session(self) -> bool:
        """True if a focus session is in progress."""
        return self._session is not None and self._session.is_active

    def get_session(self) -> Optional[FocusSession]:
        return self._session

    # -------------------------------------------------------------------
    # Pomodoro Transitions
    # -------------------------------------------------------------------

    def tick(self) -> dict:
        """
        Check current phase state. Call this periodically.
        Returns a status dict with phase, remaining time, and any action needed.
        """
        if not self._session:
            return {"status": "no_session"}

        remaining = self._session.phase_remaining_minutes
        phase = self._session.current_phase

        if remaining <= 0:
            if phase == "work":
                return {"status": "pomodoro_complete", "action": "take_break"}
            elif phase == "break":
                return {"status": "break_complete", "action": "start_work"}

        return {
            "status": "active",
            "phase": phase,
            "remaining_minutes": round(remaining, 1),
            "task": self._session.task_description,
            "pomodoros": self._session.pomodoros_completed,
            "focus_score": self._session.focus_score,
        }

    def complete_pomodoro(self):
        """Called when a 25-min work block completes."""
        if not self._session:
            return
        self._session.pomodoros_completed += 1
        self._session.current_phase = "break"
        self._session.phase_start = time.time()
        self._mem.record_pomodoro(completed=True)
        logger.debug("Pomodoro %d completed", self._session.pomodoros_completed)

    def complete_break(self):
        """Called when a break period ends — restart work phase."""
        if not self._session:
            return
        self._session.current_phase = "work"
        self._session.phase_start = time.time()
        logger.debug("Break complete, returning to work")

    def break_pomodoro(self):
        """Called when the work phase is interrupted early."""
        if not self._session:
            return
        self._session.pomodoros_broken += 1
        self._mem.record_pomodoro(completed=False)
        logger.debug("Pomodoro broken (%d broken total)", self._session.pomodoros_broken)

    # -------------------------------------------------------------------
    # Distraction Detection & Intervention
    # -------------------------------------------------------------------

    def check_in(self) -> dict:
        """
        Periodic check-in. Returns the check-in prompt and lockdown status.
        Called by Model E (Rapid) every 25 minutes.
        """
        if not self._session:
            return {"message": "No active focus session.", "locked_down": False}

        if self.is_locked_down():
            unlock_in = self._session.lockdown_until - time.time()
            return {
                "message": f"🔒 LOCKDOWN active for {int(unlock_in / 60)} more minutes. Stay focused.",
                "locked_down": True,
                "unlock_in_minutes": int(unlock_in / 60),
            }

        return {
            "message": f"Still on task? Task: '{self._session.task_description}'",
            "locked_down": False,
            "pomodoros": self._session.pomodoros_completed,
            "remaining_today": self._mem.get_daily_log().total_study_hours,
        }

    def record_distraction(self, distraction_type: str = "other", duration_min: int = 0) -> dict:
        """
        Record a distraction event.

        Returns:
            dict with intervention message and whether lockdown was triggered.
        """
        if not self._session:
            return {"message": "No active session to record distraction for."}

        now_str = datetime.now().strftime("%H:%M")
        event = {
            "time": now_str,
            "type": distraction_type,
            "duration_min": duration_min,
        }
        self._session.distractions.append(event)
        self._mem.record_distraction_event(distraction_type, duration_min)
        self.break_pomodoro()

        distraction_count = len(self._session.distractions)
        logger.warning(
            "Distraction #%d recorded: %s (%d min)",
            distraction_count, distraction_type, duration_min
        )

        # Check lockdown trigger
        if distraction_count >= self._max_distractions:
            return self._trigger_lockdown()

        remaining_allowed = self._max_distractions - distraction_count
        label = DISTRACTION_TYPES.get(distraction_type, distraction_type)
        return {
            "locked_down": False,
            "distraction_count": distraction_count,
            "message": (
                f"FOCUS ALERT: {label} detected. "
                f"Distraction #{distraction_count}/{self._max_distractions}. "
                f"{remaining_allowed} more before lockdown. "
                f"Return to: '{self._session.task_description}'"
            ),
        }

    def _trigger_lockdown(self) -> dict:
        """Activate lockdown mode."""
        if not self._session:
            return {}
        self._session.lockdown_active = True
        self._session.lockdown_until = time.time() + (self._lockdown_min * 60)
        logger.warning("FOCUS LOCKDOWN activated for %d minutes", self._lockdown_min)
        return {
            "locked_down": True,
            "lockdown_minutes": self._lockdown_min,
            "message": (
                f"🔒 LOCKDOWN: {len(self._session.distractions)} distractions recorded. "
                f"Non-essential JARVIS features suspended for {self._lockdown_min} minutes. "
                f"Return to: '{self._session.task_description}'. "
                f"Your exam requires full focus."
            ),
        }

    def is_locked_down(self) -> bool:
        """True if lockdown mode is active and not expired."""
        if not self._session or not self._session.lockdown_active:
            return False
        if time.time() > self._session.lockdown_until:
            self._session.lockdown_active = False
            self._session.lockdown_until = None
            logger.info("Lockdown expired")
            return False
        return True

    def unlock(self):
        """Manually unlock (admin override)."""
        if self._session:
            self._session.lockdown_active = False
            self._session.lockdown_until = None
            logger.info("Lockdown manually cleared")

    # -------------------------------------------------------------------
    # Session Stats
    # -------------------------------------------------------------------

    def get_session_stats(self) -> dict:
        """Return current session statistics."""
        if not self._session:
            return {"status": "no_session"}

        return {
            "task": self._session.task_description,
            "elapsed_minutes": round(self._session.elapsed_minutes, 1),
            "pomodoros_completed": self._session.pomodoros_completed,
            "pomodoros_broken": self._session.pomodoros_broken,
            "total_focused_minutes": self._session.total_focused_minutes,
            "distraction_count": len(self._session.distractions),
            "focus_score": self._session.focus_score,
            "locked_down": self.is_locked_down(),
            "phase": self._session.current_phase,
            "phase_remaining_minutes": round(self._session.phase_remaining_minutes, 1),
        }

    def get_productivity_rating(self) -> str:
        """Human-readable productivity rating from focus score."""
        if not self._session:
            return "No session"
        score = self._session.focus_score
        if score >= 90:
            return "🔥 Excellent"
        elif score >= 70:
            return "✅ Good"
        elif score >= 50:
            return "⚠ Fair"
        else:
            return "❌ Poor"
