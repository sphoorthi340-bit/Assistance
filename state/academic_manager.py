"""
JARVIS System 4 — Academic State Manager
=========================================
Tracks all academic state for the ECE student:
  - Subjects (syllabus completion, weak/strong topics)
  - Exam dates and assignment deadlines
  - CGPA history
  - Emergency mode detection (exam within N days)
  - Study session logging
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Subject:
    name: str
    full_name: str = ""
    completion_percent: float = 0.0
    weak_topics: list = field(default_factory=list)
    strong_topics: list = field(default_factory=list)
    exam_date: Optional[str] = None         # ISO date string YYYY-MM-DD
    assignment_deadlines: list = field(default_factory=list)
    study_minutes_total: int = 0
    last_studied: Optional[str] = None


@dataclass
class ExamEntry:
    subject: str
    date: str       # YYYY-MM-DD
    time: str = ""
    venue: str = ""
    syllabus_coverage: float = 0.0  # 0-1 how much is covered


# ---------------------------------------------------------------------------
# Academic Manager
# ---------------------------------------------------------------------------

class AcademicManager:
    """
    Manages ECE academic state including subjects, exams, CGPA, and
    emergency mode detection.

    Data is persisted to data/s4_profile/academic_map.json via S4MemoryManager.
    """

    EMERGENCY_DAYS = 5      # Trigger exam emergency mode when exam within N days
    ALERT_DAYS = 14         # Show alert when exam within N days

    def __init__(self, s4_memory: S4MemoryManager, settings=None):
        self._mem = s4_memory
        self._settings = settings
        logger.info("AcademicManager initialized")

    # -------------------------------------------------------------------
    # Subjects
    # -------------------------------------------------------------------

    def add_subject(self, name: str, full_name: str = "") -> Subject:
        """Add or update a subject."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        if name not in subjects:
            subject = Subject(name=name, full_name=full_name or name)
            subjects[name] = asdict(subject)
            logger.info("Added subject: %s", name)
        else:
            if full_name:
                subjects[name]["full_name"] = full_name
        data["subjects"] = subjects
        self._mem._save_profile("academic_map.json", data)
        return Subject(**{k: v for k, v in subjects[name].items()
                          if k in Subject.__dataclass_fields__})

    def get_subject(self, name: str) -> Optional[Subject]:
        """Retrieve a subject by name."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        if name in subjects:
            return Subject(**{k: v for k, v in subjects[name].items()
                              if k in Subject.__dataclass_fields__})
        return None

    def list_subjects(self) -> list[Subject]:
        """List all tracked subjects."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        return [Subject(**{k: v for k, v in s.items()
                           if k in Subject.__dataclass_fields__})
                for s in subjects.values()]

    def update_subject_completion(self, subject_name: str, completion_percent: float):
        """Update how much of a subject's syllabus has been covered."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        if subject_name in subjects:
            subjects[subject_name]["completion_percent"] = min(100.0, max(0.0, completion_percent))
            data["subjects"] = subjects
            self._mem._save_profile("academic_map.json", data)

    def add_weak_topic(self, subject_name: str, topic: str):
        """Mark a topic as weak for a subject."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        if subject_name in subjects:
            weak = subjects[subject_name].get("weak_topics", [])
            if topic not in weak:
                weak.append(topic)
            subjects[subject_name]["weak_topics"] = weak
            data["subjects"] = subjects
            self._mem._save_profile("academic_map.json", data)

    def log_study_session(self, subject_name: str, duration_minutes: int):
        """Record a study session for a subject."""
        data = self._mem.get_academic_map()
        subjects = data.get("subjects", {})
        if subject_name in subjects:
            subjects[subject_name]["study_minutes_total"] = (
                subjects[subject_name].get("study_minutes_total", 0) + duration_minutes
            )
            subjects[subject_name]["last_studied"] = datetime.now().strftime("%Y-%m-%d")
            data["subjects"] = subjects
            self._mem._save_profile("academic_map.json", data)
        # Also record in daily log
        self._mem.record_study(subject_name, duration_minutes)

    # -------------------------------------------------------------------
    # Exams & Deadlines
    # -------------------------------------------------------------------

    def add_exam(self, subject: str, date_str: str, time_str: str = "", venue: str = "") -> ExamEntry:
        """
        Add an exam date for a subject.

        Args:
            subject: Subject name (must match a tracked subject)
            date_str: Date in YYYY-MM-DD format
            time_str: Optional exam time (e.g., "09:00 AM")
            venue: Optional venue
        """
        data = self._mem.get_academic_map()

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.")

        exam = ExamEntry(subject=subject, date=date_str, time=time_str, venue=venue)

        # Update the subject's exam_date
        subjects = data.get("subjects", {})
        if subject in subjects:
            subjects[subject]["exam_date"] = date_str
            data["subjects"] = subjects

        # Add to upcoming_exams list
        exams = data.get("upcoming_exams", [])
        # Remove existing entry for same subject if present
        exams = [e for e in exams if e.get("subject") != subject]
        exams.append(asdict(exam))
        # Sort by date
        exams.sort(key=lambda e: e["date"])
        data["upcoming_exams"] = exams

        self._mem._save_profile("academic_map.json", data)
        logger.info("Exam added: %s on %s", subject, date_str)
        return exam

    def get_upcoming_exams(self, within_days: int = 30) -> list[ExamEntry]:
        """Return exams within the next N days, sorted by date."""
        data = self._mem.get_academic_map()
        exams = data.get("upcoming_exams", [])
        today = datetime.now().date()
        cutoff = today + timedelta(days=within_days)

        upcoming = []
        for e in exams:
            try:
                exam_date = datetime.strptime(e["date"], "%Y-%m-%d").date()
                if today <= exam_date <= cutoff:
                    upcoming.append(ExamEntry(**{k: v for k, v in e.items()
                                                  if k in ExamEntry.__dataclass_fields__}))
            except (ValueError, KeyError):
                continue

        return sorted(upcoming, key=lambda e: e.date)

    def days_until_exam(self, subject: str) -> Optional[int]:
        """Return days until the nearest exam for a subject, or None."""
        exams = self.get_upcoming_exams(within_days=365)
        for exam in exams:
            if exam.subject.lower() == subject.lower():
                exam_date = datetime.strptime(exam.date, "%Y-%m-%d").date()
                return (exam_date - datetime.now().date()).days
        return None

    def nearest_exam(self) -> Optional[ExamEntry]:
        """Return the next exam regardless of subject."""
        exams = self.get_upcoming_exams(within_days=365)
        return exams[0] if exams else None

    # -------------------------------------------------------------------
    # Emergency Mode
    # -------------------------------------------------------------------

    def is_exam_mode(self) -> bool:
        """
        Returns True if any exam is within EMERGENCY_DAYS.
        When True, the S4 Dispatcher boosts academic routing.
        """
        nearest = self.nearest_exam()
        if nearest is None:
            return False
        exam_date = datetime.strptime(nearest.date, "%Y-%m-%d").date()
        days_left = (exam_date - datetime.now().date()).days
        return 0 <= days_left <= self.EMERGENCY_DAYS

    def get_exam_alerts(self) -> list[dict]:
        """
        Returns alert messages for any exam within ALERT_DAYS.
        Each alert includes the subject, days remaining, and urgency level.
        """
        alerts = []
        exams = self.get_upcoming_exams(within_days=self.ALERT_DAYS)
        for exam in exams:
            exam_date = datetime.strptime(exam.date, "%Y-%m-%d").date()
            days_left = (exam_date - datetime.now().date()).days
            urgency = "critical" if days_left <= self.EMERGENCY_DAYS else \
                      "high" if days_left <= 7 else "medium"
            alerts.append({
                "subject": exam.subject,
                "date": exam.date,
                "days_left": days_left,
                "urgency": urgency,
                "message": f"[{urgency.upper()}] {exam.subject} exam in {days_left} day{'s' if days_left != 1 else ''}",
            })
        return alerts

    def get_at_risk_subjects(self, completion_threshold: float = 50.0) -> list[Subject]:
        """Return subjects with completion below threshold."""
        subjects = self.list_subjects()
        return [s for s in subjects if s.completion_percent < completion_threshold]

    # -------------------------------------------------------------------
    # CGPA
    # -------------------------------------------------------------------

    def update_cgpa(self, semester: int, cgpa: float):
        """Record CGPA for a semester."""
        data = self._mem.get_academic_map()
        history = data.get("cgpa_history", [])
        # Update or add
        updated = False
        for entry in history:
            if entry.get("semester") == semester:
                entry["cgpa"] = cgpa
                updated = True
                break
        if not updated:
            history.append({"semester": semester, "cgpa": cgpa})
        history.sort(key=lambda e: e["semester"])
        data["cgpa_history"] = history
        data["current_cgpa"] = cgpa  # Latest is current
        self._mem._save_profile("academic_map.json", data)

        # Also update user profile
        profile = self._mem.get_user_profile()
        goals = profile.get("goals", {})
        goals["current_cgpa"] = cgpa
        profile["goals"] = goals
        self._mem._save_profile("user_profile.json", profile)
        logger.info("CGPA updated: Semester %d → %.2f", semester, cgpa)

    def get_cgpa_status(self) -> dict:
        """Return CGPA summary: current, target, gap."""
        data = self._mem.get_academic_map()
        profile = self._mem.get_user_profile()
        current = data.get("current_cgpa")
        target = profile.get("goals", {}).get("target_cgpa", 8.5)
        return {
            "current": current,
            "target": target,
            "gap": round(target - current, 2) if current else None,
            "on_track": current >= target if current else None,
            "history": data.get("cgpa_history", []),
        }

    # -------------------------------------------------------------------
    # Summary for Display
    # -------------------------------------------------------------------

    def get_dashboard_summary(self) -> dict:
        """Return a complete academic summary dict for dashboard/CLI display."""
        return {
            "subjects": [asdict(s) for s in self.list_subjects()],
            "upcoming_exams": [asdict(e) for e in self.get_upcoming_exams()],
            "alerts": self.get_exam_alerts(),
            "at_risk_subjects": [asdict(s) for s in self.get_at_risk_subjects()],
            "cgpa": self.get_cgpa_status(),
            "exam_mode": self.is_exam_mode(),
        }
