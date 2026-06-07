"""
JARVIS System 4 — MS Abroad Roadmap Manager
=============================================
Tracks all MS abroad preparation state:
  - Target universities
  - Phase-based milestone tracking (Foundation → Research → Application → Decision)
  - GRE / TOEFL status and scores
  - Research experience and publications
  - Monthly checkpoint generation
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class University:
    name: str
    country: str = ""
    ranking: Optional[int] = None
    gre_required: bool = True
    gre_min: Optional[int] = None
    toefl_min: Optional[int] = None
    deadline: Optional[str] = None     # YYYY-MM-DD
    program: str = ""                   # e.g., "MS ECE - AIoT Track"
    funding_available: bool = False
    notes: str = ""
    added: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))


@dataclass
class ResearchExperience:
    title: str
    description: str
    start_date: str
    end_date: Optional[str] = None
    ongoing: bool = False
    advisor: str = ""
    outcome: str = ""       # paper | demo | report | internship | none


@dataclass
class MSMilestone:
    id: str
    phase: str              # foundation | research | application | decision
    description: str
    done: bool = False
    done_date: Optional[str] = None
    priority: int = 2       # 1=high, 2=medium, 3=low


# Default milestones (used to initialize ms_roadmap.json)
DEFAULT_MILESTONES = [
    # Foundation Phase
    MSMilestone("m1", "foundation", "Identify 10 target universities (AIoT/Edge AI programs)"),
    MSMilestone("m2", "foundation", "Start GRE prep (30 min/day minimum)"),
    MSMilestone("m3", "foundation", "Read 2 research papers/week for 4 consecutive weeks"),
    MSMilestone("m4", "foundation", "Make JARVIS demo-ready for portfolio", priority=1),
    MSMilestone("m5", "foundation", "Create GitHub profile showcasing ECE + AI projects"),
    # Research Phase
    MSMilestone("m6", "research", "First professor cold email outreach (min 5 emails)", priority=1),
    MSMilestone("m7", "research", "Contribute to 1 open-source Edge AI / TinyML project"),
    MSMilestone("m8", "research", "Submit or present a research paper"),
    MSMilestone("m9", "research", "Complete GRE practice test (target score)", priority=1),
    # Application Phase
    MSMilestone("m10", "application", "SOP first draft complete"),
    MSMilestone("m11", "application", "GRE exam taken", priority=1),
    MSMilestone("m12", "application", "TOEFL / IELTS exam taken", priority=1),
    MSMilestone("m13", "application", "Resume tailored for MS applications"),
    MSMilestone("m14", "application", "LOR requests sent to 3 professors"),
    # Decision Phase
    MSMilestone("m15", "decision", "All applications submitted"),
    MSMilestone("m16", "decision", "Visa and financial planning complete"),
    MSMilestone("m17", "decision", "Final university decision made"),
]


# ---------------------------------------------------------------------------
# MS Roadmap Manager
# ---------------------------------------------------------------------------

class MSRoadmapManager:
    """
    Manages the MS abroad preparation roadmap.

    Phase progression is automatic based on milestone completion.
    Data is persisted to data/s4_profile/ms_roadmap.json.
    """

    PHASES = ["foundation", "research", "application", "decision"]

    def __init__(self, s4_memory: S4MemoryManager, settings=None):
        self._mem = s4_memory
        self._settings = settings
        self._ensure_initialized()
        logger.info("MSRoadmapManager initialized")

    def _ensure_initialized(self):
        """Initialize ms_roadmap.json with defaults if it doesn't exist."""
        data = self._mem.get_ms_roadmap()
        if not data.get("version"):
            self._initialize_roadmap()

    def _initialize_roadmap(self):
        """Create the default roadmap structure."""
        data = {
            "version": "4.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "target_year": 2028,
            "current_phase": "foundation",
            "milestones": [asdict(m) for m in DEFAULT_MILESTONES],
            "universities": [],
            "gre": {"status": "not_started", "score": None, "target": 320},
            "toefl": {"status": "not_started", "score": None, "target": 105},
            "research_experience": [],
            "publications": [],
            "monthly_checkpoints": [],
            "target_domains": ["AIoT", "Edge AI", "TinyML", "Embedded Systems"],
        }
        self._mem._save_profile("ms_roadmap.json", data)
        logger.info("MS roadmap initialized with defaults")

    # -------------------------------------------------------------------
    # Phase & Progress
    # -------------------------------------------------------------------

    def get_current_phase(self) -> str:
        """Return the current MS preparation phase."""
        data = self._mem.get_ms_roadmap()
        return data.get("current_phase", "foundation")

    def get_phase_progress(self, phase: str = None) -> dict:
        """
        Return completion stats for a phase (or current phase).
        Returns: {phase, total, done, percent, pending_milestones}
        """
        phase = phase or self.get_current_phase()
        data = self._mem.get_ms_roadmap()
        milestones = [m for m in data.get("milestones", []) if m.get("phase") == phase]
        total = len(milestones)
        done = sum(1 for m in milestones if m.get("done"))
        pending = [m for m in milestones if not m.get("done")]
        return {
            "phase": phase,
            "total": total,
            "done": done,
            "percent": round((done / total) * 100, 1) if total else 0,
            "pending_milestones": pending,
        }

    def get_all_milestones(self) -> list[MSMilestone]:
        """Return all milestones."""
        data = self._mem.get_ms_roadmap()
        return [MSMilestone(**{k: v for k, v in m.items()
                               if k in MSMilestone.__dataclass_fields__})
                for m in data.get("milestones", [])]

    def complete_milestone(self, milestone_id: str) -> bool:
        """Mark a milestone as complete."""
        data = self._mem.get_ms_roadmap()
        milestones = data.get("milestones", [])
        for m in milestones:
            if m.get("id") == milestone_id:
                m["done"] = True
                m["done_date"] = datetime.now().strftime("%Y-%m-%d")
                data["milestones"] = milestones
                self._mem._save_profile("ms_roadmap.json", data)
                logger.info("MS milestone completed: %s", milestone_id)
                self._check_phase_advancement(data)
                return True
        return False

    def _check_phase_advancement(self, data: dict):
        """Auto-advance phase if all milestones in current phase are done."""
        current = data.get("current_phase", "foundation")
        idx = self.PHASES.index(current) if current in self.PHASES else 0
        if idx < len(self.PHASES) - 1:
            phase_milestones = [m for m in data.get("milestones", [])
                                if m.get("phase") == current]
            all_done = all(m.get("done") for m in phase_milestones)
            if all_done:
                next_phase = self.PHASES[idx + 1]
                data["current_phase"] = next_phase
                self._mem._save_profile("ms_roadmap.json", data)
                logger.info("MS phase advanced to: %s", next_phase)

    # -------------------------------------------------------------------
    # Universities
    # -------------------------------------------------------------------

    def add_university(self, name: str, **kwargs) -> University:
        """Add a target university."""
        data = self._mem.get_ms_roadmap()
        universities = data.get("universities", [])
        # Prevent duplicates
        if any(u.get("name", "").lower() == name.lower() for u in universities):
            logger.info("University already exists: %s", name)
            existing = next(u for u in universities if u.get("name", "").lower() == name.lower())
            return University(**{k: v for k, v in existing.items()
                                 if k in University.__dataclass_fields__})

        university = University(name=name, **kwargs)
        universities.append(asdict(university))
        data["universities"] = universities
        self._mem._save_profile("ms_roadmap.json", data)
        logger.info("University added: %s", name)
        return university

    def list_universities(self) -> list[University]:
        """Return all target universities."""
        data = self._mem.get_ms_roadmap()
        return [University(**{k: v for k, v in u.items()
                              if k in University.__dataclass_fields__})
                for u in data.get("universities", [])]

    # -------------------------------------------------------------------
    # Test Scores
    # -------------------------------------------------------------------

    def update_gre(self, score: int = None, status: str = None, target: int = None):
        """Update GRE status/score/target."""
        data = self._mem.get_ms_roadmap()
        gre = data.get("gre", {})
        if score is not None:
            gre["score"] = score
            gre["status"] = "taken"
        if status:
            gre["status"] = status
        if target is not None:
            gre["target"] = target
        data["gre"] = gre
        self._mem._save_profile("ms_roadmap.json", data)

    def update_toefl(self, score: int = None, status: str = None, target: int = None):
        """Update TOEFL status/score/target."""
        data = self._mem.get_ms_roadmap()
        toefl = data.get("toefl", {})
        if score is not None:
            toefl["score"] = score
            toefl["status"] = "taken"
        if status:
            toefl["status"] = status
        if target is not None:
            toefl["target"] = target
        data["toefl"] = toefl
        self._mem._save_profile("ms_roadmap.json", data)

    # -------------------------------------------------------------------
    # Research Experience
    # -------------------------------------------------------------------

    def add_research_experience(self, title: str, description: str,
                                 start_date: str, **kwargs) -> ResearchExperience:
        """Log a research experience."""
        data = self._mem.get_ms_roadmap()
        experience = ResearchExperience(
            title=title, description=description, start_date=start_date, **kwargs
        )
        experiences = data.get("research_experience", [])
        experiences.append(asdict(experience))
        data["research_experience"] = experiences
        self._mem._save_profile("ms_roadmap.json", data)
        logger.info("Research experience added: %s", title)
        return experience

    # -------------------------------------------------------------------
    # Monthly Checkpoint
    # -------------------------------------------------------------------

    def generate_monthly_checkpoint_data(self) -> dict:
        """
        Return a structured data dict for the monthly MS checkpoint.
        This is fed to Model B (Analyst) for assessment.
        """
        data = self._mem.get_ms_roadmap()
        phase_progress = self.get_phase_progress()
        weekly = self._mem.get_weekly_state()
        academic = self._mem.get_academic_map()

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "current_phase": self.get_current_phase(),
            "phase_progress_percent": phase_progress["percent"],
            "pending_milestones": phase_progress["pending_milestones"],
            "universities_identified": len(data.get("universities", [])),
            "gre": data.get("gre", {}),
            "toefl": data.get("toefl", {}),
            "papers_read_this_week": weekly.papers_read,
            "research_experiences": len(data.get("research_experience", [])),
            "current_cgpa": academic.get("current_cgpa"),
            "target_cgpa": 8.5,
        }

    def record_monthly_checkpoint(self, analyst_assessment: str):
        """Save the Analyst's monthly assessment to the roadmap."""
        data = self._mem.get_ms_roadmap()
        checkpoints = data.get("monthly_checkpoints", [])
        checkpoints.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "phase": self.get_current_phase(),
            "assessment": analyst_assessment,
        })
        data["monthly_checkpoints"] = checkpoints
        self._mem._save_profile("ms_roadmap.json", data)

    # -------------------------------------------------------------------
    # Dashboard Summary
    # -------------------------------------------------------------------

    def get_dashboard_summary(self) -> dict:
        """Full MS roadmap summary for CLI/dashboard display."""
        data = self._mem.get_ms_roadmap()
        return {
            "current_phase": self.get_current_phase(),
            "target_year": data.get("target_year"),
            "phase_progress": {
                phase: self.get_phase_progress(phase)
                for phase in self.PHASES
            },
            "universities": [asdict(u) for u in self.list_universities()],
            "gre": data.get("gre", {}),
            "toefl": data.get("toefl", {}),
            "research_count": len(data.get("research_experience", [])),
            "publications_count": len(data.get("publications", [])),
            "target_domains": data.get("target_domains", []),
        }
