"""
JARVIS System 4 — Weekly Review Engine
=========================================
5-phase Sunday review system:

  Phase 1: Data Collection    — aggregate daily logs for the week
  Phase 2: Strategic Analysis — Model B (Analyst) generates risk + opportunity report
  Phase 3: Plan Generation    — Model A (Chief) creates Week N+1 goals, B verifies
  Phase 4: Memory Update      — archive current week, write new weekly state
  Phase 5: Output             — formatted weekly report for terminal display

Usage:
    from backend.weekly_review import WeeklyReviewEngine
    engine = WeeklyReviewEngine(role_manager, s4_memory, academic_manager, ms_roadmap)
    report = engine.run()
    print(report)
"""

from datetime import datetime
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager, WeeklyState
from state.academic_manager import AcademicManager
from state.ms_roadmap import MSRoadmapManager

logger = get_logger(__name__)


class WeeklyReviewEngine:
    """
    Orchestrates the 5-phase weekly review.

    Args:
        role_manager: S4RoleManager for calling Chief and Analyst roles
        s4_memory: S4MemoryManager for data access and archival
        academic_manager: AcademicManager for academic context
        ms_roadmap: MSRoadmapManager for MS context
    """

    def __init__(
        self,
        role_manager,
        s4_memory: S4MemoryManager,
        academic_manager: AcademicManager,
        ms_roadmap: MSRoadmapManager,
    ):
        self._rm = role_manager
        self._mem = s4_memory
        self._academic = academic_manager
        self._ms = ms_roadmap

    # -------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------

    def run(self) -> str:
        """
        Execute the full 5-phase weekly review.

        Returns:
            str — complete formatted weekly report
        """
        logger.info("Starting weekly review")
        output_sections = []
        now = datetime.now()

        output_sections.append(
            f"╔══════════════════════════════════════════════╗\n"
            f"║   JARVIS S4 — WEEKLY REVIEW                 ║\n"
            f"║   {now.strftime('%A, %B %d, %Y'):44}║\n"
            f"╚══════════════════════════════════════════════╝"
        )

        # Phase 1: Data Collection
        logger.info("Weekly Review — Phase 1: Data Collection")
        output_sections.append("\n📊 PHASE 1 — DATA COLLECTION")
        data = self._phase1_collect()
        output_sections.append(self._format_phase1(data))

        # Phase 2: Strategic Analysis
        logger.info("Weekly Review — Phase 2: Strategic Analysis")
        output_sections.append("\n🔍 PHASE 2 — STRATEGIC ANALYSIS (Analyst)")
        analysis = self._phase2_analyze(data)
        output_sections.append(analysis)

        # Phase 3: Plan Generation
        logger.info("Weekly Review — Phase 3: Plan Generation")
        output_sections.append("\n📋 PHASE 3 — NEXT WEEK PLAN (Chief + Analyst)")
        plan = self._phase3_plan(data, analysis)
        output_sections.append(plan)

        # Phase 4: Memory Update
        logger.info("Weekly Review — Phase 4: Memory Update")
        output_sections.append("\n💾 PHASE 4 — MEMORY UPDATE")
        archive_file = self._phase4_update(plan)
        output_sections.append(f"  ✓ Week archived → {archive_file}")

        # Phase 5: Final Output
        output_sections.append("\n✅ PHASE 5 — COMPLETE")
        output_sections.append("  Weekly review finished. Next review: Sunday 8:00 PM.")

        report = "\n".join(output_sections)
        logger.info("Weekly review complete")
        return report

    # -------------------------------------------------------------------
    # Phase 1 — Data Collection
    # -------------------------------------------------------------------

    def _phase1_collect(self) -> dict:
        """Aggregate all data from this week's daily logs."""
        weekly = self._mem.aggregate_week_from_daily()
        academic_summary = self._academic.get_dashboard_summary()
        ms_summary = self._ms.get_dashboard_summary()

        return {
            "weekly": weekly,
            "academic": academic_summary,
            "ms": ms_summary,
            "completion_rate": (
                round(weekly.tasks_completed / weekly.tasks_planned * 100, 1)
                if weekly.tasks_planned > 0 else 0
            ),
        }

    def _format_phase1(self, data: dict) -> str:
        weekly: WeeklyState = data["weekly"]
        completion = data["completion_rate"]
        academic = data["academic"]
        ms = data["ms"]

        lines = [
            f"  Study Hours:      {weekly.study_hours_total:.1f}h / {weekly.study_target_hours:.0f}h target",
            f"  Task Completion:  {weekly.tasks_completed}/{weekly.tasks_planned} ({completion:.0f}%)",
            f"  Pomodoros Done:   {weekly.pomodoros_completed}",
            f"  Distractions:     {weekly.distraction_incidents}",
            f"  Papers Read:      {weekly.papers_read} / 2 target",
            f"  Coding Hours:     {weekly.coding_hours:.1f}h",
            f"  Subjects Covered: {', '.join(weekly.subjects_covered) or 'none'}",
        ]

        # Academic alerts
        if academic.get("alerts"):
            lines.append("")
            lines.append("  EXAM ALERTS:")
            for a in academic["alerts"]:
                lines.append(f"    {a['message']}")

        # CGPA
        cgpa = academic.get("cgpa", {})
        if cgpa.get("current"):
            lines.append(f"  CGPA: {cgpa['current']} → target {cgpa['target']}")

        # MS phase
        lines.append(f"  MS Phase: {ms['current_phase'].title()} ({ms['phase_progress'][ms['current_phase']]['percent']:.0f}% complete)")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Phase 2 — Strategic Analysis (Model B)
    # -------------------------------------------------------------------

    def _phase2_analyze(self, data: dict) -> str:
        """Ask Analyst to generate risk + opportunity assessment."""
        weekly: WeeklyState = data["weekly"]
        academic = data["academic"]
        ms = data["ms"]

        prompt = f"""Analyze this week's performance data for an ECE student targeting MS abroad.

WEEK DATA:
Study hours: {weekly.study_hours_total:.1f}h / {weekly.study_target_hours:.0f}h target
Task completion: {weekly.tasks_completed}/{weekly.tasks_planned} ({data['completion_rate']:.0f}%)
Distractions: {weekly.distraction_incidents}
Papers read: {weekly.papers_read}/2 target
Subjects covered: {', '.join(weekly.subjects_covered) or 'none'}
MS Phase: {ms['current_phase']} ({ms['phase_progress'][ms['current_phase']]['percent']:.0f}% done)
Exam alerts: {len(academic.get('alerts', []))} upcoming exams

Provide:
1. RISK ASSESSMENT: What is the biggest risk this student faces going into next week?
2. OPPORTUNITY: What is the single highest-leverage action available?
3. PATTERNS: Any concerning behavioral patterns? (distraction, avoidance, etc.)
4. MS READINESS: Is the student on track for MS application readiness?
5. RECOMMENDATION: One concrete decision to make before next week starts.

Be direct. Max 250 words."""

        try:
            result = self._rm.call_role("analyst", prompt)
            if result.success and result.content.strip():
                return self._indent(result.content.strip())
        except Exception as e:
            logger.error("Phase 2 analysis failed: %s", e)

        return self._fallback_analysis(data)

    def _fallback_analysis(self, data: dict) -> str:
        weekly: WeeklyState = data["weekly"]
        lines = ["  [Analyst offline — rule-based assessment]", ""]
        if weekly.study_hours_total < (weekly.study_target_hours * 0.7):
            lines.append("  ⚠ RISK: Study hours significantly below target.")
        if weekly.distraction_incidents > 5:
            lines.append("  ⚠ PATTERN: High distraction rate detected.")
        if weekly.papers_read < 2:
            lines.append("  ⚠ MS RISK: Research reading target not met.")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Phase 3 — Plan Generation (Model A + B verify)
    # -------------------------------------------------------------------

    def _phase3_plan(self, data: dict, analysis: str) -> str:
        """Chief generates next week's plan; Analyst verifies time allocation."""
        weekly: WeeklyState = data["weekly"]
        academic = data["academic"]

        # Upcoming exams context
        exam_context = ""
        if academic.get("upcoming_exams"):
            exam_context = "UPCOMING EXAMS:\n" + "\n".join(
                f"  {e['subject']} on {e['date']}"
                for e in academic["upcoming_exams"][:3]
            )

        prompt = f"""Based on this analysis, create next week's plan for this ECE student.

ANALYST ASSESSMENT:
{analysis}

{exam_context}

Current weekly goals (if any): {', '.join(weekly.goals) if weekly.goals else 'none set'}

Generate:
1. WEEKLY THEME: One-sentence focus for the week
2. 5 WEEKLY GOALS: Specific, achievable goals (academic + research + JARVIS dev + MS prep)
3. STUDY ALLOCATION: Which subjects need the most time and why
4. DAILY MUST-DO: The one non-negotiable task every single day
5. RISK MITIGATION: How to handle the biggest risk the analyst identified

Format goals as a numbered list. Max 300 words."""

        try:
            result = self._rm.call_role("chief", prompt)
            if result.success and result.content.strip():
                plan_text = result.content.strip()

                # Verify with Analyst if available
                verify_prompt = (
                    f"Review this weekly plan for an ECE student. "
                    f"Is the time allocation realistic given:\n"
                    f"- 6h daily study target\n"
                    f"- Exam alerts: {len(academic.get('alerts', []))}\n\n"
                    f"PLAN:\n{plan_text}\n\n"
                    f"State: APPROVED (plan is realistic) or "
                    f"ADJUSTED: [specific change needed]. Max 50 words."
                )
                verify_result = self._rm.call_role("analyst", verify_prompt)
                if verify_result.success and "ADJUSTED:" in verify_result.content:
                    adjustment = verify_result.content.split("ADJUSTED:")[-1].strip()
                    plan_text += f"\n\n  [Analyst note] {adjustment}"

                return self._indent(plan_text)
        except Exception as e:
            logger.error("Phase 3 plan generation failed: %s", e)

        return self._fallback_plan(data)

    def _fallback_plan(self, data: dict) -> str:
        weekly: WeeklyState = data["weekly"]
        lines = ["  [Chief offline — template plan]", ""]
        lines.append("  WEEKLY GOALS:")
        lines.append("  1. Complete current subject chapter (academic)")
        lines.append("  2. Read 2 research papers (research)")
        lines.append("  3. JARVIS: implement one S4 feature (development)")
        lines.append("  4. GRE practice 30 min/day (MS prep)")
        lines.append("  5. Log all habits daily (consistency)")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Phase 4 — Memory Update
    # -------------------------------------------------------------------

    def _phase4_update(self, plan_text: str) -> str:
        """Archive week and set next week goals from plan."""
        # Extract goals from plan text (simple heuristic — numbered lines)
        import re
        goal_lines = re.findall(r"^\s*\d+\.\s+(.+)$", plan_text, re.MULTILINE)
        next_goals = goal_lines[:5] if goal_lines else []

        # Update weekly state with next week goals before archiving
        state = self._mem.get_weekly_state()
        state.analyst_report = plan_text
        state.next_week_goals = next_goals
        self._mem.save_weekly_state(state)

        # Archive
        archive_file = self._mem.archive_week()
        return archive_file

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _indent(text: str, spaces: int = 2) -> str:
        """Indent all lines of text by N spaces."""
        prefix = " " * spaces
        return "\n".join(prefix + line for line in text.splitlines())
