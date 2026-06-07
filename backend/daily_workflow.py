"""
JARVIS System 4 — Daily Workflow Engine
=========================================
Generates S4-powered morning briefs and evening wraps.

Morning Brief (Model A — Chief):
  - Today's 3 MUST-DO tasks (weighted by deadline + importance)
  - Exam/assignment countdown alerts
  - Focus timer targets
  - Quick MS milestone check

Evening Wrap (Model A + E):
  - Session summary (tasks done, hours studied)
  - Wins and blockers log
  - Tomorrow preview
  - Streak and habit check
"""

from datetime import datetime
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager
from state.academic_manager import AcademicManager
from state.ms_roadmap import MSRoadmapManager
from backend.focus_guard import FocusGuard

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Morning Brief Generator
# ---------------------------------------------------------------------------

def generate_s4_morning_brief(
    role_manager,
    s4_memory: S4MemoryManager,
    academic_manager: AcademicManager,
    ms_roadmap: MSRoadmapManager,
    goal_manager=None,
    habit_manager=None,
) -> str:
    """
    Generate a morning brief using Model A (Chief of Staff).

    Args:
        role_manager: S4RoleManager instance
        s4_memory: S4MemoryManager
        academic_manager: AcademicManager
        ms_roadmap: MSRoadmapManager
        goal_manager: Optional GoalManager (for active goals context)
        habit_manager: Optional HabitManager (for habit status)

    Returns:
        str — formatted morning brief text
    """
    now = datetime.now()
    today_log = s4_memory.get_daily_log()
    weekly = s4_memory.get_weekly_state()

    # --- Build context block ---
    context_lines = [
        f"=== MORNING BRIEF DATA — {now.strftime('%A, %B %d, %Y')} ===",
        "",
    ]

    # Exam alerts
    alerts = academic_manager.get_exam_alerts()
    if alerts:
        context_lines.append("EXAM ALERTS:")
        for a in alerts:
            context_lines.append(f"  [{a['urgency'].upper()}] {a['subject']} — {a['days_left']} days")
        context_lines.append("")

    # Weekly goals progress
    if weekly.goals:
        context_lines.append("THIS WEEK'S GOALS:")
        for goal in weekly.goals:
            progress = weekly.goal_progress.get(goal, 0.0)
            bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
            context_lines.append(f"  {bar} {int(progress * 100)}% — {goal}")
        context_lines.append("")

    # Weekly study pace
    study_pace = (
        f"Study this week: {weekly.study_hours_total:.1f}h / {weekly.study_target_hours:.0f}h target"
    )
    context_lines.append(study_pace)

    # MS milestone check
    phase_progress = ms_roadmap.get_phase_progress()
    if phase_progress["pending_milestones"]:
        next_milestone = phase_progress["pending_milestones"][0]
        context_lines.append(
            f"Next MS milestone: [{phase_progress['phase'].upper()}] {next_milestone.get('description', '')}"
        )

    # Active goals from GoalManager
    if goal_manager:
        try:
            active_goals = goal_manager.list_goals(status="active")
            priority_goals = sorted(active_goals, key=lambda g: g.get("priority", 5))[:3]
            if priority_goals:
                context_lines.append("\nACTIVE GOALS (top 3 by priority):")
                for g in priority_goals:
                    deadline = f" — due {g['deadline'][:10]}" if g.get("deadline") else ""
                    context_lines.append(f"  • {g['title']}{deadline}")
        except Exception:
            pass

    # Habit status
    if habit_manager:
        try:
            habits = habit_manager.list_habits(active_only=True)
            if habits:
                context_lines.append(f"\nActive habits: {', '.join(h['name'] for h in habits)}")
        except Exception:
            pass

    context_lines.append("")
    context_lines.append(
        "Generate a concise morning brief. Identify the 3 MUST-DO tasks for today. "
        "Be direct and motivating. No more than 200 words total."
    )

    prompt = "\n".join(context_lines)

    try:
        result = role_manager.call_role(
            "chief",
            prompt,
            conversation_history=[],
            context_kwargs={"state_snapshot": s4_memory.get_s4_context_snapshot()},
        )
        if result.success and result.content.strip():
            brief_text = result.content.strip()
        else:
            brief_text = _fallback_morning_brief(today_log, weekly, alerts)
    except Exception as e:
        logger.error("Morning brief generation failed: %s", e)
        brief_text = _fallback_morning_brief(today_log, weekly, alerts)

    # Mark in daily log
    log = s4_memory.get_daily_log()
    log.morning_brief_generated = True
    s4_memory.save_daily_log(log)

    return brief_text


def _fallback_morning_brief(today_log, weekly, alerts) -> str:
    """Fallback brief when Model A is unavailable."""
    lines = [
        f"📅 Morning Brief — {today_log.date}",
        "",
    ]
    if alerts:
        lines.append("⚠ EXAM ALERTS:")
        for a in alerts:
            lines.append(f"  {a['message']}")
        lines.append("")
    lines.append(f"This week: {weekly.study_hours_total:.1f}h studied / {weekly.study_target_hours:.0f}h target")
    lines.append("")
    lines.append("MUST-DO TODAY:")
    if weekly.goals:
        for i, goal in enumerate(weekly.goals[:3], 1):
            lines.append(f"  {i}. Work on: {goal}")
    else:
        lines.append("  1. Set your goals for this week with /s4 review")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evening Wrap Generator
# ---------------------------------------------------------------------------

def generate_s4_evening_wrap(
    role_manager,
    s4_memory: S4MemoryManager,
    focus_guard: Optional[FocusGuard] = None,
) -> str:
    """
    Generate an evening wrap using Model A (Chief) for analysis
    and Model E (Rapid) for tomorrow's preview.

    Returns:
        str — formatted evening wrap text
    """
    today_log = s4_memory.get_daily_log()
    weekly = s4_memory.get_weekly_state()
    now = datetime.now()

    # Gather session stats
    context_lines = [
        f"=== EVENING WRAP DATA — {now.strftime('%A, %B %d')} ===",
        "",
        f"Study Time: {today_log.total_study_hours:.1f}h",
        f"Pomodoros: {today_log.pomodoros_completed} completed, {today_log.pomodoros_broken} broken",
        f"Distractions: {len(today_log.distraction_incidents)}",
        f"Tasks Completed: {', '.join(today_log.tasks_completed) if today_log.tasks_completed else 'none'}",
        f"Tasks Pending: {', '.join(t for t in today_log.tasks_planned if t not in today_log.tasks_completed) or 'none'}",
        f"Papers Read: {len(today_log.papers_read)}",
        f"Wins: {', '.join(today_log.wins) if today_log.wins else 'none logged'}",
        f"Blockers: {', '.join(today_log.blockers) if today_log.blockers else 'none'}",
        "",
        f"Weekly Study: {weekly.study_hours_total:.1f}h / {weekly.study_target_hours:.0f}h",
        "",
    ]

    if focus_guard and focus_guard.has_active_session():
        stats = focus_guard.get_session_stats()
        context_lines.append(f"Focus score: {stats.get('focus_score', 'N/A')}%")
        context_lines.append("")

    context_lines.append(
        "Write a brief evening wrap-up. Acknowledge wins, flag blockers, "
        "preview tomorrow's top priority. Be encouraging but direct. Max 150 words."
    )

    prompt = "\n".join(context_lines)

    try:
        result = role_manager.call_role(
            "chief",
            prompt,
            conversation_history=[],
            context_kwargs={"state_snapshot": s4_memory.get_s4_context_snapshot()},
        )
        wrap_text = result.content.strip() if result.success else _fallback_evening_wrap(today_log)
    except Exception as e:
        logger.error("Evening wrap generation failed: %s", e)
        wrap_text = _fallback_evening_wrap(today_log)

    # Mark in daily log
    log = s4_memory.get_daily_log()
    log.evening_wrap_generated = True
    s4_memory.save_daily_log(log)

    return wrap_text


def _fallback_evening_wrap(today_log) -> str:
    """Fallback wrap when Model A is unavailable."""
    lines = [
        f"🌙 Evening Wrap — {today_log.date}",
        "",
        f"📚 Study: {today_log.total_study_hours:.1f}h",
        f"✅ Tasks done: {len(today_log.tasks_completed)}",
        f"🎯 Pomodoros: {today_log.pomodoros_completed}",
        f"⚡ Distractions: {len(today_log.distraction_incidents)}",
    ]
    if today_log.wins:
        lines.append(f"🏆 Win: {today_log.wins[0]}")
    if today_log.blockers:
        lines.append(f"🔴 Blocker: {today_log.blockers[0]}")
    lines.append("")
    lines.append("Good work today. Log wins with `/s4 win <text>` anytime.")
    return "\n".join(lines)
