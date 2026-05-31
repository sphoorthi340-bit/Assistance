"""
Habit Tools
===========

Wraps the HabitManager methods into Action Engine tools.
"""
from backend.action_engine.tool_registry import registry, ToolDefinition, ToolParameter
from state.habit_manager import HabitManager
from backend.database import DatabaseManager

def _get_habit_manager() -> HabitManager:
    return HabitManager(db=DatabaseManager())

def create_habit(kwargs: dict) -> str:
    hm = _get_habit_manager()
    habit = hm.add_habit(
        name=kwargs.get("name"),
        description=kwargs.get("description"),
        frequency=kwargs.get("frequency", "daily"),
        category=kwargs.get("category", "personal"),
        target_per_period=kwargs.get("target_per_period", 1.0)
    )
    return {
        "message": f"Created habit: {habit['name']} (ID: {habit['id']})",
        "reverse_operation": {
            "intent": "delete_habit",
            "parameters": {"habit_id": habit['id']}
        }
    }

def log_habit(kwargs: dict) -> str:
    hm = _get_habit_manager()
    log = hm.log_habit(
        habit_id=kwargs.get("habit_id"),
        date=kwargs.get("date_str"),
        completed=kwargs.get("completed", True),
        value=kwargs.get("value"),
        notes=kwargs.get("notes")
    )
    return {
        "message": f"Logged habit {kwargs.get('habit_id')} on {log['date']}.",
        # Reversing a log might require a delete_habit_log tool which we'd need to create.
        # For simplicity in V2.5, we just say it's reversible if we had a delete_log.
    }

def delete_habit(kwargs: dict) -> str:
    hm = _get_habit_manager()
    habit_id = kwargs.pop("habit_id")
    hm.delete_habit(habit_id)
    return {"message": f"Deleted habit {habit_id}."}

registry.register(ToolDefinition(
    name="create_habit",
    description="Create a new habit to track.",
    parameters=[
        ToolParameter("name", "string", "The name of the habit"),
        ToolParameter("description", "string", "Optional detailed description of the habit", required=False),
        ToolParameter("frequency", "string", "Frequency of habit (daily, weekly, custom)", required=False, enum=["daily", "weekly", "custom"]),
        ToolParameter("category", "string", "Category (e.g. personal, work, fitness)", required=False),
        ToolParameter("target_per_period", "number", "Target repetitions per period", required=False)
    ],
    handler=create_habit,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="log_habit",
    description="Log a completion or progress for a habit on a specific date.",
    parameters=[
        ToolParameter("habit_id", "string", "The ID of the habit to log"),
        ToolParameter("date_str", "string", "YYYY-MM-DD date of the log", required=False),
        ToolParameter("completed", "boolean", "Whether it was completed", required=False),
        ToolParameter("value", "number", "Progress value if applicable", required=False),
        ToolParameter("notes", "string", "Any notes about this log", required=False)
    ],
    handler=log_habit,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="delete_habit",
    description="Delete a habit.",
    parameters=[
        ToolParameter("habit_id", "string", "The ID of the habit to delete")
    ],
    handler=delete_habit,
    risk_level="high"
))
