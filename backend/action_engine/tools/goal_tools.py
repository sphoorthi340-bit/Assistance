"""
Goal Tools
==========

Wraps the GoalManager methods into Action Engine tools.
"""
from backend.action_engine.tool_registry import registry, ToolDefinition, ToolParameter
from state.goal_manager import GoalManager
from backend.database import DatabaseManager

def _get_goal_manager() -> GoalManager:
    return GoalManager(db=DatabaseManager())

def create_goal(kwargs: dict) -> str:
    gm = _get_goal_manager()
    goal = gm.add_goal(
        title=kwargs.get("title"),
        description=kwargs.get("description"),
        category=kwargs.get("category", "personal"),
        target_type=kwargs.get("target_type", "completion"),
        target_value=kwargs.get("target_value"),
        deadline=kwargs.get("deadline"),
        priority=kwargs.get("priority", 3)
    )
    return {
        "message": f"Created goal: {goal['title']} (ID: {goal['id']})",
        "reverse_operation": {
            "intent": "delete_goal",
            "parameters": {"goal_id": goal['id']}
        }
    }

def update_goal(kwargs: dict) -> str:
    gm = _get_goal_manager()
    goal_id = kwargs.pop("goal_id")
    gm.update_goal(goal_id, **kwargs)
    # For update, reverse operation would require fetching previous state.
    # To keep V2.5 simple, we might just return the message or fetch old state first.
    # We will just return message for now unless we fetched previous state.
    return {"message": f"Updated goal {goal_id}."}

def delete_goal(kwargs: dict) -> str:
    gm = _get_goal_manager()
    goal_id = kwargs.pop("goal_id")
    gm.delete_goal(goal_id)
    return {"message": f"Deleted goal {goal_id}."}

registry.register(ToolDefinition(
    name="create_goal",
    description="Create a new goal to track.",
    parameters=[
        ToolParameter("title", "string", "The title of the goal"),
        ToolParameter("description", "string", "Optional detailed description of the goal", required=False),
        ToolParameter("category", "string", "Category (e.g. personal, work, fitness)", required=False),
        ToolParameter("target_type", "string", "Type of target (streak, count, completion, progress)", required=False, enum=["streak", "count", "completion", "progress"]),
        ToolParameter("target_value", "number", "Target numerical value if applicable", required=False),
        ToolParameter("deadline", "string", "ISO 8601 deadline date if any", required=False),
        ToolParameter("priority", "number", "Priority 1-5 (5 is highest)", required=False)
    ],
    handler=create_goal,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="update_goal",
    description="Update an existing goal's properties or progress.",
    parameters=[
        ToolParameter("goal_id", "string", "The ID of the goal to update"),
        ToolParameter("title", "string", "New title", required=False),
        ToolParameter("description", "string", "New description", required=False),
        ToolParameter("status", "string", "New status", required=False, enum=["active", "paused", "completed", "abandoned"]),
        ToolParameter("current_value", "number", "Update progress value", required=False),
        ToolParameter("priority", "number", "Update priority 1-5", required=False)
    ],
    handler=update_goal,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="delete_goal",
    description="Delete a goal.",
    parameters=[
        ToolParameter("goal_id", "string", "The ID of the goal to delete")
    ],
    handler=delete_goal,
    risk_level="high"
))
