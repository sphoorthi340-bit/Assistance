"""
Project Tools
=============

Wraps the ProjectManager methods into Action Engine tools.
"""
from backend.action_engine.tool_registry import registry, ToolDefinition, ToolParameter
from state.project_manager import ProjectManager
from backend.database import DatabaseManager

def _get_project_manager() -> ProjectManager:
    return ProjectManager(db=DatabaseManager())

def create_project(kwargs: dict) -> str:
    pm = _get_project_manager()
    project = pm.add_project(
        name=kwargs.get("name"),
        description=kwargs.get("description"),
        priority=kwargs.get("priority", 3)
    )
    return {
        "message": f"Created project: {project['name']} (ID: {project['id']})",
        "reverse_operation": {
            "intent": "delete_project",
            "parameters": {"project_id": project['id']}
        }
    }

def add_project_task(kwargs: dict) -> str:
    pm = _get_project_manager()
    task = pm.add_task(
        project_id=kwargs.get("project_id"),
        title=kwargs.get("title"),
        description=kwargs.get("description"),
        due_date=kwargs.get("due_date"),
        priority=kwargs.get("priority", 3)
    )
    return {
        "message": f"Added task '{task['title']}' to project {kwargs.get('project_id')}.",
        "reverse_operation": {
            "intent": "delete_task",
            "parameters": {"task_id": task['id']}
        }
    }

def update_task_status(kwargs: dict) -> str:
    pm = _get_project_manager()
    task = pm.update_task(
        task_id=kwargs.get("task_id"),
        status=kwargs.get("status")
    )
    return {"message": f"Updated task {task['id']} status to {task['status']}."}

def delete_project(kwargs: dict) -> str:
    pm = _get_project_manager()
    project_id = kwargs.pop("project_id")
    pm.delete_project(project_id)
    return {"message": f"Deleted project {project_id}."}

def delete_task(kwargs: dict) -> str:
    pm = _get_project_manager()
    task_id = kwargs.pop("task_id")
    pm.delete_task(task_id)
    return {"message": f"Deleted task {task_id}."}

registry.register(ToolDefinition(
    name="create_project",
    description="Create a new project.",
    parameters=[
        ToolParameter("name", "string", "The name of the project"),
        ToolParameter("description", "string", "Optional description", required=False),
        ToolParameter("priority", "number", "Priority 1-5", required=False)
    ],
    handler=create_project,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="add_project_task",
    description="Add a task to an existing project.",
    parameters=[
        ToolParameter("project_id", "string", "The ID of the project"),
        ToolParameter("title", "string", "Task title"),
        ToolParameter("description", "string", "Task description", required=False),
        ToolParameter("due_date", "string", "ISO 8601 due date", required=False),
        ToolParameter("priority", "number", "Priority 1-5", required=False)
    ],
    handler=add_project_task,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="update_task_status",
    description="Update the status of a project task.",
    parameters=[
        ToolParameter("task_id", "string", "The ID of the task"),
        ToolParameter("status", "string", "New status (pending, in_progress, completed, cancelled)", enum=["pending", "in_progress", "completed", "cancelled"])
    ],
    handler=update_task_status,
    risk_level="low"
))

registry.register(ToolDefinition(
    name="delete_project",
    description="Delete a project.",
    parameters=[
        ToolParameter("project_id", "string", "The ID of the project to delete")
    ],
    handler=delete_project,
    risk_level="high"
))

registry.register(ToolDefinition(
    name="delete_task",
    description="Delete a project task.",
    parameters=[
        ToolParameter("task_id", "string", "The ID of the task to delete")
    ],
    handler=delete_task,
    risk_level="high"
))
