"""
Jarvis V2.5 - Action Engine
===========================
The Action Engine processes user messages, determines actionable intents,
and safely executes background tasks to modify user state.
"""

from backend.action_engine.tool_registry import registry, ToolDefinition, ToolParameter
from backend.action_engine.heuristic_router import is_probable_action
from backend.action_engine.intent_extractor import IntentExtractor
from backend.action_engine.action_executor import ActionExecutor
from backend.action_engine.action_router import ActionRouter
import backend.action_engine.tools  # Register all built-in tools

__all__ = [
    "registry",
    "ToolDefinition",
    "ToolParameter",
    "is_probable_action",
    "IntentExtractor",
    "ActionExecutor",
    "ActionRouter"
]
