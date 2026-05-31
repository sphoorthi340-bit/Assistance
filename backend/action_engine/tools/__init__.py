"""
Tools module initialization.
Imports all tool modules so they register themselves with the central registry.
"""
from backend.action_engine.tools.goal_tools import *
from backend.action_engine.tools.habit_tools import *
from backend.action_engine.tools.project_tools import *
from backend.action_engine.tools.memory_tools import *

# Make sure all tools are loaded when this module is imported.
