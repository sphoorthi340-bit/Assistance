"""
Tool Registry
=============

Central registry for all tools available to the Action Engine.
Provides schema generation for LLM consumption.
"""
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
import inspect
from backend.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable
    risk_level: str = "low"  # low, medium, high

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        
    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool registration: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} (Risk: {tool.risk_level})")
        
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())
        
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function schemas for all tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append(self.get_tool_schema(tool.name))
        return schemas
        
    def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """Get OpenAI-compatible function schema for a specific tool."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
            
        properties = {}
        required = []
        
        for param in tool.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
                
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

# Global registry instance
registry = ToolRegistry()
