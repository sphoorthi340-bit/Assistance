"""
Action Executor
===============

Executes parsed intents safely. Logs to database and emits events.
"""
import time
from typing import Dict, Any, Optional
from backend.logger import get_logger
from backend.database import DatabaseManager
from backend.events import get_event_bus
from backend.action_engine.tool_registry import registry

logger = get_logger(__name__)

class ActionExecutor:
    def __init__(self, db=None, event_bus=None):
        self._db = db or DatabaseManager()
        self._event_bus = event_bus or get_event_bus()

    def execute(
        self, 
        intent: str, 
        parameters: Dict[str, Any], 
        confidence: Optional[float] = None, 
        user_message: Optional[str] = None,
        action_source: str = "user",
        trigger_type: str = "explicit_request",
        trigger_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an action by its intent name, log the result, and emit events.
        """
        tool = registry.get_tool(intent)
        if not tool:
            error_msg = f"Tool '{intent}' not found in registry."
            logger.error(error_msg)
            self._log_and_emit(
                tool_name=intent,
                parameters=parameters,
                status="error",
                message=error_msg,
                confidence=confidence,
                user_message=user_message,
                duration_ms=0,
                action_source=action_source,
                trigger_type=trigger_type,
                trigger_context=trigger_context
            )
            return {"status": "error", "message": error_msg}

        start_time = time.perf_counter()
        logger.info(f"Executing action '{intent}' with parameters: {parameters}")
        
        try:
            # Handlers can return a string message or a dict with 'message' and 'reverse_operation'
            result = tool.handler(parameters)
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            if isinstance(result, dict):
                result_msg = result.get("message", "Action executed successfully.")
                reverse_op = result.get("reverse_operation", None)
            else:
                result_msg = str(result)
                reverse_op = None

            logger.info(f"Action '{intent}' completed successfully: {result_msg}")
            
            undo_intent = reverse_op.get("intent") if reverse_op else None
            undo_parameters = reverse_op.get("parameters") if reverse_op else None
            reversible = bool(reverse_op)

            self._log_and_emit(
                tool_name=intent,
                parameters=parameters,
                status="success",
                message=result_msg,
                confidence=confidence,
                user_message=user_message,
                duration_ms=duration_ms,
                undo_intent=undo_intent,
                undo_parameters=undo_parameters,
                reversible=reversible,
                action_source=action_source,
                trigger_type=trigger_type,
                trigger_context=trigger_context
            )
            return {"status": "success", "message": result_msg}
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Execution failed: {str(e)}"
            logger.exception(f"Error executing action '{intent}': {e}")
            
            self._log_and_emit(
                tool_name=intent,
                parameters=parameters,
                status="error",
                message=error_msg,
                confidence=confidence,
                user_message=user_message,
                duration_ms=duration_ms,
                action_source=action_source,
                trigger_type=trigger_type,
                trigger_context=trigger_context
            )
            return {"status": "error", "message": error_msg}

    def _log_and_emit(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        message: str,
        confidence: Optional[float],
        user_message: Optional[str],
        duration_ms: int,
        undo_intent: Optional[str] = None,
        undo_parameters: Optional[dict] = None,
        reversible: bool = False,
        action_source: str = "user",
        trigger_type: str = "explicit_request",
        trigger_context: Optional[dict] = None
    ) -> None:
        """Helper to log to DB and emit an event."""
        # 1. Log to DB
        try:
            self._db.log_action(
                tool_name=tool_name,
                parameters=parameters,
                result_status=status,
                result_message=message,
                confidence=confidence,
                user_message=user_message,
                execution_time_ms=duration_ms,
                undo_intent=undo_intent,
                undo_parameters=undo_parameters,
                reversible=reversible,
                action_source=action_source,
                trigger_type=trigger_type,
                trigger_context=trigger_context
            )
        except Exception as e:
            logger.error(f"Failed to log action to database: {e}")

        # 2. Emit event
        event_name = "action_executed" if status == "success" else "action_failed"
        self._event_bus.emit_simple(
            event_name=event_name,
            payload={
                "tool_name": tool_name,
                "parameters": parameters,
                "status": status,
                "message": message,
                "duration_ms": duration_ms
            }
        )
