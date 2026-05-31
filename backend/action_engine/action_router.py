"""
Action Router
=============

Routes intents to execution or confirmation based on risk level.
"""
from typing import Dict, Any, Tuple
from backend.logger import get_logger
from backend.action_engine.tool_registry import registry
from backend.action_engine.action_executor import ActionExecutor
from configs.settings import get_settings

logger = get_logger(__name__)

class ActionRouter:
    def __init__(self, executor=None, settings=None):
        self._executor = executor or ActionExecutor()
        self._settings = settings or get_settings()

    def route(self, intent_data: Dict[str, Any], user_message: str) -> Tuple[str, Any]:
        """
        Route the extracted intent.
        Returns a tuple: (action_status, message_or_payload)
        action_status can be:
        - "executed": Action was executed automatically
        - "pending_preview": Action requires review (medium risk)
        - "pending_confirmation": Action requires explicit user confirmation (high risk)
        - "pending_clarification": Action needs clarification (low confidence)
        - "blocked_by_safe_mode": Action was blocked because safe mode is on
        - "error": Something went wrong
        """
        intent = intent_data.get("intent")
        parameters = intent_data.get("parameters", {})
        confidence = intent_data.get("confidence", 1.0)
        
        tool = registry.get_tool(intent)
        if not tool:
            return "error", f"Tool '{intent}' not found."

        risk = tool.risk_level
        action_engine_settings = self._settings.action_engine
        
        # Determine how to handle it based on settings
        if risk == "high" and self._settings.system.safe_mode:
            logger.info(f"Action '{intent}' blocked by safe mode (high risk).")
            return "blocked_by_safe_mode", {
                "intent": intent,
                "parameters": parameters,
                "risk": risk,
                "user_message": user_message
            }

        if risk in action_engine_settings.confirm_risk_levels:
            logger.info(f"Action '{intent}' is high risk. Requesting confirmation.")
            return "pending_confirmation", {
                "intent": intent,
                "parameters": parameters,
                "confidence": confidence,
                "risk": risk,
                "user_message": user_message
            }
            
        if risk in action_engine_settings.preview_risk_levels:
            logger.info(f"Action '{intent}' is medium risk. Providing preview.")
            return "pending_preview", {
                "intent": intent,
                "parameters": parameters,
                "confidence": confidence,
                "risk": risk,
                "user_message": user_message
            }

        if confidence < action_engine_settings.low_confidence_threshold:
            logger.info(f"Action '{intent}' confidence ({confidence}) below low_confidence_threshold. Requesting clarification.")
            return "pending_clarification", {
                "intent": intent,
                "parameters": parameters,
                "confidence": confidence,
                "risk": risk,
                "user_message": user_message
            }
            
        if confidence < action_engine_settings.auto_execute_threshold:
            logger.info(f"Action '{intent}' confidence ({confidence}) below auto_execute_threshold. Previewing.")
            return "pending_preview", {
                "intent": intent,
                "parameters": parameters,
                "confidence": confidence,
                "risk": risk,
                "user_message": user_message
            }
            
        # Low risk and high confidence -> Auto execute
        logger.info(f"Auto-executing action '{intent}' (Risk: {risk}, Conf: {confidence})")
        result = self._executor.execute(
            intent=intent,
            parameters=parameters,
            confidence=confidence,
            user_message=user_message
        )
        return "executed", result
