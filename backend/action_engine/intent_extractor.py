"""
Intent Extractor
================

Uses the LLM to extract structured intents from user messages.
Outputs JSON specifying which tool to call and with what parameters.
"""
import json
from typing import Optional, Dict, Any, List
from backend.logger import get_logger
from backend.llm import OllamaClient
from backend.action_engine.tool_registry import registry
from configs.settings import get_settings

logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are Jarvis's Action Engine Intent Extractor.
Your job is to analyze the user's message and determine if an action should be taken.
You MUST output ONLY a valid JSON object. Do not include markdown formatting or thoughts.

Available Tools:
{tools}

If no tool matches the intent, or the user is just chatting, output:
{{"intent": null}}

If a tool matches the intent, output:
{{
  "intent": "TOOL_NAME",
  "parameters": {{
    "param1": "value1"
  }},
  "confidence": 0.95
}}

The parameters MUST match the schema of the tool. Extract the parameters directly from the user's message. 
If a required parameter is missing, use your best judgment to infer it or omit the intent entirely.
Do NOT explain your reasoning. Output raw JSON only.
"""

class IntentExtractor:
    def __init__(self, settings=None, llm=None):
        self._settings = settings or get_settings()
        self._llm = llm or OllamaClient()
        
    def extract_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Extract the action intent from the message using the LLM.
        Returns a dict with 'intent', 'parameters', and 'confidence', or None.
        """
        if not self._settings.action_engine.enabled:
            logger.debug("Action engine disabled, skipping intent extraction.")
            return None
            
        schemas = registry.get_all_tool_schemas()
        if not schemas:
            logger.debug("No tools registered, skipping intent extraction.")
            return None
            
        tools_str = json.dumps(schemas, indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_str)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        logger.debug(f"Extracting intent for message: '{message}'")
        try:
            # We use format="json" if the model supports it. 
            # In Ollama this forces JSON output.
            response = self._llm.chat(messages, format="json")
            
            content = response.content.strip()
            # Clean up potential markdown formatting if the model still outputs it
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            parsed = json.loads(content.strip())
            
            intent = parsed.get("intent")
            if not intent or intent.lower() == "none":
                logger.debug("LLM determined no actionable intent.")
                return None
                
            confidence = parsed.get("confidence", 1.0)
            parameters = parsed.get("parameters", {})
            
            logger.info(f"Extracted intent: {intent} (conf: {confidence}) with params: {parameters}")
            return {
                "intent": intent,
                "parameters": parameters,
                "confidence": confidence
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            return None
