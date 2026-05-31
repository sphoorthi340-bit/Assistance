"""
Heuristic Router
================

Uses fast keyword matching and regex to determine if a message is likely an actionable command
or just casual chat. If it returns False, the Action Engine skips LLM intent extraction.
"""
import re
from backend.logger import get_logger

logger = get_logger(__name__)

# Action verbs that strongly imply intent
ACTION_VERBS = [
    r"\b(add|create|make|new|start)\b",
    r"\b(update|change|edit|modify)\b",
    r"\b(delete|remove|cancel|stop|drop)\b",
    r"\b(track|log|record)\b",
    r"\b(show|list|get|fetch|view)\b",
    r"\b(mark|complete|finish|done)\b"
]

# Nouns representing entities the system manages
ACTION_NOUNS = [
    r"\b(goal|goals)\b",
    r"\b(habit|habits)\b",
    r"\b(project|projects)\b",
    r"\b(task|tasks)\b",
    r"\b(memory|memories|fact|facts)\b",
    r"\b(reminder|reminders)\b",
    r"\b(timer|timers)\b"
]

def is_probable_action(message: str) -> bool:
    """
    Fast heuristic check to see if a message might be an action command.
    Returns True if it looks like an action, False if it's likely conversational.
    """
    if not message or len(message.strip()) == 0:
        return False
        
    msg_lower = message.lower()
    
    # 1. Direct commands or exact matches
    if msg_lower.startswith(("/", "jarvis", "please")):
        logger.debug("Heuristic Router: Matched prefix rule")
        return True
        
    # 2. Check for Verb + Noun presence
    has_verb = any(re.search(pattern, msg_lower) for pattern in ACTION_VERBS)
    has_noun = any(re.search(pattern, msg_lower) for pattern in ACTION_NOUNS)
    
    if has_verb and has_noun:
        logger.debug("Heuristic Router: Matched verb+noun combination")
        return True
        
    # 3. Questions that ask for state
    if msg_lower.startswith(("what is", "what are", "show me", "how many")):
        if has_noun:
            logger.debug("Heuristic Router: Matched query rule")
            return True

    logger.debug("Heuristic Router: Deemed conversational")
    return False
