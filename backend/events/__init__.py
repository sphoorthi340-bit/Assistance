"""
Jarvis V2.5 — Events Package
================================
Lightweight internal event bus for decoupled system communication.
"""

from backend.events.event_bus import EventBus, get_event_bus

__all__ = ["EventBus", "get_event_bus"]
