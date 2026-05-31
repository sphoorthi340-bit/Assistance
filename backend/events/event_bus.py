"""
Jarvis V2.5 — Event Bus
============================
Lightweight publish/subscribe event system for decoupled inter-module
communication.

Architecture decisions:
    - Simple synchronous pub/sub — no async, no threading complexity.
      Events are dispatched in the calling thread, which is fine for
      a terminal-first app with no concurrency requirements.
    - Subscribers receive an Event dataclass with typed payload.
    - Wildcard subscriptions supported (subscribe to "*" for all events).
    - Singleton pattern via get_event_bus() for global access.
    - Event history maintained (bounded deque) for debugging/observability.

    This replaces direct system-to-system calls, enabling:
    - Analytics to listen to habit_logged, goal_completed, etc.
    - Notifications to listen to state changes
    - Memory systems to react to events
    - Scheduler to observe system activity
    WITHOUT any direct coupling between producers and consumers.

Events are fire-and-forget — subscriber failures are logged but
do not propagate to the publisher. This is intentional for resilience.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import threading
from typing import Any, Callable, Optional

from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event data structures
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    Represents a system event.

    Attributes:
        name: Event identifier (e.g., 'habit_logged', 'goal_created').
        data: Arbitrary payload dict specific to the event type.
        timestamp: ISO 8601 timestamp when the event was emitted.
        source: Module or component that emitted the event.
    """
    name: str
    data: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = ""


# Type alias for event handler functions
EventHandler = Callable[[Event], None]


# ---------------------------------------------------------------------------
# Predefined event names (constants for consistency)
# ---------------------------------------------------------------------------

# Goal events
GOAL_CREATED = "goal_created"
GOAL_UPDATED = "goal_updated"
GOAL_COMPLETED = "goal_completed"
GOAL_PAUSED = "goal_paused"
GOAL_RESUMED = "goal_resumed"
GOAL_DELETED = "goal_deleted"

# Habit events
HABIT_CREATED = "habit_created"
HABIT_LOGGED = "habit_logged"
HABIT_DEACTIVATED = "habit_deactivated"
HABIT_ACTIVATED = "habit_activated"
HABIT_DELETED = "habit_deleted"

# Project events
PROJECT_CREATED = "project_created"
PROJECT_UPDATED = "project_updated"
PROJECT_COMPLETED = "project_completed"
PROJECT_DELETED = "project_deleted"

# Task events
TASK_CREATED = "task_created"
TASK_COMPLETED = "task_completed"
TASK_DELETED = "task_deleted"

# Knowledge events
DOCUMENT_INGESTED = "document_ingested"
DOCUMENT_DELETED = "document_deleted"

# Memory events
MEMORY_STORED = "memory_stored"
MEMORY_ACCESSED = "memory_accessed"
MEMORY_DELETED = "memory_deleted"

# Session events
SESSION_STARTED = "session_started"
SESSION_ENDED = "session_ended"
SESSION_SUMMARIZED = "session_summarized"

# Action engine events
ACTION_DETECTED = "action_detected"
ACTION_EXECUTED = "action_executed"
ACTION_FAILED = "action_failed"
ACTION_CONFIRMED = "action_confirmed"
ACTION_CANCELLED = "action_cancelled"


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Lightweight synchronous publish/subscribe event bus.

    Usage:
        bus = get_event_bus()

        # Subscribe
        bus.subscribe("habit_logged", my_handler)
        bus.subscribe("*", my_catch_all_handler)  # wildcard

        # Publish
        bus.emit(Event(name="habit_logged", data={"habit_id": "abc"}, source="habit_manager"))

        # Unsubscribe
        bus.unsubscribe("habit_logged", my_handler)
    """

    def __init__(self, history_size: int = 200, max_recursion_depth: int = 10):
        """
        Initialize the event bus.

        Args:
            history_size: Maximum number of events to retain in history
                         for debugging and observability.
        """
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: deque[Event] = deque(maxlen=history_size)
        self._event_counts: dict[str, int] = defaultdict(int)
        self._current_depth = threading.local()
        self._max_recursion_depth = max_recursion_depth

        logger.info("EventBus initialized (history_size=%d, max_recursion=%d)", history_size, max_recursion_depth)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """
        Register a handler for a specific event or wildcard (*).

        Args:
            event_name: The event to subscribe to, or "*" for all events.
            handler: Callable that receives an Event object.
        """
        self._subscribers[event_name].append(handler)
        logger.debug(
            "Subscribed %s to event '%s'",
            getattr(handler, '__qualname__', str(handler)),
            event_name,
        )

    def unsubscribe(self, event_name: str, handler: EventHandler) -> bool:
        """
        Remove a handler from an event.

        Args:
            event_name: The event to unsubscribe from.
            handler: The handler to remove.

        Returns:
            True if the handler was found and removed.
        """
        handlers = self._subscribers.get(event_name, [])
        try:
            handlers.remove(handler)
            logger.debug(
                "Unsubscribed %s from event '%s'",
                getattr(handler, '__qualname__', str(handler)),
                event_name,
            )
            return True
        except ValueError:
            return False

    def emit(self, event: Event) -> None:
        """
        Publish an event to all registered handlers.

        Handlers are called synchronously in registration order.
        Handler exceptions are caught and logged — they do NOT
        propagate to the emitter. This is intentional for resilience.

        Args:
            event: The Event to publish.
        """
        depth = getattr(self._current_depth, "value", 0)
        if depth >= self._max_recursion_depth:
            logger.error("EventBus recursion limit reached emitting event '%s'", event.name)
            return
            
        self._current_depth.value = depth + 1
        
        try:
            # Record in history and stats
            self._history.append(event)
            self._event_counts[event.name] += 1

            # Collect handlers: specific + wildcard
            handlers = list(self._subscribers.get(event.name, []))
            handlers.extend(self._subscribers.get("*", []))

            if handlers:
                logger.debug(
                    "Emitting event '%s' to %d handler(s) [depth=%d] — source=%s",
                    event.name, len(handlers), depth, event.source,
                )

            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(
                        "Event handler %s failed for event '%s': %s",
                        getattr(handler, '__qualname__', str(handler)),
                        event.name,
                        str(e),
                        exc_info=True,
                    )
        finally:
            self._current_depth.value = depth

    def emit_simple(
        self,
        name: str,
        data: Optional[dict] = None,
        source: str = "",
    ) -> None:
        """
        Convenience method to emit an event without constructing Event manually.

        Args:
            name: Event name.
            data: Optional payload dict.
            source: Source module name.
        """
        self.emit(Event(name=name, data=data or {}, source=source))

    # -------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------

    def get_history(self, limit: int = 50) -> list[Event]:
        """Get recent event history, newest first."""
        return list(reversed(list(self._history)))[:limit]

    def get_event_counts(self) -> dict[str, int]:
        """Get total event emission counts by event name."""
        return dict(self._event_counts)

    def get_subscriber_counts(self) -> dict[str, int]:
        """Get number of subscribers per event name."""
        return {
            name: len(handlers)
            for name, handlers in self._subscribers.items()
            if handlers
        }

    def get_stats(self) -> dict:
        """Get event bus statistics for health checks."""
        return {
            "total_events_emitted": sum(self._event_counts.values()),
            "unique_event_types": len(self._event_counts),
            "total_subscribers": sum(
                len(h) for h in self._subscribers.values()
            ),
            "history_size": len(self._history),
            "event_counts": dict(self._event_counts),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_event_bus() -> EventBus:
    """
    Get the singleton EventBus instance.

    Use this function throughout the codebase to access the event bus.
    """
    return EventBus()
