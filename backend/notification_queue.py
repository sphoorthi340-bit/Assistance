"""
Jarvis V2.5 — Notification Queue
==================================
Thread-safe queue for background → foreground communication.

Architecture decisions:
    - Background jobs (APScheduler) cannot print directly to the terminal
      without disrupting the user's input/output flow.
    - Instead, background tasks place notifications in this queue.
    - The foreground chat loop checks this queue before each interaction
      and displays any pending notifications cleanly.
    - Notifications are ephemeral (REV 6) — they exist only to be shown,
      not automatically persisted to memory.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Optional
import uuid

from backend.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Notification:
    """An ephemeral system notification."""
    title: str
    content: str
    priority: str = "normal"  # 'low', 'normal', 'high'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class NotificationQueue:
    """Thread-safe queue for system notifications."""
    
    def __init__(self):
        self._queue = deque()
        self._lock = threading.Lock()
        self._daily_counts = {"date": None, "medium": 0, "low": 0}
        logger.info("NotificationQueue initialized")
        
    def add(self, title: str, content: str, priority: str = "normal") -> Optional[Notification]:
        """Add a notification to the queue, subject to rate limits."""
        # Check rate limits
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self._daily_counts["date"] != now_date:
                self._daily_counts = {"date": now_date, "medium": 0, "low": 0}
                
            if priority == "low":
                if self._daily_counts["low"] >= 1:
                    logger.debug("Rate-limited low priority notification: %s", title)
                    return None
                self._daily_counts["low"] += 1
            elif priority == "normal":
                if self._daily_counts["medium"] >= 2:
                    logger.debug("Rate-limited normal priority notification: %s", title)
                    return None
                self._daily_counts["medium"] += 1
                
        notif = Notification(title=title, content=content, priority=priority)
        with self._lock:
            self._queue.append(notif)
        
        logger.debug("Added notification: [%s] %s", priority, title)
        return notif
        
    def get_pending(self) -> list[Notification]:
        """Get all pending notifications and clear the queue."""
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
            
        if pending:
            logger.debug("Retrieved %d pending notifications", len(pending))
        return pending
        
    def peek(self) -> list[Notification]:
        """View pending notifications without clearing the queue."""
        with self._lock:
            return list(self._queue)
            
    def clear(self) -> None:
        """Clear all pending notifications."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            
        if count > 0:
            logger.debug("Cleared %d notifications", count)


# Global singleton instance
_notification_queue = NotificationQueue()

def get_notification_queue() -> NotificationQueue:
    """Get the global notification queue instance."""
    return _notification_queue
