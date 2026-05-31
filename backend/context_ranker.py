"""
Jarvis V2.5 — Context Ranker
================================
Ranking and prioritization logic for unified context assembly (REV 3 & 4).

Architecture decisions:
    - Unifies context items from multiple sources (state, memory, knowledge, history).
    - Ensures each item is tagged with provenance metadata (REV 4).
    - Ranks items based on a combined score (relevance + recency + source priority).
    - Enforces a strict token budget (REV 3: reduced budget for local models).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Optional

from backend.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextItem:
    """
    A single piece of context for the LLM prompt.
    Includes comprehensive provenance metadata (REV 4).
    """
    content: str
    source: str                 # 'state', 'memory', 'knowledge', 'history'
    source_id: str              # Original ID of the item
    timestamp: str              # ISO format
    relevance_score: float      # Semantic or heuristic score (0.0 to 1.0)
    confidence: float           # Extraction or match confidence
    retrieval_reason: str       # Human-readable reason for retrieval
    
    # Calculated during ranking
    recency_score: float = 0.0
    combined_score: float = 0.0
    token_estimate: int = field(init=False)
    
    def __post_init__(self):
        # Rough token estimate: 1 token ≈ 4 characters
        self.token_estimate = max(1, len(self.content) // 4)


class ContextRanker:
    """
    Ranks and truncates context items based on relevance, recency, and budget.
    """

    # Source priorities (higher is better)
    # State is highest because it grounds Jarvis in the current reality.
    # History is next to maintain immediate conversational continuity.
    # Memory and Knowledge are semantic and may have noise.
    SOURCE_WEIGHTS = {
        "state": 1.0,
        "history": 0.9,
        "memory": 0.8,
        "knowledge": 0.7,
    }

    def __init__(self):
        logger.debug("ContextRanker initialized")

    def _calculate_recency_score(self, timestamp_iso: str) -> float:
        """
        Calculate an exponential decay score based on age.
        Newer items score closer to 1.0. Older items decay towards 0.1.
        """
        if not timestamp_iso:
            return 0.5
            
        try:
            # Parse ISO time, handling Python 3.11 fromisoformat quirks with Z
            clean_ts = timestamp_iso.replace("Z", "+00:00")
            item_time = datetime.fromisoformat(clean_ts)
            
            # Ensure timezone awareness
            if item_time.tzinfo is None:
                item_time = item_time.replace(tzinfo=timezone.utc)
                
            now = datetime.now(timezone.utc)
            age_days = (now - item_time).total_seconds() / (24 * 3600)
            
            # Half-life of 30 days
            # e^(-λ * t) where λ = ln(2) / half_life
            half_life = 30.0
            lambda_val = math.log(2) / half_life
            
            score = math.exp(-lambda_val * age_days)
            # Ensure it doesn't drop to absolute 0
            return max(0.1, min(1.0, score))
            
        except Exception as e:
            logger.debug("Failed to calculate recency for %s: %s", timestamp_iso, str(e))
            return 0.5

    def _calculate_combined_score(self, item: ContextItem) -> float:
        """
        Calculate the final ranking score.
        Formula: (Relevance * 0.6) + (Recency * 0.2) + (Source Priority * 0.2)
        """
        item.recency_score = self._calculate_recency_score(item.timestamp)
        source_weight = self.SOURCE_WEIGHTS.get(item.source, 0.5)
        
        # State and History are often retrieved heuristically, so their relevance might
        # be artificially 1.0. The source weight acts as the primary differentiator.
        item.combined_score = (
            (item.relevance_score * 0.6) +
            (item.recency_score * 0.2) +
            (source_weight * 0.2)
        )
        return item.combined_score

    def rank_and_truncate(
        self, 
        items: list[ContextItem], 
        budget_tokens: int
    ) -> list[ContextItem]:
        """
        Rank items and greedily pack them into the token budget.
        
        Args:
            items: List of all retrieved ContextItems.
            budget_tokens: Maximum tokens allowed.
            
        Returns:
            List of accepted ContextItems, sorted by importance (highest first).
        """
        if not items:
            return []

        # 1. Score all items
        for item in items:
            self._calculate_combined_score(item)

        # 2. Sort by combined_score descending
        items.sort(key=lambda x: x.combined_score, reverse=True)

        # 3. Pack into budget
        accepted = []
        current_tokens = 0
        
        for item in items:
            if current_tokens + item.token_estimate <= budget_tokens:
                accepted.append(item)
                current_tokens += item.token_estimate
            else:
                # If a single item is too big but we still have budget,
                # we could truncate the item itself here, but for now we just skip it
                # to maintain semantic integrity.
                logger.debug(
                    "Context item %s (score %.2f) skipped — exceeds budget (%d + %d > %d)",
                    item.source_id, item.combined_score, current_tokens, item.token_estimate, budget_tokens
                )
                
        logger.info(
            "Ranked context: accepted %d/%d items, used ~%d/%d tokens",
            len(accepted), len(items), current_tokens, budget_tokens
        )
        
        return accepted
