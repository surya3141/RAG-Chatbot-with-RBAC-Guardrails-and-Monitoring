"""Token usage tracking and cost alerting."""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Pricing per million tokens (USD) — llama-3.1-8b-instant on Groq
_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "default": {"input": 0.05, "output": 0.08},
}


class TokenTracker:
    """Track token usage per user and estimate API costs."""

    def __init__(self, alert_threshold: float = 10.0) -> None:
        self.alert_threshold = alert_threshold
        # username → list of usage records
        self._usage: Dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def track_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        username: str,
    ) -> None:
        """Record a single LLM call's token usage."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": self._calculate_cost(model, prompt_tokens, completion_tokens),
        }
        self._usage[username].append(record)
        logger.info(
            "Token usage tracked for '%s': %d prompt + %d completion = %d total "
            "(est. $%.6f).",
            username,
            prompt_tokens,
            completion_tokens,
            record["total_tokens"],
            record["cost_usd"],
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_session_cost(self, username: str) -> float:
        """Return the cumulative estimated cost (USD) for *username*."""
        return sum(r["cost_usd"] for r in self._usage.get(username, []))

    def check_alert(self, username: str) -> bool:
        """Return True if *username*'s session cost exceeds the alert threshold."""
        cost = self.get_session_cost(username)
        if cost >= self.alert_threshold:
            logger.warning(
                "COST ALERT: User '%s' has exceeded threshold ($%.2f >= $%.2f).",
                username,
                cost,
                self.alert_threshold,
            )
            return True
        return False

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics for all tracked users."""
        stats: Dict[str, Any] = {}
        for username, records in self._usage.items():
            total_tokens = sum(r["total_tokens"] for r in records)
            total_cost = sum(r["cost_usd"] for r in records)
            stats[username] = {
                "num_queries": len(records),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "alert_triggered": self.check_alert(username),
            }
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _MODEL_PRICING.get(model, _MODEL_PRICING["default"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
