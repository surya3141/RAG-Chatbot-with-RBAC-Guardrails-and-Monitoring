"""
Token cost monitoring module.

Tracks the number of tokens consumed per session / query and raises alerts
when cumulative cost exceeds the configured threshold.

Token price constants use approximate Groq Cloud pricing (as of mid-2024).
Adjust ``PRICE_PER_1K_TOKENS`` when actual prices change.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Approximate price per 1 000 tokens (USD) for Groq Llama3-8b
PRICE_PER_1K_TOKENS: Dict[str, float] = {
    "llama3-8b-8192": 0.00005,
    "llama3-70b-8192": 0.00064,
    "mixtral-8x7b-32768": 0.00024,
    "default": 0.00010,
}


class TokenTracker:
    """Tracks token usage and estimated cost across queries."""

    def __init__(
        self,
        model: str = "llama3-8b-8192",
        alert_threshold_usd: float = 1.0,
        log_dir: Optional[str] = None,
    ) -> None:
        self._model = model
        self._alert_threshold = alert_threshold_usd
        self._price_per_1k = PRICE_PER_1K_TOKENS.get(
            model, PRICE_PER_1K_TOKENS["default"]
        )
        self._log_dir = Path(log_dir) if log_dir else None
        self._records: List[dict] = []
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        query: str,
        input_tokens: int,
        output_tokens: int,
        role: str = "unknown",
    ) -> dict:
        """
        Record a query's token usage and return a summary dict.

        Args:
            query:         The user query (first 120 chars stored).
            input_tokens:  Number of prompt tokens.
            output_tokens: Number of completion tokens.
            role:          User role for audit purposes.

        Returns:
            Dict with usage stats for this query.
        """
        total = input_tokens + output_tokens
        cost = (total / 1000) * self._price_per_1k

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "query_preview": query[:120],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cost_usd": round(cost, 6),
        }
        self._records.append(record)
        self._total_tokens += total
        self._total_cost_usd += cost

        logger.info(
            f"Token usage: {total} tokens (${cost:.6f}) | "
            f"Session total: {self._total_tokens} tokens (${self._total_cost_usd:.4f})"
        )

        if self._total_cost_usd >= self._alert_threshold:
            logger.warning(
                f"⚠ COST ALERT: Session cost ${self._total_cost_usd:.4f} "
                f"has reached or exceeded the threshold of "
                f"${self._alert_threshold:.2f}."
            )

        if self._log_dir:
            self._persist(record)

        return record

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate: ~4 characters per token (GPT/Llama heuristic).
        For production use, replace with tiktoken or the model's tokenizer.
        """
        return max(1, len(text) // 4)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost_usd, 6)

    @property
    def session_summary(self) -> dict:
        return {
            "total_queries": len(self._records),
            "total_tokens": self._total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "model": self._model,
            "alert_threshold_usd": self._alert_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, record: dict) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self._log_dir / "token_usage.jsonl"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
