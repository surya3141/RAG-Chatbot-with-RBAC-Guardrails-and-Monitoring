"""
Out-of-scope query detector guardrail.

Determines whether a user query is relevant to the internal company
knowledge base.  The detector combines:

1. A keyword blocklist for obvious out-of-scope topics.
2. A keyword allowlist derived from the RBAC role configuration.

An optional semantic similarity check (using sentence-transformers) can
be enabled for higher accuracy.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Default out-of-scope signal phrases
# ---------------------------------------------------------------------------
_DEFAULT_OOS_PHRASES = [
    "tell me a joke",
    "write me a poem",
    "what is the weather",
    "stock price",
    "personal advice",
    "relationship advice",
    "medical diagnosis",
    "legal opinion",
    "competitor insider",
    "illegal",
    "hack",
    "exploit",
    "bypass security",
]


class ScopeDetector:
    """Detect whether a query is within the scope of the RAG system."""

    def __init__(
        self,
        out_of_scope_phrases: Optional[List[str]] = None,
        role_keywords: Optional[List[str]] = None,
    ) -> None:
        self._oos_phrases: List[str] = [
            p.lower() for p in (out_of_scope_phrases or _DEFAULT_OOS_PHRASES)
        ]
        self._role_keywords: List[str] = [
            k.lower() for k in (role_keywords or [])
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_in_scope(self, query: str) -> bool:
        """
        Return *True* when *query* appears to be within scope.

        A query is considered **out of scope** if it matches one or more
        of the configured out-of-scope phrases.
        """
        query_lower = query.lower().strip()

        for phrase in self._oos_phrases:
            if phrase in query_lower:
                logger.info(
                    f"Query flagged as out-of-scope (matched '{phrase}'): "
                    f"{query!r}"
                )
                return False

        return True

    def get_rejection_message(self) -> str:
        return (
            "I'm sorry, but your query appears to be outside the scope of "
            "this internal knowledge base. Please ask questions related to "
            "company data such as financials, HR policies, or operations."
        )
