"""Out-of-scope and blocked-pattern detection for incoming queries."""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

ALLOWED_TOPICS = [
    "finance",
    "financial",
    "budget",
    "revenue",
    "hr",
    "human resources",
    "employee",
    "payroll",
    "salary",
    "company",
    "policy",
    "report",
    "data",
    "information",
    "internal",
    "quarterly",
    "annual",
    "benefit",
    "leave",
    "performance",
    "headcount",
    "cost",
    "expense",
    "profit",
    "loss",
    "department",
    "team",
    "hiring",
    "recruitment",
    "onboarding",
    "training",
    "compliance",
    "regulation",
    "audit",
    "forecast",
    "projection",
    "investment",
    "capital",
    "asset",
]

BLOCKED_PATTERNS = [
    "hack",
    "exploit",
    "password",
    "bypass",
    "jailbreak",
    "ignore previous",
    "system prompt",
    "override instructions",
    "disregard",
    "forget your instructions",
    "act as",
    "pretend you are",
    "sudo",
    "rm -rf",
    "drop table",
    "delete from",
    "malware",
    "ransomware",
    "phishing",
]


class ScopeDetector:
    """Determine whether a user query is in-scope and non-malicious."""

    def is_in_scope(self, query: str) -> bool:
        """Return True if the query relates to an allowed business topic."""
        lower_query = query.lower()
        return any(topic in lower_query for topic in ALLOWED_TOPICS)

    def is_blocked(self, query: str) -> bool:
        """Return True if the query matches a blocked / adversarial pattern."""
        lower_query = query.lower()
        return any(pattern in lower_query for pattern in BLOCKED_PATTERNS)

    def check_query(self, query: str) -> Tuple[bool, str]:
        """Return (is_allowed, reason).

        - blocked queries → (False, "Query contains blocked pattern: …")
        - out-of-scope queries → (False, "Query is out of scope …")
        - everything else → (True, "Query is within scope.")
        """
        if self.is_blocked(query):
            lower_query = query.lower()
            matched = next(
                (p for p in BLOCKED_PATTERNS if p in lower_query), "unknown"
            )
            reason = f"Query contains a blocked pattern: '{matched}'."
            logger.warning("Blocked query detected. Reason: %s", reason)
            return False, reason

        if not self.is_in_scope(query):
            reason = (
                "Query appears to be out of scope for this internal knowledge base. "
                "Please ask about finance, HR, or general company topics."
            )
            logger.info("Out-of-scope query: %s", query[:80])
            return False, reason

        return True, "Query is within scope."
