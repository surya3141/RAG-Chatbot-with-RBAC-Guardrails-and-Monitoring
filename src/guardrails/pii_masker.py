"""
PII Masking guardrail.

Uses Microsoft Presidio to detect and anonymise personally-identifiable
information (PII) in both user queries and LLM responses.

When Presidio or its spaCy model is not installed the module falls back
to a simple regex-based masker so the application still runs in
lightweight environments.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Optional Presidio imports
# ---------------------------------------------------------------------------
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning(
        "presidio-analyzer / presidio-anonymizer not installed. "
        "Falling back to regex-based PII masking."
    )

# ---------------------------------------------------------------------------
# Regex fallback patterns
# ---------------------------------------------------------------------------
_REGEX_PATTERNS = [
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    # US phone numbers (various formats)
    (
        re.compile(
            r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
        ),
        "<PHONE_NUMBER>",
    ),
    # US Social Security Numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    # Credit card numbers (basic)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "<CREDIT_CARD>"),
    # Date of birth hints
    (
        re.compile(
            r"\b(DOB|date of birth)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "<DATE_OF_BIRTH>",
    ),
]


class PIIMasker:
    """Detect and mask PII in text using Presidio (or regex fallback)."""

    def __init__(self, language: str = "en") -> None:
        self._language = language
        self._presidio_ready = False

        if _PRESIDIO_AVAILABLE:
            try:
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
                self._presidio_ready = True
                logger.info("PIIMasker initialised with Presidio.")
            except Exception as exc:
                logger.warning(
                    f"Presidio initialisation failed ({exc}). "
                    "Using regex fallback."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, text: str) -> str:
        """Return *text* with PII replaced by placeholder tokens."""
        if not text:
            return text
        if self._presidio_ready:
            return self._mask_presidio(text)
        return self._mask_regex(text)

    def contains_pii(self, text: str) -> bool:
        """Return *True* if *text* appears to contain PII."""
        return self.mask(text) != text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_presidio(self, text: str) -> str:
        try:
            results = self._analyzer.analyze(text=text, language=self._language)
            if not results:
                return text
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text
        except Exception as exc:
            logger.warning(f"Presidio masking failed ({exc}). Using regex.")
            return self._mask_regex(text)

    @staticmethod
    def _mask_regex(text: str) -> str:
        for pattern, replacement in _REGEX_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
