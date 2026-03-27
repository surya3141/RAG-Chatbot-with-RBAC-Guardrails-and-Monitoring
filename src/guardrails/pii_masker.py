"""PII masking using Microsoft Presidio with regex fallback."""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Regex patterns used as fallback when Presidio is unavailable
_REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PHONE": re.compile(
        r"(\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
}


class PIIMasker:
    """Detect and mask PII in text strings.

    Uses Microsoft Presidio when available; falls back to regex otherwise.
    """

    def __init__(self) -> None:
        self._presidio_available = False
        self._analyzer = None
        self._anonymizer = None
        self._setup_presidio()

    def _setup_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            logger.info("Presidio PII engine loaded successfully.")
        except ImportError:
            logger.warning(
                "Presidio not available — falling back to regex PII masking."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_pii(self, text: str) -> str:
        """Return *text* with all detected PII replaced by type tags."""
        if not text:
            return text
        if self._presidio_available:
            return self._mask_with_presidio(text)
        return self._mask_with_regex(text)

    def contains_pii(self, text: str) -> bool:
        """Return True if *text* contains any recognised PII."""
        if not text:
            return False
        return len(self.get_pii_entities(text)) > 0

    def get_pii_entities(self, text: str) -> List[str]:
        """Return a deduplicated list of PII entity types found in *text*."""
        if not text:
            return []
        if self._presidio_available:
            return self._get_entities_presidio(text)
        return self._get_entities_regex(text)

    # ------------------------------------------------------------------
    # Presidio helpers
    # ------------------------------------------------------------------

    def _mask_with_presidio(self, text: str) -> str:
        results = self._analyzer.analyze(
            text=text,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD", "US_SSN"],
            language="en",
        )
        if not results:
            return text
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

    def _get_entities_presidio(self, text: str) -> List[str]:
        results = self._analyzer.analyze(
            text=text,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD", "US_SSN"],
            language="en",
        )
        return list({r.entity_type for r in results})

    # ------------------------------------------------------------------
    # Regex fallback helpers
    # ------------------------------------------------------------------

    def _mask_with_regex(self, text: str) -> str:
        for label, pattern in _REGEX_PATTERNS.items():
            text = pattern.sub(f"<{label}>", text)
        return text

    def _get_entities_regex(self, text: str) -> List[str]:
        return [label for label, pattern in _REGEX_PATTERNS.items() if pattern.search(text)]
