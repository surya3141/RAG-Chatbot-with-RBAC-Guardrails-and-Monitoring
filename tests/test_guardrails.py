"""
Tests for guardrail modules (PII masking and scope detection).
"""

import pytest

from src.guardrails.pii_masker import PIIMasker
from src.guardrails.scope_detector import ScopeDetector


# ---------------------------------------------------------------------------
# PII Masker
# ---------------------------------------------------------------------------

class TestPIIMasker:
    @pytest.fixture
    def masker(self):
        return PIIMasker()

    def test_email_masked(self, masker):
        text = "Contact john.doe@example.com for details."
        result = masker.mask(text)
        assert "john.doe@example.com" not in result

    def test_phone_masked(self, masker):
        text = "Call us at 555-867-5309."
        result = masker.mask(text)
        assert "555-867-5309" not in result

    def test_ssn_masked(self, masker):
        text = "Employee SSN: 123-45-6789."
        result = masker.mask(text)
        assert "123-45-6789" not in result

    def test_no_pii_unchanged(self, masker):
        text = "The quarterly revenue was $5 million."
        result = masker.mask(text)
        # No PII → text should be returned (possibly unchanged or same content)
        assert "quarterly revenue" in result

    def test_empty_string(self, masker):
        assert masker.mask("") == ""

    def test_contains_pii_true(self, masker):
        assert masker.contains_pii("Email me at test@example.com") is True

    def test_contains_pii_false(self, masker):
        text = "The budget for Q2 is approved."
        assert masker.contains_pii(text) is False


# ---------------------------------------------------------------------------
# Scope Detector
# ---------------------------------------------------------------------------

class TestScopeDetector:
    @pytest.fixture
    def detector(self):
        return ScopeDetector()

    def test_in_scope_business_query(self, detector):
        assert detector.is_in_scope("What is our annual revenue?") is True

    def test_out_of_scope_joke(self, detector):
        assert detector.is_in_scope("tell me a joke") is False

    def test_out_of_scope_weather(self, detector):
        assert detector.is_in_scope("what is the weather today?") is False

    def test_out_of_scope_illegal(self, detector):
        assert detector.is_in_scope("how to hack into the system") is False

    def test_rejection_message_is_string(self, detector):
        msg = detector.get_rejection_message()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_custom_oos_phrases(self):
        detector = ScopeDetector(out_of_scope_phrases=["forbidden topic"])
        assert detector.is_in_scope("tell me about forbidden topic") is False
        assert detector.is_in_scope("normal business question") is True
