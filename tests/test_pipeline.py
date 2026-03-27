"""Tests for the RAG pipeline, token tracker, and RBAC collection logic."""
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.token_tracker import TokenTracker
from src.rbac.middleware import get_allowed_collections
from src.rbac.policies import Role


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _MockSettings:
    GROQ_API_KEY = None
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR = "./chroma_db_test"
    TOKEN_COST_ALERT_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def _make_pipeline(self, role: str = Role.FINANCE):
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(settings=_MockSettings(), role=role)
        return pipeline

    @patch("src.guardrails.scope_detector.ScopeDetector.check_query")
    def test_pipeline_out_of_scope_query(self, mock_check):
        mock_check.return_value = (False, "Query is out of scope for this knowledge base.")
        pipeline = self._make_pipeline()
        pipeline.setup()

        result = pipeline.query("What is the weather today?", username="finance_user")

        assert result["answer"].startswith("⚠️")
        assert "out of scope" in result["answer"].lower()
        assert result["sources"] == []
        assert result["tokens_used"] == 0

    @patch("src.guardrails.scope_detector.ScopeDetector.check_query")
    def test_pipeline_blocked_query(self, mock_check):
        mock_check.return_value = (False, "Query contains a blocked pattern: 'hack'.")
        pipeline = self._make_pipeline()
        pipeline.setup()

        result = pipeline.query("hack the system", username="finance_user")

        assert result["answer"].startswith("⚠️")
        assert result["sources"] == []

    @patch("src.guardrails.scope_detector.ScopeDetector.check_query")
    @patch("src.vectorstore.chroma_store.ChromaVectorStore.query")
    def test_pipeline_no_documents_found(self, mock_vector_query, mock_check):
        mock_check.return_value = (True, "Within scope.")
        mock_vector_query.return_value = []

        pipeline = self._make_pipeline()
        pipeline.setup()

        result = pipeline.query("What is the revenue?", username="finance_user")

        assert result["role"] == Role.FINANCE
        assert isinstance(result["answer"], str)
        assert result["tokens_used"] == 0

    @patch("src.guardrails.scope_detector.ScopeDetector.check_query")
    @patch("src.vectorstore.chroma_store.ChromaVectorStore.query")
    def test_pipeline_returns_correct_role(self, mock_vector_query, mock_check):
        mock_check.return_value = (True, "Within scope.")
        mock_vector_query.return_value = []

        pipeline = self._make_pipeline(role=Role.CSUITE)
        pipeline.setup()

        result = pipeline.query("Show me all reports.", username="ceo")
        assert result["role"] == Role.CSUITE


# ---------------------------------------------------------------------------
# Token tracker tests
# ---------------------------------------------------------------------------

class TestTokenTracker:
    def test_token_tracker_initial_cost_zero(self):
        tracker = TokenTracker(alert_threshold=10.0)
        assert tracker.get_session_cost("alice") == 0.0

    def test_token_tracker_usage(self):
        tracker = TokenTracker(alert_threshold=10.0)
        tracker.track_usage(
            model="llama-3.1-8b-instant",
            prompt_tokens=1000,
            completion_tokens=500,
            username="alice",
        )
        cost = tracker.get_session_cost("alice")
        assert cost > 0.0

    def test_token_tracker_accumulates_across_calls(self):
        tracker = TokenTracker(alert_threshold=10.0)
        tracker.track_usage("llama-3.1-8b-instant", 1000, 500, "alice")
        cost1 = tracker.get_session_cost("alice")
        tracker.track_usage("llama-3.1-8b-instant", 1000, 500, "alice")
        cost2 = tracker.get_session_cost("alice")
        assert cost2 > cost1

    def test_token_tracker_alert_not_triggered_below_threshold(self):
        tracker = TokenTracker(alert_threshold=100.0)
        tracker.track_usage("llama-3.1-8b-instant", 100, 50, "alice")
        assert tracker.check_alert("alice") is False

    def test_token_tracker_alert(self):
        tracker = TokenTracker(alert_threshold=0.000001)
        tracker.track_usage("llama-3.1-8b-instant", 1000, 500, "alice")
        assert tracker.check_alert("alice") is True

    def test_token_tracker_per_user_isolation(self):
        tracker = TokenTracker(alert_threshold=10.0)
        tracker.track_usage("llama-3.1-8b-instant", 1000, 500, "alice")
        assert tracker.get_session_cost("bob") == 0.0

    def test_token_tracker_get_usage_stats(self):
        tracker = TokenTracker(alert_threshold=10.0)
        tracker.track_usage("llama-3.1-8b-instant", 200, 100, "alice")
        stats = tracker.get_usage_stats()
        assert "alice" in stats
        assert stats["alice"]["num_queries"] == 1
        assert stats["alice"]["total_tokens"] == 300

    def test_token_tracker_cost_calculation(self):
        tracker = TokenTracker(alert_threshold=10.0)
        # 1M input tokens at $0.05 → $0.05; 1M output at $0.08 → $0.08
        tracker.track_usage("llama-3.1-8b-instant", 1_000_000, 1_000_000, "test")
        cost = tracker.get_session_cost("test")
        assert abs(cost - 0.13) < 0.001


# ---------------------------------------------------------------------------
# RBAC collection tests
# ---------------------------------------------------------------------------

class TestRBACCollections:
    def test_finance_gets_correct_collections(self):
        collections = get_allowed_collections(Role.FINANCE)
        assert set(collections) == {"finance", "general"}

    def test_hr_gets_correct_collections(self):
        collections = get_allowed_collections(Role.HR)
        assert set(collections) == {"hr", "general"}

    def test_csuite_gets_all_collections(self):
        collections = get_allowed_collections(Role.CSUITE)
        assert set(collections) == {"finance", "hr", "general"}

    def test_unknown_role_gets_empty(self):
        collections = get_allowed_collections("unknown")
        assert collections == []

    def test_finance_cannot_access_hr(self):
        from src.rbac.middleware import check_permission

        assert check_permission(Role.FINANCE, "hr") is False

    def test_hr_cannot_access_finance(self):
        from src.rbac.middleware import check_permission

        assert check_permission(Role.HR, "finance") is False
