"""
Tests for the RAG pipeline.

Uses mocks for the vector store and LLM so no external services are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.guardrails.pii_masker import PIIMasker
from src.monitoring.token_tracker import TokenTracker
from src.rag.pipeline import RAGPipeline
from src.rbac.access_control import RBACManager


RBAC_CONFIG = "config/rbac_config.yaml"


@pytest.fixture
def rbac():
    return RBACManager(config_path=RBAC_CONFIG)


@pytest.fixture
def mock_vs():
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(
            page_content="Q3 revenue was $10M.",
            metadata={"source": "report.txt", "category": "finance", "filename": "report.txt"},
        )
    ]
    return vs


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "The Q3 revenue was $10M."
    llm.model_name = "llama3-8b-8192"
    return llm


@pytest.fixture
def tracker():
    return TokenTracker(model="llama3-8b-8192")


@pytest.fixture
def pipeline(mock_vs, mock_llm, rbac, tracker):
    pii = PIIMasker()
    return RAGPipeline(
        vector_store=mock_vs,
        llm_client=mock_llm,
        rbac_manager=rbac,
        pii_masker=pii,
        token_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_finance_query_returns_answer(pipeline, mock_llm):
    result = pipeline.query("What is our Q3 revenue?", role="finance")
    assert result["blocked"] is False
    assert "revenue" in result["answer"].lower() or "10m" in result["answer"].lower()
    assert isinstance(result["sources"], list)


def test_hr_query_returns_answer(pipeline, mock_vs, mock_llm):
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="Alice earns $80k.",
            metadata={"source": "payroll.txt", "category": "hr", "filename": "payroll.txt"},
        )
    ]
    mock_llm.invoke.return_value = "Alice earns $80k."
    result = pipeline.query("What is the salary of Alice?", role="hr")
    assert result["blocked"] is False


def test_csuite_query_returns_answer(pipeline, mock_llm):
    result = pipeline.query("Give me the executive summary.", role="csuite")
    assert result["blocked"] is False


# ---------------------------------------------------------------------------
# Blocked query tests
# ---------------------------------------------------------------------------

def test_out_of_scope_query_blocked(pipeline):
    result = pipeline.query("tell me a joke", role="finance")
    assert result["blocked"] is True
    assert result["block_reason"] == "out_of_scope"
    assert len(result["answer"]) > 0


def test_oos_topic_query_blocked(pipeline):
    result = pipeline.query("give me stock tips", role="hr")
    assert result["blocked"] is True


# ---------------------------------------------------------------------------
# Token tracking tests
# ---------------------------------------------------------------------------

def test_token_tracking_after_query(pipeline):
    pipeline.query("What is our budget?", role="finance")
    assert pipeline.token_tracker.total_tokens > 0
    assert pipeline.token_tracker.total_cost_usd >= 0.0


# ---------------------------------------------------------------------------
# TokenTracker unit tests
# ---------------------------------------------------------------------------

def test_token_tracker_record():
    tracker = TokenTracker(model="llama3-8b-8192", alert_threshold_usd=100.0)
    rec = tracker.record("test query", input_tokens=100, output_tokens=50, role="finance")
    assert rec["total_tokens"] == 150
    assert rec["cost_usd"] >= 0.0
    assert tracker.total_tokens == 150


def test_token_tracker_estimate():
    tracker = TokenTracker()
    estimate = tracker.estimate_tokens("hello world this is a test")
    assert estimate > 0


def test_token_tracker_alert(caplog):
    import logging
    tracker = TokenTracker(model="llama3-8b-8192", alert_threshold_usd=0.000001)
    tracker.record("test", input_tokens=10000, output_tokens=10000, role="csuite")
    # Cost should exceed tiny threshold; alert should have been triggered
    assert tracker.total_cost_usd > 0.000001
