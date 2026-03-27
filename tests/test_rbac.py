"""
Tests for the RBAC access control module.
"""

import pytest

from src.rbac.access_control import RBACManager


RBAC_CONFIG = "config/rbac_config.yaml"


@pytest.fixture
def rbac():
    return RBACManager(config_path=RBAC_CONFIG)


# ---------------------------------------------------------------------------
# Role existence
# ---------------------------------------------------------------------------

def test_valid_roles_exist(rbac):
    roles = rbac.valid_roles
    assert "finance" in roles
    assert "hr" in roles
    assert "csuite" in roles


# ---------------------------------------------------------------------------
# Allowed categories per role
# ---------------------------------------------------------------------------

def test_finance_allowed_categories(rbac):
    cats = rbac.get_allowed_categories("finance")
    assert "finance" in cats
    assert "marketing_expenses" in cats
    # HR data must NOT be accessible to finance
    assert "hr" not in cats
    assert "payroll" not in cats


def test_hr_allowed_categories(rbac):
    cats = rbac.get_allowed_categories("hr")
    assert "hr" in cats
    assert "payroll" in cats
    # Finance data must NOT be accessible to HR
    assert "finance" not in cats


def test_csuite_allowed_categories(rbac):
    cats = rbac.get_allowed_categories("csuite")
    # C-suite should have broad access
    assert "finance" in cats
    assert "hr" in cats
    assert "payroll" in cats


def test_unknown_role_returns_empty(rbac):
    cats = rbac.get_allowed_categories("intern")
    assert len(cats) == 0


# ---------------------------------------------------------------------------
# Document filtering
# ---------------------------------------------------------------------------

def _make_doc(category: str):
    from langchain_core.documents import Document
    return Document(page_content="test", metadata={"category": category})


def test_filter_finance_docs(rbac):
    docs = [
        _make_doc("finance"),
        _make_doc("hr"),
        _make_doc("payroll"),
        _make_doc("marketing_expenses"),
    ]
    filtered = rbac.filter_documents(docs, role="finance")
    categories = {d.metadata["category"] for d in filtered}
    assert "finance" in categories
    assert "marketing_expenses" in categories
    assert "hr" not in categories
    assert "payroll" not in categories


def test_filter_hr_docs(rbac):
    docs = [_make_doc("finance"), _make_doc("hr"), _make_doc("payroll")]
    filtered = rbac.filter_documents(docs, role="hr")
    categories = {d.metadata["category"] for d in filtered}
    assert "hr" in categories
    assert "payroll" in categories
    assert "finance" not in categories


def test_filter_csuite_sees_all(rbac):
    docs = [_make_doc("finance"), _make_doc("hr"), _make_doc("payroll")]
    filtered = rbac.filter_documents(docs, role="csuite")
    # C-suite can see all three categories
    assert len(filtered) == 3


def test_filter_unknown_role_returns_empty(rbac):
    docs = [_make_doc("finance"), _make_doc("hr")]
    filtered = rbac.filter_documents(docs, role="intern")
    assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Query permission
# ---------------------------------------------------------------------------

def test_out_of_scope_query_blocked(rbac):
    assert rbac.is_query_allowed("give me stock tips", "finance") is False


def test_in_scope_query_allowed(rbac):
    assert rbac.is_query_allowed("what is our Q3 revenue?", "finance") is True


def test_role_description(rbac):
    desc = rbac.get_role_description("finance")
    assert desc is not None
    assert len(desc) > 0
