"""Tests for RBAC authentication and permission logic."""
import pytest

from src.rbac.middleware import authenticate_user, check_permission, get_allowed_collections
from src.rbac.policies import Role


class TestAuthentication:
    def test_authenticate_valid_user(self):
        user = authenticate_user("finance_user", "finance123")
        assert user is not None
        assert user["username"] == "finance_user"
        assert user["role"] == Role.FINANCE
        assert user["name"] == "Alice Finance"

    def test_authenticate_invalid_user(self):
        user = authenticate_user("nonexistent_user", "somepassword")
        assert user is None

    def test_authenticate_wrong_password(self):
        user = authenticate_user("finance_user", "wrongpassword")
        assert user is None

    def test_authenticate_hr_user(self):
        user = authenticate_user("hr_user", "hr123")
        assert user is not None
        assert user["role"] == Role.HR

    def test_authenticate_ceo(self):
        user = authenticate_user("ceo", "ceo123")
        assert user is not None
        assert user["role"] == Role.CSUITE

    def test_authenticate_empty_credentials(self):
        assert authenticate_user("", "") is None

    def test_authenticate_case_sensitive_username(self):
        # Usernames are case-sensitive
        assert authenticate_user("Finance_User", "finance123") is None


class TestPermissions:
    def test_finance_permissions(self):
        collections = get_allowed_collections(Role.FINANCE)
        assert "finance" in collections
        assert "general" in collections
        assert "hr" not in collections

    def test_hr_permissions(self):
        collections = get_allowed_collections(Role.HR)
        assert "hr" in collections
        assert "general" in collections
        assert "finance" not in collections

    def test_csuite_permissions(self):
        collections = get_allowed_collections(Role.CSUITE)
        assert "finance" in collections
        assert "hr" in collections
        assert "general" in collections

    def test_unknown_role_returns_empty(self):
        collections = get_allowed_collections("unknown_role")
        assert collections == []

    def test_check_permission_allowed(self):
        # Finance role can access finance collection
        assert check_permission(Role.FINANCE, "finance") is True

    def test_check_permission_denied(self):
        # Finance role cannot access hr collection
        assert check_permission(Role.FINANCE, "hr") is False

    def test_check_permission_general_allowed_for_all(self):
        assert check_permission(Role.FINANCE, "general") is True
        assert check_permission(Role.HR, "general") is True
        assert check_permission(Role.CSUITE, "general") is True

    def test_check_permission_hr_cannot_access_finance(self):
        assert check_permission(Role.HR, "finance") is False

    def test_check_permission_csuite_can_access_all(self):
        assert check_permission(Role.CSUITE, "finance") is True
        assert check_permission(Role.CSUITE, "hr") is True
        assert check_permission(Role.CSUITE, "general") is True

    def test_check_permission_unknown_role(self):
        assert check_permission("ghost", "finance") is False
