"""RBAC policies — roles, permissions, and demo users."""
from enum import Enum
from typing import Dict, List


class Role(str, Enum):
    FINANCE = "finance"
    HR = "hr"
    CSUITE = "csuite"


# Collections / ChromaDB namespaces each role can query
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    Role.FINANCE: ["finance", "general"],
    Role.HR: ["hr", "general"],
    Role.CSUITE: ["finance", "hr", "general"],
}

# Predefined demo users (in production replace with a real auth system)
DEMO_USERS: Dict[str, Dict] = {
    "finance_user": {
        "password": "finance123",
        "role": Role.FINANCE,
        "name": "Alice Finance",
    },
    "hr_user": {
        "password": "hr123",
        "role": Role.HR,
        "name": "Bob HR",
    },
    "ceo": {
        "password": "ceo123",
        "role": Role.CSUITE,
        "name": "Charlie CEO",
    },
}
