"""RBAC middleware — authentication and permission checks."""
import logging
from typing import Dict, List, Optional

from src.rbac.policies import DEMO_USERS, ROLE_PERMISSIONS

logger = logging.getLogger(__name__)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Validate credentials against the demo user store.

    Returns the user dict (without the password) on success, or None on failure.
    """
    user = DEMO_USERS.get(username)
    if user and user["password"] == password:
        logger.info("User '%s' authenticated successfully.", username)
        return {
            "username": username,
            "role": user["role"],
            "name": user["name"],
        }
    logger.warning("Authentication failed for username '%s'.", username)
    return None


def get_allowed_collections(role: str) -> List[str]:
    """Return the list of ChromaDB collections the given role may access."""
    collections = ROLE_PERMISSIONS.get(role, [])
    logger.debug("Role '%s' has access to collections: %s", role, collections)
    return collections


def check_permission(role: str, collection: str) -> bool:
    """Return True if *role* is allowed to query *collection*."""
    allowed = collection in ROLE_PERMISSIONS.get(role, [])
    if not allowed:
        logger.warning(
            "Permission denied: role '%s' tried to access collection '%s'.",
            role,
            collection,
        )
    return allowed
