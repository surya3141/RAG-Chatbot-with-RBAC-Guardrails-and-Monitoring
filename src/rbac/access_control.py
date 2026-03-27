"""
RBAC (Role-Based Access Control) module.

Loads role definitions from the YAML configuration file and exposes
helpers to:
- look up which document categories a role may access,
- filter a list of retrieved documents to only those the role may see,
- validate whether a user query is within the scope of a role.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from langchain_core.documents import Document
from loguru import logger


class RBACManager:
    """Manages role-based access control for document retrieval."""

    def __init__(self, config_path: str) -> None:
        self._config = self._load_config(config_path)
        self._roles: Dict[str, dict] = self._config.get("roles", {})
        self._out_of_scope: List[str] = self._config.get("out_of_scope_topics", [])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def valid_roles(self) -> List[str]:
        """Return a list of configured role names."""
        return list(self._roles.keys())

    def get_allowed_categories(self, role: str) -> Set[str]:
        """Return the set of document categories accessible by *role*."""
        role = role.lower()
        if role not in self._roles:
            logger.warning(f"Unknown role '{role}'. Returning empty category set.")
            return set()
        return set(self._roles[role].get("allowed_categories", []))

    def get_allowed_keywords(self, role: str) -> List[str]:
        """Return topic keywords that are relevant for *role*."""
        role = role.lower()
        if role not in self._roles:
            return []
        return self._roles[role].get("allowed_keywords", [])

    def filter_documents(
        self, documents: List[Document], role: str
    ) -> List[Document]:
        """
        Filter *documents* to only those accessible by *role*.

        The filter is based on the ``category`` field stored in each
        document's metadata.  C-suite roles with an empty keyword list
        receive all documents.
        """
        allowed = self.get_allowed_categories(role)
        if not allowed:
            return []

        filtered = [
            doc
            for doc in documents
            if doc.metadata.get("category", "").lower() in allowed
        ]
        logger.debug(
            f"RBAC filter [{role}]: {len(documents)} → {len(filtered)} docs"
        )
        return filtered

    def is_query_allowed(self, query: str, role: str) -> bool:
        """
        Return *True* when *query* does **not** touch out-of-scope topics.

        This is a lightweight keyword check; the guardrails layer performs
        a deeper semantic analysis.
        """
        query_lower = query.lower()
        for topic in self._out_of_scope:
            if topic.lower() in query_lower:
                logger.info(
                    f"Query blocked for role '{role}': matches out-of-scope "
                    f"topic '{topic}'."
                )
                return False
        return True

    def get_role_description(self, role: str) -> Optional[str]:
        """Return the human-readable description of *role*."""
        role = role.lower()
        return self._roles.get(role, {}).get("description")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RBAC config not found at '{config_path}'."
            )
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
