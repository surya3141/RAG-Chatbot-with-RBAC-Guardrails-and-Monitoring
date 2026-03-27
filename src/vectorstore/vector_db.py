"""
Vector store module (ChromaDB).

Provides a thin wrapper around ChromaDB via LangChain that:
- persists embeddings to disk,
- stores the RBAC *category* tag in metadata,
- supports metadata-filtered similarity search.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class VectorStore:
    """ChromaDB-backed vector store with RBAC metadata filtering."""

    _COLLECTION_NAME = "rag_chatbot"

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._persist_dir = persist_dir
        logger.info(f"Loading embedding model '{embedding_model}' …")
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store = Chroma(
            collection_name=self._COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )
        logger.info(
            f"VectorStore ready. Persist dir: '{persist_dir}'. "
            f"Collection: '{self._COLLECTION_NAME}'."
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        """Add (or update) *documents* in the vector store."""
        if not documents:
            logger.warning("No documents to add.")
            return
        self._store.add_documents(documents)
        logger.info(f"Added {len(documents)} document chunk(s) to vector store.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        allowed_categories: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve the top-*k* most relevant document chunks for *query*.

        When *allowed_categories* is provided the search is restricted to
        chunks whose ``category`` metadata field is in that list — this is
        the primary RBAC enforcement point at retrieval time.

        Args:
            query:              User query string.
            allowed_categories: List of category strings the requesting
                                role may access.  ``None`` means no filter
                                (use for C-suite / admin).
            k:                  Number of results to return.

        Returns:
            List of matching ``Document`` chunks.
        """
        where_filter = None
        if allowed_categories is not None and len(allowed_categories) > 0:
            if len(allowed_categories) == 1:
                where_filter = {"category": {"$eq": allowed_categories[0]}}
            else:
                where_filter = {
                    "$or": [
                        {"category": {"$eq": cat}}
                        for cat in allowed_categories
                    ]
                }

        try:
            results = self._store.similarity_search(
                query,
                k=k,
                filter=where_filter,
            )
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

        logger.debug(
            f"Retrieved {len(results)} chunk(s) for query "
            f"(filter={allowed_categories})."
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_count(self) -> int:
        """Return the total number of chunks in the collection."""
        try:
            return self._store._collection.count()
        except Exception:
            return 0
