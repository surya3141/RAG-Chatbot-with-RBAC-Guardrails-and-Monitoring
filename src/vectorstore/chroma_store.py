"""ChromaDB vector store wrapper."""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("ChromaDB / LangChain not available — ChromaVectorStore is a stub.")

    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class ChromaVectorStore:
    """Thin wrapper around ChromaDB with per-collection namespacing."""

    def __init__(self, persist_dir: str, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self._embeddings = None
        self._client = None
        self._collections: dict = {}

        if _CHROMA_AVAILABLE:
            self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(
                "ChromaVectorStore initialised (persist_dir='%s', model='%s').",
                persist_dir,
                embedding_model,
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document], collection_name: str) -> None:
        """Upsert *documents* into the named ChromaDB collection."""
        if not _CHROMA_AVAILABLE or not documents:
            return
        vectorstore = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self._embeddings,
        )
        vectorstore.add_documents(documents)
        self._collections[collection_name] = vectorstore
        logger.info(
            "Added %d chunk(s) to collection '%s'.", len(documents), collection_name
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        collection_names: List[str],
        n_results: int = 5,
    ) -> List[Document]:
        """Perform similarity search across multiple collections.

        Results from all collections are merged and de-duplicated by content.
        """
        if not _CHROMA_AVAILABLE:
            return []

        seen: set = set()
        results: List[Document] = []

        for name in collection_names:
            vectorstore = self._get_or_load_collection(name)
            if vectorstore is None:
                continue
            try:
                docs = vectorstore.similarity_search(query_text, k=n_results)
                for doc in docs:
                    key = doc.page_content[:200]
                    if key not in seen:
                        seen.add(key)
                        results.append(doc)
            except Exception as exc:
                logger.error("Error querying collection '%s': %s", name, exc)

        logger.debug(
            "Retrieved %d unique document(s) for query across %s.",
            len(results),
            collection_names,
        )
        return results

    def get_retriever(self, collection_names: List[str], k: int = 5):
        """Return a LangChain retriever that searches across *collection_names*."""
        if not _CHROMA_AVAILABLE:
            return None

        all_docs: List[Document] = []

        # Build a unified in-memory Chroma instance from all allowed collections
        for name in collection_names:
            store = self._get_or_load_collection(name)
            if store:
                try:
                    # Fetch all stored documents for the collection
                    raw = store.get()
                    if raw and raw.get("documents"):
                        for content, meta in zip(raw["documents"], raw["metadatas"]):
                            all_docs.append(Document(page_content=content, metadata=meta or {}))
                except Exception as exc:
                    logger.warning("Could not fetch docs from '%s': %s", name, exc)

        if not all_docs:
            logger.warning("No documents found for collections: %s", collection_names)
            return None

        merged_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self._embeddings,
        )
        return merged_store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_load_collection(self, collection_name: str) -> Optional["Chroma"]:
        if collection_name in self._collections:
            return self._collections[collection_name]
        try:
            existing = [c.name for c in self._client.list_collections()]
            if collection_name not in existing:
                logger.debug("Collection '%s' does not exist yet.", collection_name)
                return None
            vectorstore = Chroma(
                client=self._client,
                collection_name=collection_name,
                embedding_function=self._embeddings,
            )
            self._collections[collection_name] = vectorstore
            return vectorstore
        except Exception as exc:
            logger.error("Failed to load collection '%s': %s", collection_name, exc)
            return None
