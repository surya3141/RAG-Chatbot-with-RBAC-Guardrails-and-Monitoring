"""Document loading utilities."""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_core.documents import Document

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available — DocumentLoader will return empty lists.")

    class Document:  # type: ignore[no-redef]
        """Minimal stub when LangChain is absent."""

        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class DocumentLoader:
    """Load documents from the filesystem into LangChain Document objects."""

    def load_directory(self, directory: str, collection_name: str) -> List[Document]:
        """Load all supported files from *directory* and tag them with *collection_name*.

        Each document receives metadata keys: ``source``, ``collection``, and ``role``.
        """
        if not os.path.isdir(directory):
            logger.warning("Directory does not exist: %s", directory)
            return []

        if not _LANGCHAIN_AVAILABLE:
            return []

        documents: List[Document] = []
        for root, _, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                doc_list = self._load_file(filepath)
                for doc in doc_list:
                    doc.metadata.update(
                        {
                            "source": filepath,
                            "collection": collection_name,
                            "role": collection_name,
                        }
                    )
                documents.extend(doc_list)

        logger.info(
            "Loaded %d document(s) from '%s' into collection '%s'.",
            len(documents),
            directory,
            collection_name,
        )
        return documents

    def load_documents(self, data_dir: str) -> List[Document]:
        """Recursively load all supported documents under *data_dir*.

        Sub-directories are treated as collection names.
        """
        if not os.path.isdir(data_dir):
            logger.warning("Data directory does not exist: %s", data_dir)
            return []

        all_documents: List[Document] = []
        for entry in os.scandir(data_dir):
            if entry.is_dir():
                collection_name = entry.name
                docs = self.load_directory(entry.path, collection_name)
                all_documents.extend(docs)

        logger.info("Total documents loaded: %d", len(all_documents))
        return all_documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, filepath: str) -> List[Document]:
        """Load a single file; skip unsupported formats."""
        if not _LANGCHAIN_AVAILABLE:
            return []

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
                return loader.load()
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(filepath)
                return loader.load()
            if ext in {".docx", ".doc"}:
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(filepath)
                return loader.load()
        except Exception as exc:
            logger.error("Failed to load '%s': %s", filepath, exc)
        return []
