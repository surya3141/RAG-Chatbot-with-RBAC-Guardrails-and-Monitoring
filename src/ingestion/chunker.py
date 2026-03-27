"""Document chunking utilities."""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available — DocumentChunker will return empty lists.")

    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class DocumentChunker:
    """Split LangChain Documents into smaller, overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if _LANGCHAIN_AVAILABLE:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Return a flat list of chunks for all input documents."""
        if not documents or not _LANGCHAIN_AVAILABLE:
            return []
        chunks = self._splitter.split_documents(documents)
        logger.info(
            "Chunked %d document(s) into %d chunk(s) (size=%d, overlap=%d).",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_by_collection(
        self, documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """Chunk documents and group the results by their ``collection`` metadata key."""
        chunks = self.chunk_documents(documents)
        grouped: Dict[str, List[Document]] = {}
        for chunk in chunks:
            collection = chunk.metadata.get("collection", "general")
            grouped.setdefault(collection, []).append(chunk)
        logger.info(
            "Documents grouped into %d collection(s): %s",
            len(grouped),
            list(grouped.keys()),
        )
        return grouped
