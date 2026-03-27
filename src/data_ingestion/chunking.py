"""
Document chunking module.

Splits loaded documents into smaller chunks suitable for embedding and
retrieval.  All original metadata (including the RBAC *category* tag) is
preserved on every chunk.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents:     List of documents produced by the ingestion module.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive
                       chunks (helps preserve context at boundaries).

    Returns:
        List of chunked ``Document`` objects with preserved metadata.
    """
    if not documents:
        logger.warning("No documents to chunk.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Created {len(chunks)} chunks from {len(documents)} document(s) "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks
