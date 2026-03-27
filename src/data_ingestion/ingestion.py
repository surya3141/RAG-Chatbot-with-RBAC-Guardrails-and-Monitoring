"""
Document ingestion module.

Loads text documents from a directory tree where the sub-directory name
is used as the RBAC category tag stored in document metadata.
Supported file types: .txt, .md  (extensible via loader map).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger


# Mapping of file extensions to simple loaders (no heavy deps required)
def _load_text_file(path: Path, category: str) -> List[Document]:
    """Load a plain text or markdown file as a single Document."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "category": category,
                    "filename": path.name,
                },
            )
        ]
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")
        return []


_EXTENSION_LOADERS = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
}


def load_documents(data_dir: str) -> List[Document]:
    """
    Recursively load all supported documents from *data_dir*.

    Each immediate sub-directory of *data_dir* becomes a **category** tag
    that is attached to every document found inside it.  This category is
    later used by the RBAC layer to filter which documents a user may see.

    Args:
        data_dir: Root directory containing category sub-directories.

    Returns:
        List of LangChain ``Document`` objects with populated metadata.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir!r} does not exist.")
        return []

    documents: List[Document] = []

    for entry in sorted(data_path.iterdir()):
        if not entry.is_dir():
            continue
        category = entry.name.lower()
        logger.info(f"Ingesting category '{category}' from {entry}")
        for root, _dirs, files in os.walk(entry):
            for filename in sorted(files):
                file_path = Path(root) / filename
                ext = file_path.suffix.lower()
                loader_fn = _EXTENSION_LOADERS.get(ext)
                if loader_fn is None:
                    logger.debug(f"Skipping unsupported file type: {file_path}")
                    continue
                docs = loader_fn(file_path, category)
                documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} doc(s) from {file_path}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
