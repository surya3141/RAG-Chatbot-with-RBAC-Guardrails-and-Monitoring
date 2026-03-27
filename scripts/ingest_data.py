#!/usr/bin/env python3
"""Ingest documents from the data/ directory into ChromaDB collections."""
import logging
import os
import sys

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from config.settings import get_settings
    from src.ingestion.chunker import DocumentChunker
    from src.ingestion.loader import DocumentLoader
    from src.vectorstore.chroma_store import ChromaVectorStore

    settings = get_settings()

    logger.info("Starting data ingestion from '%s'…", settings.DATA_DIR)

    # 1. Load documents
    loader = DocumentLoader()
    documents = loader.load_documents(settings.DATA_DIR)

    if not documents:
        logger.warning(
            "No documents found in '%s'. "
            "Ensure the data/ directory contains .txt, .pdf, or .docx files.",
            settings.DATA_DIR,
        )
        return

    logger.info("Loaded %d document(s).", len(documents))

    # 2. Chunk documents
    chunker = DocumentChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks_by_collection = chunker.chunk_by_collection(documents)

    logger.info("Chunked into %d collection(s).", len(chunks_by_collection))

    # 3. Ingest into ChromaDB
    vector_store = ChromaVectorStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    total_chunks = 0
    for collection_name, chunks in chunks_by_collection.items():
        vector_store.add_documents(chunks, collection_name)
        total_chunks += len(chunks)
        logger.info(
            "Collection '%s': %d chunk(s) ingested.", collection_name, len(chunks)
        )

    # 4. Summary
    print("\n" + "=" * 55)
    print("  Ingestion Complete")
    print("=" * 55)
    print(f"  Data directory  : {settings.DATA_DIR}")
    print(f"  ChromaDB path   : {settings.CHROMA_PERSIST_DIR}")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Collections     : {list(chunks_by_collection.keys())}")
    print("=" * 55)


if __name__ == "__main__":
    main()
