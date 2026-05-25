"""
This is a CLI ingestion tool.

"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_loader import DocumentLoader
from core.vectorstore import get_vector_store
from core.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_metadata_registry(chunks, tables, source_path: str) -> dict:
    """Build a searchable metadata registry from ingested documents."""
    doc_names = list({c.source for c in chunks})
    registry = {
        "indexed_at": datetime.utcnow().isoformat(),
        "source_directory": str(source_path),
        "total_chunks": len(chunks),
        "total_tables": len(tables),
        "documents": [],
    }
    for name in doc_names:
        doc_chunks = [c for c in chunks if c.source == name]
        doc_tables = [t for t in tables if t.source == name]
        doc_type = doc_chunks[0].doc_type if doc_chunks else "unknown"
        pages = [c.page for c in doc_chunks if c.page is not None]
        registry["documents"].append({
            "filename": name,
            "doc_type": doc_type,
            "chunks": len(doc_chunks),
            "tables": len(doc_tables),
            "pages": max(pages) if pages else None,
            "metadata": doc_chunks[0].metadata if doc_chunks else {},
        })
    return registry


def main():
    parser = argparse.ArgumentParser(description="IntelliDoc Agent — Document Ingestion")
    parser.add_argument(
        "--source",
        default=config.ingest.sample_docs_path,
        help="Path to directory containing documents to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing index before ingesting",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        sys.exit(1)

    logger.info(f"Starting ingestion from: {source_path}")
    logger.info(f"LLM Provider: {config.llm.provider}")
    logger.info(f"Vector Store: {config.vector_store.backend}")

    # Load documents
    loader = DocumentLoader(
        chunk_size=config.ingest.chunk_size,
        chunk_overlap=config.ingest.chunk_overlap,
    )
    chunks, tables = loader.load_directory(str(source_path))

    if not chunks:
        logger.error("No valid chunks produced. Check your documents and try again.")
        sys.exit(1)

    # Index into vector store
    store = get_vector_store()
    store.add_chunks(chunks)

    # Save metadata registry
    registry = build_metadata_registry(chunks, tables, source_path)
    registry_path = Path(config.ingest.metadata_registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Save tables for the table agent
    tables_path = Path("data/tables.json")
    tables_data = [
        {
            "source": t.source,
            "table_index": t.table_index,
            "dataframe_json": t.dataframe_json,
            "description": t.description,
        }
        for t in tables
    ]
    with open(tables_path, "w") as f:
        json.dump(tables_data, f, indent=2)

    logger.info("=" * 60)
    logger.info("✅ Ingestion complete!")
    logger.info(f"   Documents: {len(registry['documents'])}")
    logger.info(f"   Chunks indexed: {len(chunks)}")
    logger.info(f"   Tables extracted: {len(tables)}")
    logger.info(f"   Registry saved: {registry_path}")
    logger.info(f"   Tables saved: {tables_path}")
    logger.info("=" * 60)
    logger.info("Run the app: streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
