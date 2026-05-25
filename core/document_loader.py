"""
This file is a document ingestion pipeline.

Supports: PDF, DOCX, CSV, TXT
Applies data quality checks before returning chunks.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text from a document."""
    content: str
    source: str # Filename
    page: Optional[int] # Page number (PDF) or None
    doc_type: str # "pdf" | "docx" | "csv" | "txt"
    chunk_id: str # Unique ID: "filename_page_idx"
    metadata: dict # Extra document-level metadata


@dataclass
class TableRecord:
    """A structured table extracted from a document."""
    source: str
    table_index: int
    dataframe_json: str # JSON-serialized DataFrame
    description: str # Auto-generated description


class DocumentLoader:

    # Loads and chunks documents from a directory.
    # Applies quality validation at each step.

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_directory(self, path: str) -> tuple[list[DocumentChunk], list[TableRecord]]:

        directory = Path(path) # Load all supported documents from a directory.
        if not directory.exists():
            raise FileNotFoundError(f"Document directory not found: {path}")

        all_chunks: list[DocumentChunk] = []
        all_tables: list[TableRecord] = []
        supported = {".pdf", ".docx", ".csv", ".txt"}

        files = [f for f in directory.iterdir() if f.suffix.lower() in supported]
        logger.info(f"Found {len(files)} document(s) in {path}")

        for file_path in files:
            try:
                chunks, tables = self._load_file(file_path)
                all_chunks.extend(chunks)
                all_tables.extend(tables)
                logger.info(f"✓ {file_path.name}: {len(chunks)} chunks, {len(tables)} tables")
            except Exception as e:
                logger.error(f"✗ Failed to load {file_path.name}: {e}")

        # validation for quality
        all_chunks = self._validate_chunks(all_chunks)
        logger.info(f"Total after validation: {len(all_chunks)} chunks, {len(all_tables)} tables")
        return all_chunks, all_tables

    def _load_file(self, path: Path) -> tuple[list[DocumentChunk], list[TableRecord]]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix == ".docx":
            return self._load_docx(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        elif suffix == ".txt":
            return self._load_txt(path)
        return [], []

    def _load_pdf(self, path: Path) -> tuple[list[DocumentChunk], list[TableRecord]]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Run: pip install pdfplumber")

        chunks, tables = [], []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    for i, chunk in enumerate(self._split_text(text)):
                        chunks.append(DocumentChunk(
                            content=chunk,
                            source=path.name,
                            page=page_num,
                            doc_type="pdf",
                            chunk_id=f"{path.stem}_p{page_num}_c{i}",
                            metadata={"total_pages": len(pdf.pages)},
                        ))

                # Extract tables
                for t_idx, table in enumerate(page.extract_tables()):
                    if table:
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(TableRecord(
                                source=path.name,
                                table_index=t_idx,
                                dataframe_json=df.to_json(orient="records"),
                                description=f"Table {t_idx + 1} from page {page_num} of {path.name}",
                            ))
                        except Exception as e:
                            logger.warning(f"Could not parse table {t_idx} on page {page_num}: {e}")

        return chunks, tables

    def _load_docx(self, path: Path) -> tuple[list[DocumentChunk], list[TableRecord]]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Run: pip install python-docx")

        doc = Document(str(path))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        chunks = [
            DocumentChunk(
                content=chunk,
                source=path.name,
                page=None,
                doc_type="docx",
                chunk_id=f"{path.stem}_c{i}",
                metadata={},
            )
            for i, chunk in enumerate(self._split_text(full_text))
        ]
        return chunks, []

    def _load_csv(self, path: Path) -> tuple[list[DocumentChunk], list[TableRecord]]:
        df = pd.read_csv(str(path))
        # Schema check
        if df.empty:
            logger.warning(f"CSV {path.name} is empty — skipping")
            return [], []

        # Convert to text chunks (row batches) + one table record
        rows_per_chunk = 20
        chunks = []
        for batch_start in range(0, len(df), rows_per_chunk):
            batch = df.iloc[batch_start:batch_start + rows_per_chunk]
            text = batch.to_string(index=False)
            chunks.append(DocumentChunk(
                content=text,
                source=path.name,
                page=None,
                doc_type="csv",
                chunk_id=f"{path.stem}_rows{batch_start}",
                metadata={"columns": list(df.columns), "total_rows": len(df)},
            ))

        tables = [TableRecord(
            source=path.name,
            table_index=0,
            dataframe_json=df.to_json(orient="records"),
            description=f"CSV dataset from {path.name} — {len(df)} rows, columns: {', '.join(df.columns)}",
        )]
        return chunks, tables

    def _load_txt(self, path: Path) -> tuple[list[DocumentChunk], list[TableRecord]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [
            DocumentChunk(
                content=chunk,
                source=path.name,
                page=None,
                doc_type="txt",
                chunk_id=f"{path.stem}_c{i}",
                metadata={},
            )
            for i, chunk in enumerate(self._split_text(text))
        ]
        return chunks, []

    def _split_text(self, text: str) -> list[str]:
        # Simple sliding-window chunker.
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            chunk = " ".join(chunk_words).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _validate_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        #Data quality: drop empty or too-short chunks.
        MIN_WORDS = 10
        valid = []
        dropped = 0
        for chunk in chunks:
            word_count = len(chunk.content.split())
            if word_count < MIN_WORDS:
                dropped += 1
                continue
            if not chunk.source or not chunk.chunk_id:
                dropped += 1
                continue
            valid.append(chunk)
        if dropped:
            logger.info(f"Quality check: dropped {dropped} under-size chunks")
        return valid
