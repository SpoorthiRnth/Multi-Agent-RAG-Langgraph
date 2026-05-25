"""
This is a vectore store abstraction.

Supports FAISS (default, local) and ChromaDB (optional, via Docker).
Switch via VECTOR_STORE env var: "faiss" | "chroma"
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Protocol

from core.config import config
from core.document_loader import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    def add_chunks(self, chunks: list[DocumentChunk]) -> None: ...
    def search(self, query: str, k: int = 5) -> list[dict]: ...
    def is_empty(self) -> bool: ...


class FAISSVectorStore:
    # FAISS-based local vector store.
    # Embeddings: sentence-transformers (HuggingFace, runs fully offline).

    def __init__(self):
        self._index = None
        self._chunks: list[DocumentChunk] = []
        self._embedder = None
        self._index_path = Path(config.vector_store.faiss_index_path)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {config.vector_store.embedding_model}")
            self._embedder = SentenceTransformer(config.vector_store.embedding_model)
        return self._embedder

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        import faiss
        import numpy as np

        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return

        embedder = self._get_embedder()
        texts = [c.content for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype("float32")

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(f"FAISS index now contains {self._index.ntotal} vectors")
        self._save()

    def search(self, query: str, k: int = 5) -> list[dict]:
        import faiss
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty — run ingestion first")
            return []

        embedder = self._get_embedder()
        query_vec = embedder.encode([query]).astype("float32")
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append({
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
                "doc_type": chunk.doc_type,
                "chunk_id": chunk.chunk_id,
                "score": float(dist),
            })
        return results

    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def _save(self):
        import faiss
        self._index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path / "index.faiss"))
        with open(self._index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"FAISS index saved to {self._index_path}")

    def load(self) -> bool:
        """Load existing index from disk. Returns True if successful."""
        import faiss
        index_file = self._index_path / "index.faiss"
        chunks_file = self._index_path / "chunks.pkl"
        if not index_file.exists() or not chunks_file.exists():
            return False
        self._index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors, {len(self._chunks)} chunks")
        return True


class ChromaVectorStore:
    # ChromaDB-based vector store (optional, run via Docker Compose).

    def __init__(self):
        try:
            import chromadb
            self._client = chromadb.HttpClient(
                host=config.vector_store.chroma_host,
                port=config.vector_store.chroma_port,
            )
            self._collection = self._client.get_or_create_collection(
                name=config.vector_store.collection_name
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB connection failed: {e}")
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.vector_store.embedding_model)
        return self._embedder

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        embedder = self._get_embedder()
        texts = [c.content for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[c.chunk_id for c in chunks],
            metadatas=[{"source": c.source, "page": c.page or 0, "doc_type": c.doc_type} for c in chunks],
        )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection '{config.vector_store.collection_name}'")

    def search(self, query: str, k: int = 5) -> list[dict]:
        embedder = self._get_embedder()
        query_vec = embedder.encode([query]).tolist()
        results = self._collection.query(query_embeddings=query_vec, n_results=k)
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "doc_type": meta.get("doc_type", ""),
                "chunk_id": "",
                "score": float(dist),
            })
        return output

    def is_empty(self) -> bool:
        return self._collection.count() == 0


def get_vector_store() -> VectorStoreProtocol:
    """Factory: returns configured vector store instance."""
    backend = config.vector_store.backend.lower()
    if backend == "faiss":
        store = FAISSVectorStore()
        store.load() # Attempt to load existing index
        return store
    elif backend == "chroma":
        return ChromaVectorStore()
    else:
        raise ValueError(f"Unknown VECTOR_STORE '{backend}'. Options: 'faiss', 'chroma'")
