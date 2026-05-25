"""
This is a central configuration for the whole app.
All settings are loaded from environment variables via .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    # setting up anthropic keys for claude
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    # setting up ollama to ollama's port so that offline llm is enabled
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    # choosing a model from ollama
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral"))


@dataclass
class VectorStoreConfig:
    backend: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "faiss"))
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "intellidoc"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


@dataclass
class IngestConfig:
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    sample_docs_path: str = field(default_factory=lambda: os.getenv("SAMPLE_DOCS_PATH", "data/sample_docs"))
    metadata_registry_path: str = field(
        default_factory=lambda: os.getenv("METADATA_REGISTRY_PATH", "data/metadata_registry.json")
    )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    show_agent_trace: bool = field(
        default_factory=lambda: os.getenv("SHOW_AGENT_TRACE", "true").lower() == "true"
    )


# config instance
config = AppConfig()
