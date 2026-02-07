"""RAG Module - Retrieval-Augmented Generation for clinical knowledge."""

from rag.ingestion import (
    Document,
    DocumentChunk,
    load_document,
    load_documents_from_directory,
    chunk_document,
    ingest_directory,
)
from rag.store import VectorStore
from rag.retriever import (
    KnowledgeRetriever,
    RetrievalResult,
    get_retriever,
)

__all__ = [
    # Ingestion
    "Document",
    "DocumentChunk",
    "load_document",
    "load_documents_from_directory",
    "chunk_document",
    "ingest_directory",
    # Store
    "VectorStore",
    # Retriever
    "KnowledgeRetriever",
    "RetrievalResult",
    "get_retriever",
]
