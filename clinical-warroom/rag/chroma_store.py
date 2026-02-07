"""
Clinical War Room - ChromaDB Vector Store

ChromaDB-based vector store for RAG retrieval.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except (ImportError, Exception) as e:
    # ChromaDB may fail on Python 3.14 due to pydantic v1 compatibility
    HAS_CHROMADB = False

from core.logging import logger


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class ChromaStore:
    """
    ChromaDB-based vector store for clinical knowledge retrieval.
    
    Uses ChromaDB's built-in embedding function for simplicity.
    Falls back to local TF-IDF if ChromaDB is not available.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "clinical_knowledge",
    ):
        self.persist_directory = persist_directory or str(
            Path(__file__).parent.parent / "data" / "chroma_db"
        )
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        
        if not HAS_CHROMADB:
            logger.warning("ChromaDB not installed. Using fallback TF-IDF store.")
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            if not HAS_CHROMADB:
                raise ImportError("ChromaDB not installed")
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client
    
    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Clinical knowledge for gait analysis"}
            )
        return self._collection
    
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Unique identifiers for each document
            documents: Text content of each document
            metadatas: Optional metadata for each document
        """
        if not HAS_CHROMADB:
            logger.warning("ChromaDB not available, documents not indexed")
            return
        
        collection = self._get_collection()
        
        # ChromaDB handles embeddings automatically
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if not HAS_CHROMADB:
            return []
        
        collection = self._get_collection()
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        # Parse results
        retrieval_results = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                retrieval_results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=results['documents'][0][i] if results['documents'] else "",
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    score=1.0 - (results['distances'][0][i] if results['distances'] else 0),
                ))
        
        return retrieval_results
    
    def count(self) -> int:
        """Return number of documents in collection."""
        if not HAS_CHROMADB:
            return 0
        return self._get_collection().count()
    
    def clear(self) -> None:
        """Clear all documents from collection."""
        if not HAS_CHROMADB:
            return
        
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
        except Exception:
            pass
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        return HAS_CHROMADB


# Global store instance
_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Get the global ChromaDB store instance."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def index_knowledge_documents(data_dir: Optional[str] = None) -> int:
    """
    Index all knowledge documents from the data directory.
    
    Args:
        data_dir: Path to knowledge documents directory
        
    Returns:
        Number of documents indexed
    """
    store = get_chroma_store()
    
    if not store.is_available():
        logger.warning("ChromaDB not available, skipping indexing")
        return 0
    
    data_dir = data_dir or str(Path(__file__).parent.parent / "data" / "knowledge")
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Knowledge directory not found: {data_dir}")
        return 0
    
    # Find all markdown files
    md_files = list(data_path.glob("*.md"))
    
    if not md_files:
        logger.warning("No markdown files found in knowledge directory")
        return 0
    
    ids = []
    documents = []
    metadatas = []
    
    for md_file in md_files:
        content = md_file.read_text()
        
        # Split into chunks (simple paragraph-based chunking)
        chunks = [c.strip() for c in content.split("\n\n") if c.strip() and len(c.strip()) > 50]
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{md_file.stem}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": md_file.name,
                "chunk_index": i,
            })
    
    if documents:
        # Clear existing and add new
        store.clear()
        store.add_documents(ids, documents, metadatas)
    
    logger.info(f"Indexed {len(documents)} chunks from {len(md_files)} files")
    return len(documents)
