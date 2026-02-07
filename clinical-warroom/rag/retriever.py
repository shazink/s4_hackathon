"""
Clinical War Room - RAG Retriever

Query interface for retrieving relevant knowledge chunks.
NO reasoning, NO decisions, evidence ONLY.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from rag.ingestion import ingest_directory, DocumentChunk
from rag.store import VectorStore


@dataclass
class RetrievalResult:
    """Result from retrieval query."""
    chunk_id: str
    content: str
    source_file: str
    source_title: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "source_title": self.source_title,
            "similarity_score": round(self.similarity_score, 4),
            "metadata": self.metadata,
        }


class KnowledgeRetriever:
    """
    Retriever for medical knowledge base.
    
    This component:
    - Retrieves relevant text chunks given a query
    - Returns evidence, NOT conclusions
    - Does NOT reason or make decisions
    - Is completely independent of agents
    """
    
    def __init__(
        self,
        knowledge_dir: str,
        persist_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the retriever.
        
        Args:
            knowledge_dir: Path to knowledge base directory
            persist_dir: Path for persisting vector store
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
        """
        self.knowledge_dir = knowledge_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.store = VectorStore(persist_directory=persist_dir)
        self._chunks: Dict[str, DocumentChunk] = {}
        self._initialized = False
    
    def initialize(self, force_rebuild: bool = False) -> int:
        """
        Initialize the retriever by loading or building the index.
        
        Args:
            force_rebuild: If True, rebuild index even if persisted version exists
            
        Returns:
            Number of chunks indexed
        """
        # Try to load existing index
        if not force_rebuild and self.persist_dir and self.store.load():
            self._initialized = True
            return self.store.count()
        
        # Build new index
        chunks = ingest_directory(
            self.knowledge_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        if not chunks:
            return 0
        
        # Store chunk references
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Add to vector store
        self.store.add_documents(
            chunk_ids=[c.chunk_id for c in chunks],
            contents=[c.content for c in chunks],
            metadatas=[{
                "source_file": c.source_file,
                "source_title": c.source_title,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                **c.metadata,
            } for c in chunks],
        )
        
        # Persist if configured
        if self.persist_dir:
            self.store.save()
        
        self._initialized = True
        return len(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge chunks for a query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult objects
        """
        if not self._initialized:
            self.initialize()
        
        if not query.strip():
            return []
        
        results = self.store.search(
            query=query,
            top_k=top_k,
            threshold=min_similarity,
        )
        
        retrieval_results = []
        for entry, score in results:
            retrieval_results.append(RetrievalResult(
                chunk_id=entry.chunk_id,
                content=entry.content,
                source_file=entry.metadata.get("source_file", ""),
                source_title=entry.metadata.get("source_title", ""),
                similarity_score=score,
                metadata=entry.metadata,
            ))
        
        return retrieval_results
    
    def retrieve_for_case(
        self,
        gait_features: Dict[str, Any],
        patient_info: Dict[str, Any],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge for a clinical case.
        
        Builds a query from case features and retrieves
        relevant clinical guidelines.
        
        Args:
            gait_features: Gait analysis results
            patient_info: Patient demographics and history
            top_k: Maximum results
            
        Returns:
            List of RetrievalResult
        """
        # Build query from case features
        query_parts = []
        
        # Add gait-related terms
        gait_speed = gait_features.get("gait_speed", 0)
        if gait_speed < 0.8:
            query_parts.append("slow gait speed low mobility")
        
        asymmetry = gait_features.get("asymmetry_index", 0)
        if asymmetry > 0.15:
            query_parts.append("gait asymmetry unilateral weakness")
        
        variability = gait_features.get("variability", 0)
        if variability > 0.08:
            query_parts.append("step variability fall risk")
        
        # Add patient history terms
        history = patient_info.get("medical_history", [])
        for condition in history:
            condition_lower = condition.lower()
            if "parkinson" in condition_lower:
                query_parts.append("Parkinson's disease gait freezing")
            if "stroke" in condition_lower or "cva" in condition_lower:
                query_parts.append("stroke CVA hemiparetic gait")
            if "diabet" in condition_lower:
                query_parts.append("diabetes neuropathy fall risk")
            if "fall" in condition_lower:
                query_parts.append("previous falls fall risk recurrent")
        
        # Add age-related terms
        age = patient_info.get("age")
        if age and age >= 75:
            query_parts.append("elderly geriatric fall risk assessment")
        
        # Default query if nothing specific
        if not query_parts:
            query_parts.append("gait analysis fall risk assessment")
        
        query = " ".join(query_parts)
        
        return self.retrieve(query, top_k=top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "initialized": self._initialized,
            "total_chunks": self.store.count(),
            "knowledge_dir": self.knowledge_dir,
            "persist_dir": self.persist_dir,
        }


# Default retriever instance
_default_retriever: Optional[KnowledgeRetriever] = None


def get_retriever(
    knowledge_dir: str = None,
    persist_dir: str = None,
) -> KnowledgeRetriever:
    """
    Get or create the default retriever instance.
    
    Args:
        knowledge_dir: Path to knowledge base
        persist_dir: Path for persistence
        
    Returns:
        KnowledgeRetriever instance
    """
    global _default_retriever
    
    if _default_retriever is None:
        if knowledge_dir is None:
            # Default paths
            base_dir = Path(__file__).parent.parent
            knowledge_dir = str(base_dir / "data" / "knowledge")
            persist_dir = str(base_dir / "data" / "vector_store")
        
        _default_retriever = KnowledgeRetriever(
            knowledge_dir=knowledge_dir,
            persist_dir=persist_dir,
        )
    
    return _default_retriever
