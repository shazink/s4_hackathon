"""
Clinical War Room - RAG Vector Store

Simple vector store using sentence embeddings.
Uses local storage, no external dependencies like ChromaDB for simplicity.
"""

import os
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re


@dataclass
class VectorEntry:
    """A document chunk with its embedding vector."""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleEmbedding:
    """
    Simple TF-IDF-like embedding for text.
    
    This is a lightweight embedding that doesn't require external models.
    For production, replace with sentence-transformers.
    """
    
    def __init__(self, vocabulary: Optional[Dict[str, int]] = None):
        self.vocabulary = vocabulary or {}
        self.idf = {}
        self.dimension = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation and split on whitespace
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize by document length
        length = len(tokens) or 1
        return {k: v / length for k, v in tf.items()}
    
    def fit(self, documents: List[str]) -> None:
        """Build vocabulary and IDF from documents."""
        # Build vocabulary
        all_tokens = set()
        doc_counts = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_tokens.update(tokens)
            for token in tokens:
                doc_counts[token] = doc_counts.get(token, 0) + 1
        
        # Create vocabulary mapping
        self.vocabulary = {token: i for i, token in enumerate(sorted(all_tokens))}
        self.dimension = len(self.vocabulary)
        
        # Compute IDF
        num_docs = len(documents) or 1
        self.idf = {
            token: math.log(num_docs / (count + 1)) + 1
            for token, count in doc_counts.items()
        }
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if not self.vocabulary:
            return []
        
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        
        # Create TF-IDF vector
        vector = [0.0] * self.dimension
        for token, freq in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf.get(token, 1.0)
                vector[idx] = freq * idf
        
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
    
    def save(self, path: str) -> None:
        """Save embedding model to file."""
        data = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "dimension": self.dimension,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load embedding model from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocabulary = data["vocabulary"]
        self.idf = data["idf"]
        self.dimension = data["dimension"]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1)) or 1.0
    norm2 = math.sqrt(sum(b * b for b in vec2)) or 1.0
    
    return dot_product / (norm1 * norm2)


class VectorStore:
    """
    Simple in-memory vector store with persistence.
    
    For production, replace with FAISS or ChromaDB.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.entries: List[VectorEntry] = []
        self.embedding_model = SimpleEmbedding()
        self.persist_directory = persist_directory
        self._is_fitted = False
    
    def add_documents(
        self,
        chunk_ids: List[str],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            chunk_ids: Unique identifiers for each chunk
            contents: Text content of each chunk
            metadatas: Optional metadata for each chunk
        """
        if not self._is_fitted:
            # Fit embedding model on all contents
            self.embedding_model.fit(contents)
            self._is_fitted = True
        
        metadatas = metadatas or [{} for _ in contents]
        
        for chunk_id, content, metadata in zip(chunk_ids, contents, metadatas):
            embedding = self.embedding_model.embed(content)
            entry = VectorEntry(
                chunk_id=chunk_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
            )
            self.entries.append(entry)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[VectorEntry, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (VectorEntry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        query_embedding = self.embedding_model.embed(query)
        
        # Compute similarities
        results = []
        for entry in self.entries:
            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity >= threshold:
                results.append((entry, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def save(self) -> None:
        """Persist vector store to disk."""
        if not self.persist_directory:
            return
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Save entries
        entries_data = [
            {
                "chunk_id": e.chunk_id,
                "content": e.content,
                "embedding": e.embedding,
                "metadata": e.metadata,
            }
            for e in self.entries
        ]
        
        entries_path = os.path.join(self.persist_directory, "entries.json")
        with open(entries_path, 'w') as f:
            json.dump(entries_data, f)
        
        # Save embedding model
        model_path = os.path.join(self.persist_directory, "embedding_model.json")
        self.embedding_model.save(model_path)
    
    def load(self) -> bool:
        """Load vector store from disk."""
        if not self.persist_directory:
            return False
        
        entries_path = os.path.join(self.persist_directory, "entries.json")
        model_path = os.path.join(self.persist_directory, "embedding_model.json")
        
        if not os.path.exists(entries_path) or not os.path.exists(model_path):
            return False
        
        try:
            # Load embedding model
            self.embedding_model.load(model_path)
            self._is_fitted = True
            
            # Load entries
            with open(entries_path, 'r') as f:
                entries_data = json.load(f)
            
            self.entries = [
                VectorEntry(
                    chunk_id=e["chunk_id"],
                    content=e["content"],
                    embedding=e["embedding"],
                    metadata=e["metadata"],
                )
                for e in entries_data
            ]
            
            return True
        except Exception:
            return False
    
    def count(self) -> int:
        """Return number of entries in store."""
        return len(self.entries)
    
    def clear(self) -> None:
        """Clear all entries."""
        self.entries = []
        self._is_fitted = False
