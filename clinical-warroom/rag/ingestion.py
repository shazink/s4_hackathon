"""
Clinical War Room - RAG Document Ingestion

Loads, chunks, and prepares documents for vector storage.
NO LLM reasoning. Evidence retrieval only.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""
    chunk_id: str
    content: str
    source_file: str
    source_title: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "source_title": self.source_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """A loaded document with metadata."""
    file_path: str
    title: str
    content: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_title_from_markdown(content: str) -> str:
    """Extract title from first H1 heading in markdown."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def infer_topic_from_path(file_path: str) -> str:
    """Infer document topic from file path."""
    path = Path(file_path)
    name = path.stem.replace('_', ' ').replace('-', ' ').title()
    return name


def load_document(file_path: str) -> Optional[Document]:
    """
    Load a document from file.
    
    Supports:
    - Markdown (.md)
    - Plain text (.txt)
    """
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    # Determine file type
    suffix = path.suffix.lower()
    if suffix not in ['.md', '.txt', '.markdown']:
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None
    
    # Extract title
    if suffix in ['.md', '.markdown']:
        title = extract_title_from_markdown(content)
        file_type = 'markdown'
    else:
        title = ""
        file_type = 'text'
    
    if not title:
        title = infer_topic_from_path(file_path)
    
    return Document(
        file_path=str(path.absolute()),
        title=title,
        content=content,
        file_type=file_type,
        metadata={
            "filename": path.name,
            "topic": infer_topic_from_path(file_path),
        }
    )


def load_documents_from_directory(
    directory: str,
    extensions: List[str] = None
) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to directory
        extensions: List of file extensions to include (default: .md, .txt)
        
    Returns:
        List of loaded documents
    """
    if extensions is None:
        extensions = ['.md', '.txt', '.markdown']
    
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return documents
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            doc = load_document(str(file_path))
            if doc:
                documents.append(doc)
    
    return documents


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None,
) -> List[str]:
    """
    Split text into chunks with configurable size and overlap.
    
    Uses hierarchical splitting:
    1. Try to split on paragraph breaks
    2. Then sentence breaks
    3. Then word breaks
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        separators: Custom separators (default: paragraphs, sentences)
        
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ']
    
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    current_chunk = ""
    
    # Split by first separator
    sep = separators[0]
    parts = text.split(sep)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # If adding this part exceeds chunk size
        if len(current_chunk) + len(part) + len(sep) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of current chunk
                if chunk_overlap > 0:
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:].strip() + sep
                else:
                    current_chunk = ""
            
            # If single part is too long, recursively chunk it
            if len(part) > chunk_size and len(separators) > 1:
                sub_chunks = chunk_text(
                    part, 
                    chunk_size, 
                    chunk_overlap, 
                    separators[1:]
                )
                chunks.extend(sub_chunks[:-1] if len(sub_chunks) > 1 else [])
                if sub_chunks:
                    current_chunk = sub_chunks[-1]
            else:
                current_chunk = part
        else:
            if current_chunk:
                current_chunk += sep + part
            else:
                current_chunk = part
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Chunk a document into smaller pieces with metadata.
    
    Args:
        document: Document to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of DocumentChunk objects
    """
    text_chunks = chunk_text(
        document.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    doc_chunks = []
    for i, chunk_text_content in enumerate(text_chunks):
        chunk_id = f"{Path(document.file_path).stem}_{i:03d}"
        
        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            content=chunk_text_content,
            source_file=document.file_path,
            source_title=document.title,
            chunk_index=i,
            total_chunks=len(text_chunks),
            metadata={
                **document.metadata,
                "file_type": document.file_type,
            }
        )
        doc_chunks.append(doc_chunk)
    
    return doc_chunks


def ingest_directory(
    directory: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Load and chunk all documents from a directory.
    
    Args:
        directory: Path to knowledge base directory
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all document chunks
    """
    documents = load_documents_from_directory(directory)
    
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    
    return all_chunks
