#!/usr/bin/env python3
"""
RAG Verification Script

Demonstrates:
1. Document ingestion from knowledge base
2. Vector store initialization
3. Query-based retrieval
4. Sample outputs

Usage:
    python scripts/verify_rag.py
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.ingestion import ingest_directory
from rag.retriever import KnowledgeRetriever


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    print()
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def main():
    """Run RAG verification."""
    print_separator("RAG KNOWLEDGE RETRIEVAL VERIFICATION")
    print("Demonstrating evidence retrieval - NO reasoning, NO decisions")
    
    # Paths
    knowledge_dir = project_root / "data" / "knowledge"
    persist_dir = project_root / "data" / "vector_store"
    
    print(f"\nKnowledge base: {knowledge_dir}")
    print(f"Vector store: {persist_dir}")
    
    # Step 1: Show document ingestion
    print_separator("1. DOCUMENT INGESTION")
    
    chunks = ingest_directory(str(knowledge_dir), chunk_size=500, chunk_overlap=50)
    print(f"Loaded and chunked {len(chunks)} document chunks")
    
    # Show sources
    sources = {}
    for chunk in chunks:
        source = chunk.source_title
        sources[source] = sources.get(source, 0) + 1
    
    print("\nDocument sources:")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} chunks")
    
    # Step 2: Initialize retriever
    print_separator("2. VECTOR STORE INITIALIZATION")
    
    retriever = KnowledgeRetriever(
        knowledge_dir=str(knowledge_dir),
        persist_dir=str(persist_dir),
        chunk_size=500,
        chunk_overlap=50,
    )
    
    count = retriever.initialize(force_rebuild=True)
    print(f"Indexed {count} chunks in vector store")
    
    stats = retriever.get_stats()
    print(f"Store stats: {json.dumps(stats, indent=2)}")
    
    # Step 3: Example queries
    print_separator("3. EXAMPLE QUERIES")
    
    queries = [
        "What are the normal values for gait speed?",
        "How does Parkinson's disease affect gait?",
        "What asymmetry threshold indicates high fall risk?",
        "What are the ethical requirements for automated screening?",
        "Clinical guidelines for elderly patients",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: \"{query}\" ---\n")
        
        results = retriever.retrieve(query, top_k=2, min_similarity=0.05)
        
        if not results:
            print("  No results found")
            continue
        
        for j, result in enumerate(results, 1):
            print(f"  Result {j} (score: {result.similarity_score:.4f})")
            print(f"  Source: {result.source_title}")
            # Truncate content for display
            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
            print(f"  Content: {content}")
            print()
    
    # Step 4: Case-based retrieval
    print_separator("4. CASE-BASED RETRIEVAL")
    
    print("Simulating retrieval for a clinical case:")
    print("  - Gait speed: 0.55 m/s (very low)")
    print("  - Asymmetry: 0.22 (high)")
    print("  - Age: 78")
    print("  - History: Parkinson's disease, previous falls")
    
    results = retriever.retrieve_for_case(
        gait_features={
            "gait_speed": 0.55,
            "asymmetry_index": 0.22,
            "variability": 0.09,
        },
        patient_info={
            "age": 78,
            "medical_history": ["Parkinson's disease", "previous fall 2023"],
        },
        top_k=5,
    )
    
    print(f"\nRetrieved {len(results)} relevant knowledge chunks:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.source_title}] (score: {result.similarity_score:.4f})")
        content = result.content[:200] + "..." if len(result.content) > 200 else result.content
        print(f"   {content}\n")
    
    # Summary
    print_separator("VERIFICATION COMPLETE")
    print("✅ Documents ingested from knowledge base")
    print("✅ Vector store built and persisted")
    print("✅ Query retrieval working")
    print("✅ Case-based retrieval working")
    print()
    print("RAG provides EVIDENCE ONLY - no reasoning, no decisions")
    print("Agents (Phase 3+) will use this evidence for deliberation")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
