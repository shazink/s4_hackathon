"""
Quick test script to verify RAG/ChromaDB functionality
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.patient_store import get_patient_store

def test_rag():
    print("=" * 60)
    print("Testing RAG/ChromaDB Functionality")
    print("=" * 60)
    
    store = get_patient_store()
    
    # 1. Check how many patients are stored
    count = store.count()
    print(f"\n✓ Total patients in database: {count}")
    
    # 2. List all patients
    print(f"\n✓ Listing all patients:")
    patients = store.list_patients()
    for p in patients:
        print(f"  - {p['name']} (ID: {p['patient_id']}, Age: {p['age']}, Gender: {p['gender']})")
    
    # 3. Test semantic search
    print(f"\n✓ Testing semantic search:")
    test_queries = [
        "patient with diabetes",
        "elderly patient",
        "mobility issues",
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = store.search_patients(query, top_k=3)
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['name']} (Score: {result['score']:.3f})")
                print(f"       Preview: {result['preview'][:100]}...")
        else:
            print("    No results found")
    
    # 4. Test retrieval by ID
    if patients:
        test_id = patients[0]['patient_id']
        print(f"\n✓ Testing retrieval by ID: {test_id}")
        record = store.get_patient(test_id)
        if record:
            print(f"  Name: {record.name}")
            print(f"  Age: {record.age}")
            print(f"  Medical History Preview: {record.medical_history[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ RAG Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_rag()
