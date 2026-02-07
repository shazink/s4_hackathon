"""
Clinical War Room - Patient Store

ChromaDB-based patient records storage.
Supports: PDF, DOCX, TXT, and Images (OCR)
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import chromadb
from chromadb.config import Settings

# PDF support
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# DOCX support
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Image OCR support
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


@dataclass
class PatientRecord:
    """A patient record stored in the database."""
    patient_id: str
    name: str
    age: int
    gender: str
    medical_history: str
    gait_data: Dict[str, float]
    chief_complaint: str
    raw_text: str
    created_at: str
    updated_at: str


class PatientStore:
    """
    ChromaDB-based patient records store.
    
    Supports:
    - Adding patients from text or PDF
    - Semantic search for patient records
    - Full patient retrieval by ID
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "patient_records",
    ):
        self.persist_directory = persist_directory or str(
            Path(__file__).parent.parent / "data" / "patient_db"
        )
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Patient records for clinical analysis"}
        )
        
        # Also keep a JSON backup of full patient records
        self.records_file = Path(self.persist_directory) / "patient_records.json"
        self._records: Dict[str, PatientRecord] = {}
        self._load_records()
    
    def _load_records(self):
        """Load patient records from JSON backup."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                    for pid, rec in data.items():
                        self._records[pid] = PatientRecord(**rec)
            except Exception as e:
                print(f"Warning: Failed to load patient records: {e}")
    
    def _save_records(self):
        """Save patient records to JSON backup."""
        with open(self.records_file, 'w') as f:
            json.dump({pid: asdict(rec) for pid, rec in self._records.items()}, f, indent=2)
    
    def add_patient_from_text(
        self,
        text: str,
        name: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        gait_data: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Add a patient from text description.
        
        Args:
            text: Full text describing the patient
            name: Patient name (extracted from text if not provided)
            age: Patient age
            gender: Patient gender
            gait_data: Optional gait measurements
            
        Returns:
            Patient ID
        """
        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        
        # Create patient record
        record = PatientRecord(
            patient_id=patient_id,
            name=name or f"Patient {patient_id[-4:]}",
            age=age or 0,
            gender=gender or "Unknown",
            medical_history=text,
            gait_data=gait_data or {},
            chief_complaint="",
            raw_text=text,
            created_at=now,
            updated_at=now,
        )
        
        # Store in memory and JSON backup
        self._records[patient_id] = record
        self._save_records()
        
        # Store in ChromaDB for semantic search
        self._collection.add(
            ids=[patient_id],
            documents=[text],
            metadatas=[{
                "patient_id": patient_id,
                "name": record.name,
                "age": str(record.age),
                "gender": record.gender,
            }],
        )
        
        print(f"Patient {patient_id} added to ChromaDB")
        return patient_id
    
    def add_patient_from_pdf(self, pdf_path: str) -> str:
        """Add a patient from a PDF file."""
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 not installed. Run: pip install pypdf2")
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return self.add_patient_from_text(text.strip())
    
    def add_patient_from_docx(self, docx_path: str) -> str:
        """Add a patient from a DOCX file."""
        if not HAS_DOCX:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        doc = DocxDocument(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        return self.add_patient_from_text(text.strip())
    
    def add_patient_from_txt(self, txt_path: str) -> str:
        """Add a patient from a TXT file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.add_patient_from_text(text.strip())
    
    def add_patient_from_image(self, image_path: str) -> str:
        """Add a patient from an image file using OCR."""
        if not HAS_OCR:
            raise ImportError("OCR not available. Run: pip install pillow pytesseract")
        
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            raise ValueError("Could not extract text from image. Ensure image contains readable text.")
        
        return self.add_patient_from_text(text.strip())
    
    def add_patient_from_file(self, file_path: str) -> str:
        """
        Add a patient from any supported file type.
        Automatically detects file type and uses appropriate method.
        
        Supported: PDF, DOCX, TXT, PNG, JPG, JPEG
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.add_patient_from_pdf(file_path)
        elif ext == '.docx':
            return self.add_patient_from_docx(file_path)
        elif ext in ['.txt', '.md']:
            return self.add_patient_from_txt(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self.add_patient_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: pdf, docx, txt, png, jpg")
    
    def search_patients(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for patients by natural language query using ChromaDB.
        
        Args:
            query: Search query (e.g., "patient with fall history")
            top_k: Number of results
            
        Returns:
            List of matching patients with scores
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        matches = []
        if results['ids'] and results['ids'][0]:
            for i, patient_id in enumerate(results['ids'][0]):
                record = self._records.get(patient_id)
                if record:
                    distance = results['distances'][0][i] if results['distances'] else 0
                    matches.append({
                        "patient_id": patient_id,
                        "name": record.name,
                        "age": record.age,
                        "preview": record.raw_text[:200] + "..." if len(record.raw_text) > 200 else record.raw_text,
                        "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                    })
        
        return matches
    
    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        """Get a patient by ID."""
        return self._records.get(patient_id)
    
    def list_patients(self) -> List[Dict[str, Any]]:
        """List all patients."""
        return [
            {
                "patient_id": rec.patient_id,
                "name": rec.name,
                "age": rec.age,
                "gender": rec.gender,
                "created_at": rec.created_at,
            }
            for rec in self._records.values()
        ]
    
    def count(self) -> int:
        """Return number of patients."""
        return len(self._records)
    
    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient by ID."""
        if patient_id in self._records:
            del self._records[patient_id]
            self._save_records()
            
            # Also delete from ChromaDB
            try:
                self._collection.delete(ids=[patient_id])
            except Exception:
                pass
            
            return True
        return False


# Global store instance
_patient_store: Optional[PatientStore] = None


def get_patient_store() -> PatientStore:
    """Get the global patient store instance."""
    global _patient_store
    if _patient_store is None:
        _patient_store = PatientStore()
    return _patient_store
