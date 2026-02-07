"""
Clinical War Room - Patient Case Schema

Data model for patient cases submitted to the system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid


class CaseStatus(str, Enum):
    """Status of a patient case in the pipeline."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    DEBATING = "debating"
    AWAITING_REVIEW = "awaiting_review"
    DECIDED = "decided"
    ESCALATED = "escalated"
    REFUSED = "refused"


class CasePriority(str, Enum):
    """Priority level of a patient case."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatientData:
    """
    Core patient information.
    
    Contains ONLY objective data, no opinions or recommendations.
    """
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    vitals: Dict[str, Any] = field(default_factory=dict)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    lab_results: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "sex": self.sex,
            "medical_history": self.medical_history,
            "current_medications": self.current_medications,
            "vitals": self.vitals,
            "sensor_data": self.sensor_data,
            "lab_results": self.lab_results,
            "notes": self.notes,
        }


@dataclass
class CaseMetadata:
    """Metadata about the case submission."""
    submitted_by: str
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    source_system: Optional[str] = None
    urgency_flag: bool = False
    notes: Optional[str] = None


@dataclass
class PatientCase:
    """
    Complete patient case submitted to the War Room.
    
    This is the primary input to the entire system.
    """
    case_id: str
    patient: PatientData
    query: str  # What is being asked about this patient
    status: CaseStatus = CaseStatus.PENDING
    priority: CasePriority = CasePriority.MEDIUM
    metadata: Optional[CaseMetadata] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        patient: PatientData,
        query: str,
        submitted_by: str = "api",
        priority: CasePriority = CasePriority.MEDIUM
    ) -> "PatientCase":
        """Factory method to create a new case."""
        return cls(
            case_id=str(uuid.uuid4()),
            patient=patient,
            query=query,
            priority=priority,
            metadata=CaseMetadata(submitted_by=submitted_by)
        )
    
    def update_status(self, new_status: CaseStatus) -> None:
        """Update case status with timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "patient": self.patient.to_dict(),
            "query": self.query,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
