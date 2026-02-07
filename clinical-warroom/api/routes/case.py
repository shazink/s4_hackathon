"""
Clinical War Room - Case Submission API

Endpoints for submitting and managing patient cases.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.case import PatientCase, PatientData, CaseStatus, CasePriority
from core.logging import logger


router = APIRouter()


# =============================================================================
# Pydantic Models (API layer)
# =============================================================================

class PatientDataRequest(BaseModel):
    """API request model for patient data."""
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    sensor_data: Dict[str, Any] = Field(default_factory=dict)
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class CaseSubmitRequest(BaseModel):
    """API request for submitting a new case."""
    patient: PatientDataRequest
    query: str = Field(..., description="What is being asked about this patient")
    priority: str = Field(default="medium", description="Case priority level")
    submitted_by: str = Field(default="api", description="Submitter identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient": {
                    "patient_id": "P12345",
                    "age": 72,
                    "sex": "M",
                    "medical_history": ["hypertension", "diabetes"],
                    "vitals": {"heart_rate": 78, "blood_pressure": "140/90"},
                },
                "query": "Assess fall risk and recommend intervention",
                "priority": "high",
                "submitted_by": "dr_smith",
            }
        }


class CaseResponse(BaseModel):
    """API response model for a case."""
    case_id: str
    status: str
    priority: str
    query: str
    patient_id: str
    created_at: datetime
    message: str


class CaseListResponse(BaseModel):
    """API response for listing cases."""
    cases: List[CaseResponse]
    total: int


# =============================================================================
# In-memory storage (temporary for Phase 0)
# =============================================================================

_cases: Dict[str, PatientCase] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def submit_case(request: CaseSubmitRequest):
    """
    Submit a new patient case for analysis.
    
    The case will be queued for processing by the War Room.
    """
    # Convert API model to domain model
    patient_data = PatientData(
        patient_id=request.patient.patient_id,
        age=request.patient.age,
        sex=request.patient.sex,
        medical_history=request.patient.medical_history,
        current_medications=request.patient.current_medications,
        vitals=request.patient.vitals,
        sensor_data=request.patient.sensor_data,
        lab_results=request.patient.lab_results,
        notes=request.patient.notes,
    )
    
    priority = CasePriority(request.priority)
    
    case = PatientCase.create(
        patient=patient_data,
        query=request.query,
        submitted_by=request.submitted_by,
        priority=priority,
    )
    
    # Store case
    _cases[case.case_id] = case
    
    logger.case_submitted(case.case_id, patient_data.patient_id)
    
    return CaseResponse(
        case_id=case.case_id,
        status=case.status.value,
        priority=case.priority.value,
        query=case.query,
        patient_id=patient_data.patient_id,
        created_at=case.created_at,
        message="Case submitted successfully. Processing not yet implemented (Phase 0).",
    )


@router.get("/cases/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str):
    """Get a specific case by ID."""
    if case_id not in _cases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found",
        )
    
    case = _cases[case_id]
    return CaseResponse(
        case_id=case.case_id,
        status=case.status.value,
        priority=case.priority.value,
        query=case.query,
        patient_id=case.patient.patient_id,
        created_at=case.created_at,
        message="",
    )


@router.get("/cases", response_model=CaseListResponse)
async def list_cases(
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
):
    """List all cases with optional filtering."""
    cases = list(_cases.values())
    
    # Filter by status if provided
    if status:
        cases = [c for c in cases if c.status.value == status]
    
    # Pagination
    total = len(cases)
    cases = cases[offset:offset + limit]
    
    return CaseListResponse(
        cases=[
            CaseResponse(
                case_id=c.case_id,
                status=c.status.value,
                priority=c.priority.value,
                query=c.query,
                patient_id=c.patient.patient_id,
                created_at=c.created_at,
                message="",
            )
            for c in cases
        ],
        total=total,
    )
