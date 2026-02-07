"""
Clinical War Room - HITL Data Models

Data models for human-in-the-loop review.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class HumanAction(str, Enum):
    """Actions a human reviewer can take."""
    APPROVE = "APPROVE"           # Approve system recommendation
    OVERRIDE = "OVERRIDE"         # Override with different action
    REQUEST_DATA = "REQUEST_DATA" # Request more data
    DEFER = "DEFER"              # Defer decision


class SystemAction(str, Enum):
    """Possible system recommendations."""
    EXECUTE = "EXECUTE"
    ESCALATE = "ESCALATE"
    REFUSE = "REFUSE"
    REQUEST_MORE_DATA = "REQUEST_MORE_DATA"


class ReviewStatus(str, Enum):
    """Status of a review request."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"


class CaseSummary(BaseModel):
    """Summary of a case for human review."""
    case_id: str
    patient_summary: str = ""
    
    # MCP-computed metrics
    mcp_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent opinions from debate
    agent_opinions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Debate summary
    debate_summary: str = ""
    disagreement_score: float = 0.0
    
    # Rule layer outcome
    safety_allowed: bool = True
    safety_explanation: str = ""
    triggered_rule: Optional[str] = None
    
    # RL recommendation
    system_action: SystemAction
    rl_confidence: float = 0.5
    rl_explanation: str = ""


class ReviewRequest(BaseModel):
    """Request for human review."""
    request_id: str
    case_id: str
    case_summary: CaseSummary
    
    # Why review is required
    review_reason: str
    is_mandatory: bool = False
    
    # Status
    status: ReviewStatus = ReviewStatus.PENDING
    
    # Timestamps
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "case_id": self.case_id,
            "review_reason": self.review_reason,
            "is_mandatory": self.is_mandatory,
            "status": self.status.value,
            "system_action": self.case_summary.system_action.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class HumanDecision(BaseModel):
    """A human reviewer's decision."""
    decision_id: str
    case_id: str
    request_id: str
    
    # System recommendation
    system_action: SystemAction
    
    # Human action
    human_action: HumanAction
    
    # If OVERRIDE, what action they chose
    override_action: Optional[SystemAction] = None
    
    # Notes / justification
    notes: str = ""
    
    # Reviewer info
    reviewer_id: str
    reviewer_name: Optional[str] = None
    
    # Timestamp
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "case_id": self.case_id,
            "request_id": self.request_id,
            "system_action": self.system_action.value,
            "human_action": self.human_action.value,
            "override_action": self.override_action.value if self.override_action else None,
            "notes": self.notes,
            "reviewer_id": self.reviewer_id,
            "reviewer_name": self.reviewer_name,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @property
    def final_action(self) -> str:
        """Get the final action after human review."""
        if self.human_action == HumanAction.APPROVE:
            return self.system_action.value
        elif self.human_action == HumanAction.OVERRIDE:
            return self.override_action.value if self.override_action else self.system_action.value
        elif self.human_action == HumanAction.REQUEST_DATA:
            return "REQUEST_MORE_DATA"
        else:
            return "DEFER"


class DecisionSubmission(BaseModel):
    """API request to submit a human decision."""
    case_id: str
    request_id: str
    human_action: HumanAction
    override_action: Optional[SystemAction] = None
    notes: str = ""
    reviewer_id: str
    reviewer_name: Optional[str] = None


class DecisionResponse(BaseModel):
    """API response after submitting a decision."""
    success: bool
    decision_id: str
    final_action: str
    message: str
