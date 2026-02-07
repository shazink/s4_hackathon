"""
Clinical War Room - Final Decision Schema

Data model for the final recommendation produced by the system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from models.agent_output import AgentOutput, AgentVote


class FinalDecision(str, Enum):
    """
    The system's final recommendation.
    
    These are the ONLY possible outputs.
    The system NEVER recommends specific treatments.
    """
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    REFUSE_ACTION = "refuse_action"
    REQUEST_MORE_DATA = "request_more_data"


class DecisionReason(str, Enum):
    """Reason for the decision."""
    CONSENSUS_REACHED = "consensus_reached"
    SAFETY_VETO = "safety_veto"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    ETHICS_CONCERN = "ethics_concern"
    INSUFFICIENT_DATA = "insufficient_data"
    AGENT_DISAGREEMENT = "agent_disagreement"
    HUMAN_OVERRIDE = "human_override"
    RL_POLICY = "rl_policy"


@dataclass
class VoteSummary:
    """Summary of agent votes."""
    total_agents: int
    votes: Dict[str, int]  # vote -> count
    unanimous: bool
    majority_vote: Optional[AgentVote] = None
    disagreement_score: float = 0.0  # 0.0 = full agreement, 1.0 = no agreement
    
    def to_dict(self) -> dict:
        return {
            "total_agents": self.total_agents,
            "votes": self.votes,
            "unanimous": self.unanimous,
            "majority_vote": self.majority_vote.value if self.majority_vote else None,
            "disagreement_score": self.disagreement_score,
        }


@dataclass
class RiskAssessment:
    """Aggregated risk assessment."""
    overall_risk: float  # 0.0 to 1.0
    overall_confidence: float  # 0.0 to 1.0
    risk_factors: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    
    @property
    def is_high_risk(self) -> bool:
        return self.overall_risk >= 0.7
    
    @property
    def is_low_confidence(self) -> bool:
        return self.overall_confidence < 0.6
    
    def to_dict(self) -> dict:
        return {
            "overall_risk": self.overall_risk,
            "overall_confidence": self.overall_confidence,
            "risk_factors": self.risk_factors,
            "mitigations": self.mitigations,
            "is_high_risk": self.is_high_risk,
            "is_low_confidence": self.is_low_confidence,
        }


@dataclass
class DecisionExplanation:
    """
    Human-readable explanation of the decision.
    
    The system MUST be able to explain its reasoning.
    """
    summary: str
    key_factors: List[str]
    dissenting_opinions: List[str] = field(default_factory=list)
    unresolved_concerns: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_factors": self.key_factors,
            "dissenting_opinions": self.dissenting_opinions,
            "unresolved_concerns": self.unresolved_concerns,
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass
class WarRoomDecision:
    """
    Complete decision output from the Clinical War Room.
    
    This is the final output of the entire pipeline.
    """
    case_id: str
    decision: FinalDecision
    reason: DecisionReason
    
    # Assessment
    votes: VoteSummary
    risk: RiskAssessment
    explanation: DecisionExplanation
    
    # Agent outputs
    agent_opinions: List[AgentOutput] = field(default_factory=list)
    
    # Metadata
    decided_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    
    # Human review
    requires_human_review: bool = False
    human_override: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    # Safety flags
    safety_rules_triggered: List[str] = field(default_factory=list)
    ethics_flags: List[str] = field(default_factory=list)
    
    def approve(self, reviewer: str) -> None:
        """Record human approval."""
        self.human_override = "approved"
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.utcnow()
    
    def override(self, reviewer: str, new_decision: FinalDecision, reason: str) -> None:
        """Record human override."""
        self.decision = new_decision
        self.reason = DecisionReason.HUMAN_OVERRIDE
        self.human_override = reason
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "decision": self.decision.value,
            "reason": self.reason.value,
            "votes": self.votes.to_dict(),
            "risk": self.risk.to_dict(),
            "explanation": self.explanation.to_dict(),
            "agent_opinions": [a.to_dict() for a in self.agent_opinions],
            "decided_at": self.decided_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "requires_human_review": self.requires_human_review,
            "human_override": self.human_override,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "safety_rules_triggered": self.safety_rules_triggered,
            "ethics_flags": self.ethics_flags,
        }
