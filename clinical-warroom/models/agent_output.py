"""
Clinical War Room - Agent Output Schema

Structured output format for specialist agent opinions.
All agents MUST produce outputs conforming to this schema.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AgentVote(str, Enum):
    """Agent's vote on the recommended action."""
    EXECUTE = "execute"       # Proceed with action
    ESCALATE = "escalate"     # Requires human review
    REFUSE = "refuse"         # Do not proceed
    REQUEST_DATA = "request_data"  # Need more information
    ABSTAIN = "abstain"       # Cannot make determination


@dataclass
class Evidence:
    """
    Evidence supporting an agent's claim.
    
    Evidence comes from MCP tools or RAG retrieval,
    NEVER from the agent's own reasoning.
    """
    source: str  # e.g., "gait_feature_extractor", "clinical_guidelines"
    content: str
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Concern:
    """
    A concern or risk identified by an agent.
    
    Concerns are explicitly tracked to ensure they're
    addressed during debate and decision-making.
    """
    description: str
    severity: str  # "low", "medium", "high", "critical"
    mitigation: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
        }


@dataclass
class AgentOutput:
    """
    Structured output from a specialist agent.
    
    This is the standard format for all agent analyses.
    The debate engine and coordinator rely on this structure.
    """
    agent_name: str
    case_id: str
    
    # Core opinion
    claim: str  # The agent's main claim/assessment
    vote: AgentVote
    confidence: float  # 0.0 to 1.0
    risk: float  # 0.0 to 1.0
    
    # Supporting information
    reasoning: str
    evidence: List[Evidence] = field(default_factory=list)
    concerns: List[Concern] = field(default_factory=list)
    
    # Metadata
    analysis_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Revision tracking (updated during debate)
    revision_count: int = 0
    original_confidence: Optional[float] = None
    original_vote: Optional[AgentVote] = None
    
    def __post_init__(self):
        """Store original values for tracking revisions."""
        if self.original_confidence is None:
            self.original_confidence = self.confidence
        if self.original_vote is None:
            self.original_vote = self.vote
    
    def revise(
        self,
        new_confidence: float,
        new_vote: AgentVote,
        new_reasoning: str
    ) -> None:
        """Revise opinion after receiving critiques."""
        self.confidence = new_confidence
        self.vote = new_vote
        self.reasoning = new_reasoning
        self.revision_count += 1
    
    @property
    def confidence_changed(self) -> bool:
        """Check if confidence changed from original."""
        return self.confidence != self.original_confidence
    
    @property
    def vote_changed(self) -> bool:
        """Check if vote changed from original."""
        return self.vote != self.original_vote
    
    def has_critical_concerns(self) -> bool:
        """Check if agent raised any critical concerns."""
        return any(c.severity == "critical" for c in self.concerns)
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "case_id": self.case_id,
            "claim": self.claim,
            "vote": self.vote.value,
            "confidence": self.confidence,
            "risk": self.risk,
            "reasoning": self.reasoning,
            "evidence": [e.to_dict() for e in self.evidence],
            "concerns": [c.to_dict() for c in self.concerns],
            "revision_count": self.revision_count,
            "original_confidence": self.original_confidence,
            "original_vote": self.original_vote.value if self.original_vote else None,
            "timestamp": self.timestamp.isoformat(),
        }
