"""
Clinical War Room - Debate Schemas

Structured schemas for debate rounds, critiques, and voting.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class VoteChoice(str, Enum):
    """Vote choices for agents."""
    EXECUTE = "execute"           # Proceed with automated assessment
    ESCALATE = "escalate"         # Escalate to human review
    REFUSE = "refuse"             # Refuse to proceed (safety concern)
    REQUEST_MORE_DATA = "request_more_data"  # Need additional information


class Critique(BaseModel):
    """
    A critique from one agent to another.
    
    Critiques target logic flaws, overconfidence, or missing evidence.
    """
    critic_agent: str = Field(..., description="Agent making the critique")
    target_agent: str = Field(..., description="Agent being critiqued")
    critique_type: str = Field(
        default="general",
        description="Type: logic_flaw, overconfidence, missing_evidence, bias"
    )
    critique_text: str = Field(..., description="The critique explanation")
    severity: str = Field(default="moderate", description="low, moderate, high")
    suggested_adjustment: Optional[str] = Field(
        default=None,
        description="Suggested change to target's position"
    )
    
    def to_dict(self) -> dict:
        return {
            "critic_agent": self.critic_agent,
            "target_agent": self.target_agent,
            "critique_type": self.critique_type,
            "critique_text": self.critique_text,
            "severity": self.severity,
            "suggested_adjustment": self.suggested_adjustment,
        }


class Vote(BaseModel):
    """
    An agent's vote on recommended action.
    """
    agent_name: str = Field(..., description="Voting agent")
    vote_choice: VoteChoice = Field(..., description="The vote")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in vote")
    reasoning: str = Field(default="", description="Reason for vote")
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "vote_choice": self.vote_choice.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class AgentPosition(BaseModel):
    """
    An agent's position at a point in the debate.
    
    Includes opinion and any revisions made.
    """
    agent_name: str
    claim: str
    confidence: float
    risk: float
    evidence_count: int
    concerns_count: int
    has_veto: bool = False
    revision_note: Optional[str] = None
    
    @classmethod
    def from_agent_output(cls, output) -> "AgentPosition":
        """Create from AgentOutput."""
        return cls(
            agent_name=output.agent_name,
            claim=output.claim[:200],
            confidence=output.confidence,
            risk=output.risk,
            evidence_count=len(output.evidence),
            concerns_count=len(output.concerns),
            has_veto=output.veto,
        )
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "claim": self.claim,
            "confidence": self.confidence,
            "risk": self.risk,
            "evidence_count": self.evidence_count,
            "concerns_count": self.concerns_count,
            "has_veto": self.has_veto,
            "revision_note": self.revision_note,
        }


class DebateRound(BaseModel):
    """Summary of a debate round."""
    round_number: int
    round_name: str
    participants: List[str]
    summary: str


class DebateResult(BaseModel):
    """
    Complete output from a debate session.
    
    Contains all rounds, critiques, revisions, votes, and metrics.
    """
    case_id: str = Field(..., description="Case being debated")
    
    # Round outputs
    initial_opinions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Initial agent positions"
    )
    critiques: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Cross-agent critiques"
    )
    revised_opinions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Post-critique revised positions"
    )
    votes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Final agent votes"
    )
    
    # Metrics
    disagreement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Degree of disagreement (0=consensus, 1=complete disagreement)"
    )
    
    # Metadata
    rounds_completed: int = Field(default=0)
    total_agents: int = Field(default=0)
    has_veto: bool = Field(default=False)
    veto_agent: Optional[str] = Field(default=None)
    
    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "initial_opinions": self.initial_opinions,
            "critiques": self.critiques,
            "revised_opinions": self.revised_opinions,
            "votes": self.votes,
            "disagreement_score": round(self.disagreement_score, 4),
            "rounds_completed": self.rounds_completed,
            "total_agents": self.total_agents,
            "has_veto": self.has_veto,
            "veto_agent": self.veto_agent,
        }
    
    def summary(self) -> str:
        """Generate summary of debate."""
        vote_counts = {}
        for v in self.votes:
            choice = v.get("vote_choice", "unknown")
            vote_counts[choice] = vote_counts.get(choice, 0) + 1
        
        vote_str = ", ".join(f"{k}: {v}" for k, v in vote_counts.items())
        veto_str = f" [VETO by {self.veto_agent}]" if self.has_veto else ""
        
        return (
            f"Debate on {self.case_id}: "
            f"{self.total_agents} agents, "
            f"disagreement={self.disagreement_score:.0%}, "
            f"votes: {vote_str}{veto_str}"
        )
