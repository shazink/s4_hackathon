"""
Clinical War Room - Agent Output Schema

Structured output format for all specialist agents.
Validated with Pydantic.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    """A piece of evidence supporting the agent's claim."""
    source: str = Field(..., description="Source of evidence (MCP tool, RAG, etc.)")
    content: str = Field(..., description="The evidence content")
    relevance: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="How relevant this evidence is (0-1)"
    )


class Concern(BaseModel):
    """A concern raised by the agent."""
    description: str = Field(..., description="Description of the concern")
    severity: str = Field(default="moderate", description="low, moderate, high, critical")
    mitigation: Optional[str] = Field(default=None, description="Suggested mitigation")


class AgentOutput(BaseModel):
    """
    Mandatory output structure for all specialist agents.
    
    This schema enforces a consistent structure across all agent outputs,
    enabling downstream aggregation and debate.
    """
    
    agent_name: str = Field(..., description="Name of the agent")
    
    claim: str = Field(
        ..., 
        description="The agent's primary claim or assessment",
        min_length=10,
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence in the claim (0-1)"
    )
    
    risk: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Assessed risk level (0-1)"
    )
    
    risk_level: RiskLevel = Field(
        default=RiskLevel.UNKNOWN,
        description="Categorical risk level"
    )
    
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence supporting the claim"
    )
    
    concerns: List[Concern] = Field(
        default_factory=list,
        description="Concerns or caveats"
    )
    
    reasoning: str = Field(
        default="",
        description="Detailed reasoning for the claim"
    )
    
    veto: bool = Field(
        default=False,
        description="Whether this agent vetoes proceeding (Ethics agent only)"
    )
    
    veto_reason: Optional[str] = Field(
        default=None,
        description="Reason for veto if veto=True"
    )
    
    @model_validator(mode='after')
    def set_risk_level_from_risk(self):
        """Auto-set risk level from risk score if not explicitly set."""
        if self.risk_level == RiskLevel.UNKNOWN:
            if self.risk >= 0.75:
                self.risk_level = RiskLevel.CRITICAL
            elif self.risk >= 0.5:
                self.risk_level = RiskLevel.HIGH
            elif self.risk >= 0.25:
                self.risk_level = RiskLevel.MODERATE
            else:
                self.risk_level = RiskLevel.LOW
        return self
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "claim": self.claim,
            "confidence": self.confidence,
            "risk": self.risk,
            "risk_level": self.risk_level.value,
            "evidence": [e.model_dump() for e in self.evidence],
            "concerns": [c.model_dump() for c in self.concerns],
            "reasoning": self.reasoning,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
        }
    
    def summary(self) -> str:
        """Generate a brief summary."""
        veto_note = " [VETO]" if self.veto else ""
        return (
            f"{self.agent_name}{veto_note}: {self.claim[:100]}... "
            f"(confidence: {self.confidence:.0%}, risk: {self.risk:.0%})"
        )


class CaseContext(BaseModel):
    """
    Input context provided to each agent.
    
    Contains patient case, MCP tool outputs, and RAG knowledge.
    """
    
    case_id: str = Field(..., description="Case identifier")
    
    patient_info: dict = Field(
        default_factory=dict,
        description="Patient demographics and history"
    )
    
    tool_outputs: dict = Field(
        default_factory=dict,
        description="Outputs from MCP tools (gait features, quality, risk)"
    )
    
    knowledge_chunks: List[dict] = Field(
        default_factory=list,
        description="Retrieved RAG knowledge chunks"
    )
    
    def format_for_prompt(self) -> str:
        """Format context for inclusion in agent prompt."""
        lines = [
            f"## Case ID: {self.case_id}",
            "",
            "## Patient Information",
        ]
        
        for key, value in self.patient_info.items():
            lines.append(f"- {key}: {value}")
        
        lines.extend(["", "## MCP Tool Outputs"])
        
        for tool_name, output in self.tool_outputs.items():
            lines.append(f"\n### {tool_name}")
            if isinstance(output, dict):
                for k, v in output.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(str(output))
        
        if self.knowledge_chunks:
            lines.extend(["", "## Relevant Medical Knowledge"])
            for i, chunk in enumerate(self.knowledge_chunks[:5], 1):
                source = chunk.get("source_title", "Unknown")
                content = chunk.get("content", "")[:300]
                lines.append(f"\n### Reference {i} ({source})")
                lines.append(content)
        
        return "\n".join(lines)
