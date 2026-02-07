"""
Clinical War Room - Case Processor

End-to-end processing pipeline for patient cases.
Connects the UI input to the full agent + debate + safety + RL + HITL flow.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from core.logging import logger
from core.mcp.tools import compute_fall_risk, compute_gait_features
from agents.coordinator import AgentCoordinator
from debate.orchestrator import DebateOrchestrator
from safety.rules import SafetyRuleEngine
from rl.coordinator import RLCoordinator


@dataclass
class CaseInput:
    """Input data for a patient case."""
    patient_age: int
    patient_gender: str
    medical_history: str
    stride_length: float
    gait_speed: float
    symmetry_index: float
    chief_complaint: str


@dataclass
class ProcessedCase:
    """Fully processed case with all results."""
    case_id: str
    patient_summary: str
    mcp_metrics: Dict[str, Any]
    agent_opinions: List[Dict[str, Any]]
    debate: Dict[str, Any]
    disagreement_score: float
    safety: Dict[str, Any]
    decision: Dict[str, Any]
    pending_review: bool
    timestamp: str


class CaseProcessor:
    """
    End-to-end case processing pipeline.
    
    Flow:
    1. Compute MCP metrics (deterministic)
    2. Run specialist agents
    3. Run debate
    4. Check safety rules
    5. RL coordinator decision
    6. Determine if HITL review needed
    """
    
    def __init__(self):
        self.agent_coordinator = AgentCoordinator()
        self.debate_orchestrator = DebateOrchestrator()
        self.safety_engine = SafetyRuleEngine()
        self.rl_coordinator = RLCoordinator()
    
    def process(self, case_input: CaseInput) -> ProcessedCase:
        """
        Process a patient case through the full pipeline.
        
        Args:
            case_input: Patient case input data
            
        Returns:
            ProcessedCase with all analysis results
        """
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        logger.info(f"Processing case {case_id}", phase="case_processor")
        
        # Step 1: Build patient summary
        patient_summary = self._build_summary(case_input)
        
        # Step 2: Compute MCP metrics (deterministic, no AI)
        mcp_metrics = self._compute_metrics(case_input)
        logger.info(f"MCP metrics computed: fall_risk={mcp_metrics.get('fall_risk_score', 0):.0%}", phase="mcp")
        
        # Step 3: Run specialist agents
        agent_opinions = self.agent_coordinator.analyze(
            case_id=case_id,
            patient_summary=patient_summary,
            mcp_metrics=mcp_metrics,
        )
        logger.info(f"Got opinions from {len(agent_opinions)} agents", phase="agents")
        
        # Step 4: Run debate
        debate_result = self.debate_orchestrator.run_debate(agent_opinions)
        disagreement_score = debate_result.get('disagreement_score', 0)
        logger.info(f"Debate complete: disagreement={disagreement_score:.0%}", phase="debate")
        
        # Step 5: Safety rules
        safety_result = self.safety_engine.evaluate(
            mcp_metrics=mcp_metrics,
            agent_opinions=agent_opinions,
            debate_result=debate_result,
        )
        logger.info(f"Safety check: allowed={safety_result.allowed}", phase="safety")
        
        # Step 6: RL coordinator decision
        rl_decision = self.rl_coordinator.decide(
            mcp_metrics=mcp_metrics,
            agent_opinions=agent_opinions,
            debate_result=debate_result,
            safety_result=safety_result,
        )
        
        # Build decision info
        decision = {
            "action": rl_decision.action,
            "confidence": rl_decision.confidence,
            "explanation": rl_decision.explanation,
            "was_forced": rl_decision.was_overridden,
        }
        
        # Check if review needed
        pending_review = (
            decision["action"] == "ESCALATE"
            or decision["confidence"] < 0.6
            or not safety_result.allowed
        )
        
        # Format safety info
        safety_info = {
            "allowed": safety_result.allowed,
            "triggered_rule": safety_result.triggered_rule or "",
            "explanation": safety_result.explanation,
        }
        
        # Format agent opinions for UI
        formatted_agents = self._format_agents(agent_opinions)
        
        # Format debate for UI
        formatted_debate = self._format_debate(debate_result)
        
        return ProcessedCase(
            case_id=case_id,
            patient_summary=patient_summary,
            mcp_metrics=mcp_metrics,
            agent_opinions=formatted_agents,
            debate=formatted_debate,
            disagreement_score=disagreement_score,
            safety=safety_info,
            decision=decision,
            pending_review=pending_review,
            timestamp=datetime.now().isoformat(),
        )
    
    def _build_summary(self, case_input: CaseInput) -> str:
        """Build a patient summary string."""
        return (
            f"{case_input.patient_age}-year-old {case_input.patient_gender.lower()} "
            f"presenting for {case_input.chief_complaint.lower()}. "
            f"Medical history: {case_input.medical_history or 'No significant history reported'}."
        )
    
    def _compute_metrics(self, case_input: CaseInput) -> Dict[str, Any]:
        """Compute MCP metrics from input."""
        # Use MCP tools
        gait_features = compute_gait_features(
            stride_length=case_input.stride_length,
            gait_speed=case_input.gait_speed,
            cadence=case_input.gait_speed / case_input.stride_length * 60 if case_input.stride_length > 0 else 100,
        )
        
        fall_risk = compute_fall_risk(
            gait_variability=1.0 - case_input.symmetry_index,
            stride_length=case_input.stride_length,
            balance_score=case_input.symmetry_index,
            age=case_input.patient_age,
        )
        
        return {
            "stride_length": case_input.stride_length,
            "gait_speed": case_input.gait_speed,
            "symmetry_index": case_input.symmetry_index,
            "fall_risk_score": fall_risk.risk_score,
            **gait_features,
        }
    
    def _format_agents(self, opinions: List[Any]) -> List[Dict[str, Any]]:
        """Format agent opinions for UI display."""
        formatted = []
        for op in opinions:
            formatted.append({
                "agent_name": op.agent_name if hasattr(op, 'agent_name') else op.get('agent_name', 'Unknown'),
                "claim": op.claim if hasattr(op, 'claim') else op.get('claim', ''),
                "confidence": op.confidence if hasattr(op, 'confidence') else op.get('confidence', 0.5),
                "risk": op.risk_assessment if hasattr(op, 'risk_assessment') else op.get('risk', 0.5),
                "evidence_count": len(op.evidence) if hasattr(op, 'evidence') else op.get('evidence_count', 0),
                "concerns_count": len(op.concerns) if hasattr(op, 'concerns') else op.get('concerns_count', 0),
                "veto": op.veto if hasattr(op, 'veto') else op.get('veto', False),
            })
        return formatted
    
    def _format_debate(self, debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format debate result for UI display."""
        return {
            "critiques": debate_result.get('critiques', []),
            "revisions": debate_result.get('revisions', []),
            "votes": debate_result.get('votes', []),
            "rounds": debate_result.get('rounds', 4),
        }


# Helper function for quick processing
def process_case(
    patient_age: int,
    patient_gender: str,
    medical_history: str,
    stride_length: float,
    gait_speed: float,
    symmetry_index: float,
    chief_complaint: str,
) -> Dict[str, Any]:
    """
    Quick helper to process a case and return dict format.
    """
    processor = CaseProcessor()
    
    case_input = CaseInput(
        patient_age=patient_age,
        patient_gender=patient_gender,
        medical_history=medical_history,
        stride_length=stride_length,
        gait_speed=gait_speed,
        symmetry_index=symmetry_index,
        chief_complaint=chief_complaint,
    )
    
    result = processor.process(case_input)
    return asdict(result)
