"""
Clinical War Room - HITL Coordinator

Coordinates human-in-the-loop review flow.
"""

from datetime import timedelta
from typing import Optional, Dict, Any

from hitl.models import (
    CaseSummary,
    ReviewRequest,
    HumanDecision,
    SystemAction,
)
from hitl.store import ReviewStore, get_review_store
from safety.evaluator import SafetyOutput
from rl.coordinator import CoordinatorOutput
from core.logging import logger


class HITLCoordinator:
    """
    Coordinates human review process.
    
    Determines when review is required and manages
    the review lifecycle.
    """
    
    # Thresholds for mandatory review
    CONFIDENCE_THRESHOLD = 0.60
    
    def __init__(self, store: Optional[ReviewStore] = None):
        self.store = store or get_review_store()
        self.log = logger.with_context(phase="hitl")
    
    def needs_review(
        self,
        safety_output: SafetyOutput,
        rl_output: CoordinatorOutput,
    ) -> tuple:
        """
        Determine if human review is required.
        
        Returns:
            (is_mandatory, reason)
        """
        reasons = []
        is_mandatory = False
        
        # Check 1: Safety forced ESCALATE
        if (
            not safety_output.allowed
            and safety_output.forced_action
            and safety_output.forced_action.value == "ESCALATE"
        ):
            reasons.append("Safety rules require escalation")
            is_mandatory = True
        
        # Check 2: RL chose ESCALATE
        if rl_output.selected_action == "ESCALATE":
            reasons.append("System recommends escalation to human")
            is_mandatory = True
        
        # Check 3: Low confidence
        if rl_output.policy_confidence < self.CONFIDENCE_THRESHOLD:
            reasons.append(
                f"System confidence ({rl_output.policy_confidence:.0%}) "
                f"below threshold ({self.CONFIDENCE_THRESHOLD:.0%})"
            )
            is_mandatory = True
        
        # Check 4: Ethics concerns (from safety input)
        if (
            safety_output.input_metrics
            and safety_output.input_metrics.get("has_ethics_veto")
        ):
            reasons.append("Ethics agent raised concerns")
            is_mandatory = True
        
        if reasons:
            return is_mandatory, "; ".join(reasons)
        
        return False, ""
    
    def create_review_request(
        self,
        case_id: str,
        patient_summary: str,
        agent_opinions: list,
        debate_summary: str,
        disagreement_score: float,
        safety_output: SafetyOutput,
        rl_output: CoordinatorOutput,
        mcp_metrics: Optional[Dict[str, Any]] = None,
        expires_in: Optional[timedelta] = None,
    ) -> ReviewRequest:
        """
        Create a review request for human review.
        """
        is_mandatory, reason = self.needs_review(safety_output, rl_output)
        
        # Build case summary
        case_summary = CaseSummary(
            case_id=case_id,
            patient_summary=patient_summary,
            mcp_metrics=mcp_metrics or {},
            agent_opinions=agent_opinions,
            debate_summary=debate_summary,
            disagreement_score=disagreement_score,
            safety_allowed=safety_output.allowed,
            safety_explanation=safety_output.explanation,
            triggered_rule=(
                safety_output.triggered_rule.rule_id
                if safety_output.triggered_rule else None
            ),
            system_action=SystemAction(rl_output.selected_action),
            rl_confidence=rl_output.policy_confidence,
            rl_explanation=rl_output.explanation,
        )
        
        return self.store.create_review(
            case_id=case_id,
            case_summary=case_summary,
            review_reason=reason or "Routine review requested",
            is_mandatory=is_mandatory,
            expires_in=expires_in,
        )
    
    def get_final_action(
        self,
        case_id: str,
        rl_output: CoordinatorOutput,
    ) -> tuple:
        """
        Get the final action for a case.
        
        Returns:
            (final_action, was_overridden, decision_id)
        """
        decision = self.store.get_decision_for_case(case_id)
        
        if decision:
            # Human decision overrides everything
            return (
                decision.final_action,
                decision.human_action.value != "APPROVE",
                decision.decision_id,
            )
        
        # No human decision yet - return system recommendation
        return rl_output.selected_action, False, None
