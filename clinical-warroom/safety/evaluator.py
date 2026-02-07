"""
Clinical War Room - Safety Rule Evaluator

Evaluates debate output against safety rules.
This is the final authority before any learning-based logic.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json

from safety.rules import (
    SafetyThresholds,
    SafetyRule,
    ForcedAction,
    SAFETY_RULES,
    get_rules_by_priority,
)
from core.logging import logger


@dataclass
class SafetyInput:
    """
    Input to the safety evaluator.
    
    Extracted from debate output and agent metrics.
    """
    # From debate
    has_ethics_veto: bool
    disagreement_score: float
    
    # From agent opinions
    max_risk_score: float
    min_confidence_score: float
    
    # From tool outputs
    data_quality_score: float
    
    # Optional: which agent vetoed
    veto_agent: Optional[str] = None
    
    # Raw debate result for reference
    debate_votes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "has_ethics_veto": self.has_ethics_veto,
            "veto_agent": self.veto_agent,
            "disagreement_score": self.disagreement_score,
            "max_risk_score": self.max_risk_score,
            "min_confidence_score": self.min_confidence_score,
            "data_quality_score": self.data_quality_score,
        }
    
    @classmethod
    def from_debate_result(cls, debate_result: dict, data_quality: float = 1.0) -> "SafetyInput":
        """
        Create SafetyInput from a DebateResult dict.
        
        Args:
            debate_result: DebateResult.to_dict() output
            data_quality: Data quality score from MCP tools
        """
        # Extract max risk and min confidence from revised opinions
        revised = debate_result.get("revised_opinions", [])
        if not revised:
            revised = debate_result.get("initial_opinions", [])
        
        risks = [op.get("risk", 0.5) for op in revised]
        confidences = [op.get("confidence", 0.5) for op in revised]
        
        max_risk = max(risks) if risks else 0.5
        min_confidence = min(confidences) if confidences else 0.5
        
        return cls(
            has_ethics_veto=debate_result.get("has_veto", False),
            veto_agent=debate_result.get("veto_agent"),
            disagreement_score=debate_result.get("disagreement_score", 0.0),
            max_risk_score=max_risk,
            min_confidence_score=min_confidence,
            data_quality_score=data_quality,
            debate_votes=debate_result.get("votes", []),
        )


@dataclass
class SafetyOutput:
    """
    Output from the safety evaluator.
    
    Determines whether the case is allowed to proceed.
    """
    allowed: bool
    forced_action: Optional[ForcedAction]
    triggered_rule: Optional[SafetyRule]
    explanation: str
    
    # Additional context
    all_rules_checked: List[str] = field(default_factory=list)
    input_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "forced_action": self.forced_action.value if self.forced_action else None,
            "triggered_rule": self.triggered_rule.rule_id if self.triggered_rule else None,
            "triggered_rule_name": self.triggered_rule.name if self.triggered_rule else None,
            "explanation": self.explanation,
            "all_rules_checked": self.all_rules_checked,
            "input_metrics": self.input_metrics,
        }


class SafetyEvaluator:
    """
    Evaluates debate output against safety rules.
    
    Design principles:
    - DETERMINISTIC: Same input always produces same output
    - PRIORITY-BASED: Rules evaluated in order, first trigger wins
    - FINAL AUTHORITY: Cannot be overridden by RL or agents
    - EXPLAINABLE: Every decision includes clear explanation
    """
    
    def __init__(self, thresholds: Optional[SafetyThresholds] = None):
        self.thresholds = thresholds or SafetyThresholds()
        self.rules = get_rules_by_priority()
        self.log = logger.with_context(phase="safety")
    
    def evaluate(self, safety_input: SafetyInput) -> SafetyOutput:
        """
        Evaluate safety rules against input.
        
        Args:
            safety_input: Metrics from debate and tools
            
        Returns:
            SafetyOutput indicating if case can proceed
        """
        self.log.info("Evaluating safety rules...")
        
        checked_rules = []
        
        # Check rules in priority order
        for rule in self.rules:
            triggered, explanation = self._check_rule(rule, safety_input)
            checked_rules.append(rule.rule_id)
            
            if triggered:
                self.log.warning(
                    f"Safety rule triggered: {rule.name} → {rule.forced_action.value}"
                )
                
                return SafetyOutput(
                    allowed=False,
                    forced_action=rule.forced_action,
                    triggered_rule=rule,
                    explanation=explanation,
                    all_rules_checked=checked_rules,
                    input_metrics=safety_input.to_dict(),
                )
        
        # All rules passed
        self.log.info("All safety rules passed - case allowed to proceed")
        
        return SafetyOutput(
            allowed=True,
            forced_action=None,
            triggered_rule=None,
            explanation="All safety rules passed. Case is allowed to proceed with automated assessment.",
            all_rules_checked=checked_rules,
            input_metrics=safety_input.to_dict(),
        )
    
    def _check_rule(
        self,
        rule: SafetyRule,
        input_data: SafetyInput
    ) -> tuple:
        """
        Check a single rule against input.
        
        Returns:
            (triggered: bool, explanation: str)
        """
        if rule.rule_id == "RULE_001":
            return self._check_ethics_veto(input_data)
        
        elif rule.rule_id == "RULE_002":
            return self._check_extreme_risk(input_data)
        
        elif rule.rule_id == "RULE_003":
            return self._check_low_confidence(input_data)
        
        elif rule.rule_id == "RULE_004":
            return self._check_severe_disagreement(input_data)
        
        elif rule.rule_id == "RULE_005":
            return self._check_poor_data_quality(input_data)
        
        return False, ""
    
    def _check_ethics_veto(self, input_data: SafetyInput) -> tuple:
        """RULE 1: Ethics Veto"""
        if input_data.has_ethics_veto:
            agent = input_data.veto_agent or "an agent"
            return True, (
                f"ETHICS VETO: {agent} has issued a veto. "
                f"This case must be escalated to human review. "
                f"Ethics concerns cannot be overridden by automated systems."
            )
        return False, ""
    
    def _check_extreme_risk(self, input_data: SafetyInput) -> tuple:
        """RULE 2: Extreme Risk"""
        if input_data.max_risk_score >= self.thresholds.risk_high:
            return True, (
                f"EXTREME RISK: Maximum risk score ({input_data.max_risk_score:.0%}) "
                f"exceeds threshold ({self.thresholds.risk_high:.0%}). "
                f"High-risk cases require human clinical judgment. "
                f"Automated assessment is refused."
            )
        return False, ""
    
    def _check_low_confidence(self, input_data: SafetyInput) -> tuple:
        """RULE 3: Low Confidence"""
        if input_data.min_confidence_score <= self.thresholds.confidence_low:
            return True, (
                f"LOW CONFIDENCE: Minimum confidence ({input_data.min_confidence_score:.0%}) "
                f"is at or below threshold ({self.thresholds.confidence_low:.0%}). "
                f"Insufficient certainty for safe automated decision-making. "
                f"Automated assessment is refused."
            )
        return False, ""
    
    def _check_severe_disagreement(self, input_data: SafetyInput) -> tuple:
        """RULE 4: Severe Disagreement"""
        if input_data.disagreement_score >= self.thresholds.disagreement_high:
            return True, (
                f"SEVERE DISAGREEMENT: Disagreement score ({input_data.disagreement_score:.0%}) "
                f"exceeds threshold ({self.thresholds.disagreement_high:.0%}). "
                f"Significant disagreement indicates complexity requiring human judgment. "
                f"Case must be escalated to human review."
            )
        return False, ""
    
    def _check_poor_data_quality(self, input_data: SafetyInput) -> tuple:
        """RULE 5: Poor Data Quality"""
        if input_data.data_quality_score <= self.thresholds.data_quality_low:
            return True, (
                f"POOR DATA QUALITY: Data quality score ({input_data.data_quality_score:.0%}) "
                f"is at or below threshold ({self.thresholds.data_quality_low:.0%}). "
                f"Decisions based on poor quality data are unreliable. "
                f"Additional data collection is required before proceeding."
            )
        return False, ""
    
    def format_report(self, output: SafetyOutput) -> str:
        """Format a human-readable safety report."""
        lines = [
            "=" * 60,
            "  SAFETY EVALUATION REPORT",
            "=" * 60,
            "",
        ]
        
        if output.allowed:
            lines.append("✅ STATUS: ALLOWED")
            lines.append("")
            lines.append("All safety rules passed.")
        else:
            lines.append("⛔ STATUS: BLOCKED")
            lines.append("")
            lines.append(f"Triggered Rule: {output.triggered_rule.name}")
            lines.append(f"Forced Action:  {output.forced_action.value}")
            lines.append("")
            lines.append("Explanation:")
            lines.append(f"  {output.explanation}")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("Input Metrics:")
        
        if output.input_metrics:
            for key, value in output.input_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.0%}")
                elif isinstance(value, bool):
                    lines.append(f"  {key}: {'Yes' if value else 'No'}")
                elif value is not None:
                    lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append(f"Rules Checked: {', '.join(output.all_rules_checked)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
