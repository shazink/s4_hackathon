"""
Clinical War Room - Safety Rules

Defines the mandatory safety rules that override all agent decisions.
These rules represent hospital policy and ethical constraints.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ForcedAction(str, Enum):
    """Actions that can be forced by safety rules."""
    REFUSE = "REFUSE"
    ESCALATE = "ESCALATE"
    REQUEST_MORE_DATA = "REQUEST_MORE_DATA"


@dataclass
class SafetyThresholds:
    """
    Configurable thresholds for safety rules.
    
    These can be adjusted per deployment but should be
    reviewed by clinical governance.
    """
    # Risk threshold (0-1): Above this, REFUSE
    risk_high: float = 0.85
    
    # Confidence threshold (0-1): Below this, REFUSE  
    confidence_low: float = 0.30
    
    # Disagreement threshold (0-1): Above this, ESCALATE
    disagreement_high: float = 0.60
    
    # Data quality threshold (0-1): Below this, REQUEST_MORE_DATA
    data_quality_low: float = 0.50
    
    def to_dict(self) -> dict:
        return {
            "risk_high": self.risk_high,
            "confidence_low": self.confidence_low,
            "disagreement_high": self.disagreement_high,
            "data_quality_low": self.data_quality_low,
        }


@dataclass
class SafetyRule:
    """Definition of a safety rule."""
    rule_id: str
    name: str
    priority: int  # Lower = higher priority
    description: str
    forced_action: ForcedAction
    
    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "description": self.description,
            "forced_action": self.forced_action.value,
        }


# Mandatory safety rules ordered by priority
SAFETY_RULES = [
    SafetyRule(
        rule_id="RULE_001",
        name="Ethics Veto",
        priority=1,
        description=(
            "If any agent issues a veto, the case MUST be escalated to human review. "
            "Ethics concerns cannot be overridden by any automated system."
        ),
        forced_action=ForcedAction.ESCALATE,
    ),
    SafetyRule(
        rule_id="RULE_002",
        name="Extreme Risk",
        priority=2,
        description=(
            "If the maximum risk score exceeds the high-risk threshold, "
            "the system MUST refuse to proceed with automated assessment. "
            "High-risk cases require human clinical judgment."
        ),
        forced_action=ForcedAction.REFUSE,
    ),
    SafetyRule(
        rule_id="RULE_003",
        name="Low Confidence",
        priority=3,
        description=(
            "If the minimum confidence score falls below the threshold, "
            "the system MUST refuse to proceed. Low confidence indicates "
            "insufficient certainty for safe automated decision-making."
        ),
        forced_action=ForcedAction.REFUSE,
    ),
    SafetyRule(
        rule_id="RULE_004",
        name="Severe Disagreement",
        priority=4,
        description=(
            "If the disagreement score among agents exceeds the threshold, "
            "the case MUST be escalated for human review. Significant "
            "disagreement indicates complexity requiring human judgment."
        ),
        forced_action=ForcedAction.ESCALATE,
    ),
    SafetyRule(
        rule_id="RULE_005",
        name="Poor Data Quality",
        priority=5,
        description=(
            "If data quality falls below the threshold, the system MUST "
            "request more data before proceeding. Decisions based on "
            "poor quality data are unreliable and potentially harmful."
        ),
        forced_action=ForcedAction.REQUEST_MORE_DATA,
    ),
]


def get_rule_by_id(rule_id: str) -> Optional[SafetyRule]:
    """Get a safety rule by its ID."""
    for rule in SAFETY_RULES:
        if rule.rule_id == rule_id:
            return rule
    return None


def get_rules_by_priority() -> list:
    """Get all rules sorted by priority (lowest number = highest priority)."""
    return sorted(SAFETY_RULES, key=lambda r: r.priority)
