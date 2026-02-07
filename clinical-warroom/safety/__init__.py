"""
Clinical War Room - Safety Layer

Phase 5: Rule-based safety system that enforces non-negotiable constraints.
"""

from safety.rules import (
    SafetyRule,
    SafetyThresholds,
    ForcedAction,
    SAFETY_RULES,
    get_rule_by_id,
    get_rules_by_priority,
)
from safety.evaluator import (
    SafetyInput,
    SafetyOutput,
    SafetyEvaluator,
)


__all__ = [
    # Rules
    "SafetyRule",
    "SafetyThresholds",
    "ForcedAction",
    "SAFETY_RULES",
    "get_rule_by_id",
    "get_rules_by_priority",
    # Evaluator
    "SafetyInput",
    "SafetyOutput",
    "SafetyEvaluator",
]
