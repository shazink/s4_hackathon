"""
Clinical War Room - RL Coordinator

Integrates RL policy with safety rules for final decision.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os

from rl.state import RLState
from rl.policy import QLearningPolicy, train_policy
from rl.environment import Action, ACTION_NAMES
from safety.evaluator import SafetyOutput
from safety.rules import ForcedAction
from core.logging import logger


@dataclass
class CoordinatorOutput:
    """
    Output from the RL coordinator.
    """
    selected_action: str
    policy_confidence: float
    explanation: str
    was_overridden: bool = False
    override_rule: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "selected_action": self.selected_action,
            "policy_confidence": self.policy_confidence,
            "explanation": self.explanation,
            "was_overridden": self.was_overridden,
            "override_rule": self.override_rule,
        }


class RLCoordinator:
    """
    RL-based decision coordinator.
    
    CRITICAL CONSTRAINTS:
    - NEVER overrides safety rules
    - If forced_action exists, that action is used
    - RL only chooses among allowed actions
    - No access to raw patient data
    - No MCP tool calls
    - No LLM calls
    """
    
    # Default path to trained policy
    DEFAULT_POLICY_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "models", "rl_policy.pkl"
    )
    
    def __init__(
        self,
        policy: Optional[QLearningPolicy] = None,
        policy_path: Optional[str] = None,
    ):
        self.log = logger.with_context(phase="rl_coordinator")
        
        if policy is not None:
            self.policy = policy
        else:
            # Try default trained policy first
            default_path = policy_path or self.DEFAULT_POLICY_PATH
            if os.path.exists(default_path):
                self.policy = QLearningPolicy()
                self.policy.load(default_path)
                self.log.info(f"Loaded trained policy from {default_path}")
            else:
                # Fall back to training a new policy
                self.log.info("No trained policy found, training default...")
                self.policy = train_policy(episodes=500)
    
    def decide(
        self,
        rl_state: RLState,
        safety_output: SafetyOutput,
    ) -> CoordinatorOutput:
        """
        Make a decision based on state and safety constraints.
        
        Args:
            rl_state: Encoded state from debate
            safety_output: Output from safety evaluator
            
        Returns:
            CoordinatorOutput with selected action
        """
        # RULE 1: If safety forced an action, use it
        if not safety_output.allowed and safety_output.forced_action:
            action_name = safety_output.forced_action.value
            
            self.log.info(f"Safety override: {action_name}")
            
            return CoordinatorOutput(
                selected_action=action_name,
                policy_confidence=1.0,  # Safety rules are absolute
                explanation=(
                    f"Action forced by safety rule: {safety_output.triggered_rule.name}. "
                    f"{safety_output.explanation}"
                ),
                was_overridden=True,
                override_rule=safety_output.triggered_rule.rule_id,
            )
        
        # RULE 2: RL chooses among allowed actions
        action, confidence = self.policy.select_action(rl_state, explore=False)
        action_name = ACTION_NAMES[action]
        
        explanation = self._generate_explanation(rl_state, action, confidence)
        
        self.log.info(f"RL selected: {action_name} (confidence: {confidence:.0%})")
        
        return CoordinatorOutput(
            selected_action=action_name,
            policy_confidence=confidence,
            explanation=explanation,
            was_overridden=False,
        )
    
    def _generate_explanation(
        self,
        state: RLState,
        action: Action,
        confidence: float,
    ) -> str:
        """Generate human-readable explanation for action."""
        reasons = []
        
        if action == Action.EXECUTE:
            reasons.append("Risk levels are within acceptable bounds")
            if state.avg_confidence > 0.6:
                reasons.append(f"Average confidence is high ({state.avg_confidence:.0%})")
            if state.vote_execute > 0.4:
                reasons.append(f"Majority of agents voted to execute")
                
        elif action == Action.ESCALATE:
            reasons.append("Case complexity warrants human review")
            if state.disagreement_score > 0.4:
                reasons.append(f"Significant disagreement among agents ({state.disagreement_score:.0%})")
            if state.max_risk > 0.5:
                reasons.append(f"Elevated risk detected ({state.max_risk:.0%})")
                
        elif action == Action.REFUSE:
            reasons.append("Case is too high-risk for automated processing")
            if state.max_risk > 0.7:
                reasons.append(f"Maximum risk score is very high ({state.max_risk:.0%})")
            if state.min_confidence < 0.4:
                reasons.append(f"Low agent confidence ({state.min_confidence:.0%})")
                
        else:  # REQUEST_MORE_DATA
            reasons.append("Additional data needed for reliable assessment")
            if state.data_quality_score < 0.7:
                reasons.append(f"Data quality is suboptimal ({state.data_quality_score:.0%})")
        
        if not reasons:
            reasons = ["Policy determined this action is optimal for current state"]
        
        return " ".join(reasons) + f" Policy confidence: {confidence:.0%}."


def create_coordinator(
    policy_path: Optional[str] = None,
    train_episodes: int = 500,
) -> RLCoordinator:
    """
    Factory function to create a coordinator.
    
    Args:
        policy_path: Path to saved policy (optional)
        train_episodes: Episodes to train if no saved policy
        
    Returns:
        Configured RLCoordinator
    """
    if policy_path and os.path.exists(policy_path):
        return RLCoordinator(policy_path=policy_path)
    
    # Train new policy
    policy = train_policy(episodes=train_episodes)
    
    # Optionally save
    if policy_path:
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        policy.save(policy_path)
    
    return RLCoordinator(policy=policy)
