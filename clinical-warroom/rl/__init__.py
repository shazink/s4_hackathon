"""
Clinical War Room - RL Coordinator Module

Phase 6: Reinforcement learning-based decision coordinator.
"""

from rl.state import RLState, discretize_state
from rl.environment import Action, ACTION_NAMES, Outcome, ClinicalEnvironment
from rl.policy import QLearningPolicy, train_policy
from rl.coordinator import RLCoordinator, CoordinatorOutput, create_coordinator


__all__ = [
    # State
    "RLState",
    "discretize_state",
    # Environment
    "Action",
    "ACTION_NAMES",
    "Outcome",
    "ClinicalEnvironment",
    # Policy
    "QLearningPolicy",
    "train_policy",
    # Coordinator
    "RLCoordinator",
    "CoordinatorOutput",
    "create_coordinator",
]
