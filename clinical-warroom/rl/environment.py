"""
Clinical War Room - RL Environment

Simulated environment for training the RL coordinator.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import random

from rl.state import RLState, discretize_state


class Action(IntEnum):
    """Action space for RL agent."""
    EXECUTE = 0
    ESCALATE = 1
    REFUSE = 2
    REQUEST_MORE_DATA = 3


ACTION_NAMES = {
    Action.EXECUTE: "EXECUTE",
    Action.ESCALATE: "ESCALATE",
    Action.REFUSE: "REFUSE",
    Action.REQUEST_MORE_DATA: "REQUEST_MORE_DATA",
}


@dataclass
class Outcome:
    """
    Simulated outcome for training.
    
    In a real system, this would come from human feedback.
    """
    was_safe: bool
    was_correct: bool
    efficiency_score: float  # 0-1, higher is more efficient
    
    def to_dict(self) -> dict:
        return {
            "was_safe": self.was_safe,
            "was_correct": self.was_correct,
            "efficiency_score": self.efficiency_score,
        }


class ClinicalEnvironment:
    """
    Simulated environment for RL training.
    
    Generates synthetic states and outcomes for offline training.
    Does NOT interact with real patients or systems.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.current_state: Optional[RLState] = None
        self.ground_truth_risk: float = 0.5
    
    def reset(self) -> RLState:
        """Generate a new random state."""
        # Sample ground truth risk (hidden from agent)
        self.ground_truth_risk = self.rng.uniform(0.0, 1.0)
        
        # Add noise to create observed measurements
        noise = 0.15
        
        # Generate state with noise around ground truth
        avg_conf = np.clip(self.rng.normal(0.6, 0.2), 0.1, 0.95)
        min_conf = np.clip(avg_conf - self.rng.uniform(0.1, 0.3), 0.05, avg_conf)
        max_conf = np.clip(avg_conf + self.rng.uniform(0.1, 0.3), avg_conf, 0.99)
        
        observed_risk = np.clip(
            self.ground_truth_risk + self.rng.normal(0, noise), 0.0, 1.0
        )
        
        # Disagreement correlates with uncertainty
        disagreement = np.clip(self.rng.uniform(0.1, 0.5) + noise, 0.0, 1.0)
        
        # Data quality
        data_quality = np.clip(self.rng.uniform(0.5, 1.0), 0.0, 1.0)
        
        # Vote distribution (slightly correlated with risk)
        if observed_risk > 0.7:
            votes = [0.2, 0.5, 0.2, 0.1]  # More escalate
        elif observed_risk < 0.3:
            votes = [0.6, 0.2, 0.1, 0.1]  # More execute
        else:
            votes = [0.4, 0.3, 0.15, 0.15]  # Mixed
        
        # Add noise to votes
        votes = np.clip(np.array(votes) + self.rng.normal(0, 0.1, 4), 0, 1)
        votes = votes / votes.sum()
        
        # Veto based on extreme risk
        has_veto = 1.0 if self.ground_truth_risk > 0.9 else 0.0
        
        self.current_state = RLState(
            avg_confidence=float(avg_conf),
            min_confidence=float(min_conf),
            max_confidence=float(max_conf),
            avg_risk=float(observed_risk),
            max_risk=float(np.clip(observed_risk + 0.1, 0, 1)),
            disagreement_score=float(disagreement),
            data_quality_score=float(data_quality),
            vote_execute=float(votes[0]),
            vote_escalate=float(votes[1]),
            vote_refuse=float(votes[2]),
            vote_request_data=float(votes[3]),
            has_veto=float(has_veto),
        )
        
        return self.current_state
    
    def step(self, action: Action) -> Tuple[Outcome, float]:
        """
        Execute action and get outcome + reward.
        
        Returns:
            (outcome, reward)
        """
        if self.current_state is None:
            raise ValueError("Must call reset() before step()")
        
        outcome = self._simulate_outcome(action)
        reward = self._compute_reward(action, outcome)
        
        return outcome, reward
    
    def _simulate_outcome(self, action: Action) -> Outcome:
        """
        Simulate outcome based on action and ground truth.
        
        This is a simplified model for training purposes.
        """
        risk = self.ground_truth_risk
        
        if action == Action.EXECUTE:
            # Executing is only safe if risk is low
            was_safe = risk < 0.5
            was_correct = risk < 0.4  # Optimal threshold
            efficiency = 1.0 if was_correct else 0.2
            
        elif action == Action.ESCALATE:
            # Escalating is always safe but may be inefficient
            was_safe = True
            was_correct = 0.4 <= risk <= 0.75  # Good for medium risk
            efficiency = 0.6 if was_correct else 0.4
            
        elif action == Action.REFUSE:
            # Refusing is safe but should only be for high risk
            was_safe = True
            was_correct = risk > 0.7
            efficiency = 0.5 if was_correct else 0.3
            
        else:  # REQUEST_MORE_DATA
            # Requesting more data is safe but delays decision
            was_safe = True
            data_quality = self.current_state.data_quality_score
            was_correct = data_quality < 0.6  # Good if data quality was low
            efficiency = 0.4 if was_correct else 0.2
        
        return Outcome(
            was_safe=was_safe,
            was_correct=was_correct,
            efficiency_score=efficiency,
        )
    
    def _compute_reward(self, action: Action, outcome: Outcome) -> float:
        """
        Compute reward for RL training.
        
        Reward structure:
        - Large negative for unsafe actions
        - Positive for correct decisions
        - Small penalty for inefficiency
        """
        if not outcome.was_safe:
            return -10.0  # Large penalty for unsafe
        
        if outcome.was_correct:
            return 1.0 + outcome.efficiency_score  # Up to +2.0
        else:
            return -0.5 + outcome.efficiency_score * 0.5  # Small penalty
    
    def sample_batch(
        self,
        batch_size: int = 100,
    ) -> list:
        """
        Generate batch of (state, best_action) pairs.
        
        Used for supervised pretraining or imitation learning.
        """
        batch = []
        
        for _ in range(batch_size):
            state = self.reset()
            risk = self.ground_truth_risk
            
            # Optimal policy based on ground truth
            if risk > 0.75:
                best_action = Action.REFUSE
            elif risk > 0.5:
                best_action = Action.ESCALATE
            elif self.current_state.data_quality_score < 0.5:
                best_action = Action.REQUEST_MORE_DATA
            else:
                best_action = Action.EXECUTE
            
            batch.append((state, best_action))
        
        return batch
