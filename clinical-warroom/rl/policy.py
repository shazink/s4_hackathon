"""
Clinical War Room - Q-Learning Policy

Tabular Q-learning for action selection.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pickle
import os

from rl.state import RLState, discretize_state
from rl.environment import Action, ACTION_NAMES


class QLearningPolicy:
    """
    Tabular Q-learning policy.
    
    Uses discretized state space for tractable learning.
    Deterministic at inference time.
    """
    
    def __init__(
        self,
        state_bins: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.state_bins = state_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.rng = np.random.RandomState(seed)
        
        # Q-table: maps discretized state to action values
        self.q_table: Dict[tuple, np.ndarray] = {}
        
        # Training stats
        self.training_episodes = 0
        self.total_reward = 0.0
    
    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self.q_table:
            # Initialize with small random values
            self.q_table[state_key] = self.rng.uniform(-0.1, 0.1, len(Action))
        return self.q_table[state_key]
    
    def select_action(
        self,
        state: RLState,
        explore: bool = False,
    ) -> Tuple[Action, float]:
        """
        Select action based on current policy.
        
        Args:
            state: RLState to act on
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            (action, confidence)
        """
        state_key = discretize_state(state, self.state_bins)
        q_values = self._get_q_values(state_key)
        
        if explore and self.rng.random() < self.epsilon:
            # Random exploration
            action = Action(self.rng.randint(len(Action)))
            confidence = 0.5
        else:
            # Greedy action
            action = Action(np.argmax(q_values))
            
            # Confidence based on Q-value margin
            sorted_q = np.sort(q_values)[::-1]
            if len(sorted_q) > 1 and sorted_q[0] > sorted_q[1]:
                margin = sorted_q[0] - sorted_q[1]
                confidence = min(1.0, 0.5 + margin * 0.2)
            else:
                confidence = 0.5
        
        return action, confidence
    
    def update(
        self,
        state: RLState,
        action: Action,
        reward: float,
        next_state: Optional[RLState] = None,
    ):
        """
        Update Q-values with training sample.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None for terminal)
        """
        state_key = discretize_state(state, self.state_bins)
        q_values = self._get_q_values(state_key)
        
        if next_state is not None:
            next_key = discretize_state(next_state, self.state_bins)
            next_q = self._get_q_values(next_key)
            target = reward + self.gamma * np.max(next_q)
        else:
            target = reward
        
        # Q-learning update
        q_values[action] += self.lr * (target - q_values[action])
        
        self.total_reward += reward
        self.training_episodes += 1
    
    def train_batch(
        self,
        transitions: list,
    ):
        """
        Train on a batch of (state, action, reward) tuples.
        
        For offline training without next states.
        """
        for state, action, reward in transitions:
            self.update(state, action, reward, None)
    
    def save(self, path: str):
        """Save policy to file."""
        data = {
            "q_table": self.q_table,
            "state_bins": self.state_bins,
            "training_episodes": self.training_episodes,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load policy from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.state_bins = data["state_bins"]
        self.training_episodes = data["training_episodes"]
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "training_episodes": self.training_episodes,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.training_episodes),
            "q_table_size": len(self.q_table),
        }


def train_policy(
    episodes: int = 1000,
    seed: Optional[int] = 42,
) -> QLearningPolicy:
    """
    Train a Q-learning policy using simulated environment.
    
    Args:
        episodes: Number of training episodes
        seed: Random seed for reproducibility
        
    Returns:
        Trained policy
    """
    from rl.environment import ClinicalEnvironment
    
    env = ClinicalEnvironment(seed=seed)
    policy = QLearningPolicy(seed=seed)
    
    for _ in range(episodes):
        state = env.reset()
        action, _ = policy.select_action(state, explore=True)
        outcome, reward = env.step(action)
        policy.update(state, action, reward, None)
    
    return policy
