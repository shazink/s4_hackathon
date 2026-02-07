"""
Clinical War Room - RL Trainer

Trains the Q-learning policy using generated training data.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from rl.state import RLState
from rl.environment import Action, ClinicalEnvironment, ACTION_NAMES
from rl.policy import QLearningPolicy
from rl.training_data import TrainingCase, generate_training_dataset, load_training_dataset


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    episodes: int = 5000
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    initial_exploration: float = 0.5
    final_exploration: float = 0.05
    exploration_decay: float = 0.995
    state_bins: int = 5
    seed: int = 42
    save_interval: int = 500


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring."""
    episode: int
    avg_reward: float
    exploration_rate: float
    q_table_size: int
    accuracy: float
    
    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "avg_reward": self.avg_reward,
            "exploration_rate": self.exploration_rate,
            "q_table_size": self.q_table_size,
            "accuracy": self.accuracy,
        }


class RLTrainer:
    """
    Trainer for the Q-learning policy.
    
    Supports:
    - Environment-based training (random states)
    - Dataset-based training (labeled cases)
    - Evaluation on held-out data
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.policy = QLearningPolicy(
            state_bins=self.config.state_bins,
            learning_rate=self.config.learning_rate,
            discount_factor=self.config.discount_factor,
            exploration_rate=self.config.initial_exploration,
            seed=self.config.seed,
        )
        self.env = ClinicalEnvironment(seed=self.config.seed)
        self.metrics_history: List[TrainingMetrics] = []
    
    def case_to_state(self, case: TrainingCase) -> RLState:
        """Convert a training case to an RLState."""
        # Estimate vote distribution from risk
        risk = case.ground_truth_risk
        if risk > 0.75:
            votes = [0.1, 0.3, 0.5, 0.1]
        elif risk > 0.5:
            votes = [0.2, 0.5, 0.2, 0.1]
        elif risk > 0.25:
            votes = [0.5, 0.3, 0.1, 0.1]
        else:
            votes = [0.7, 0.15, 0.05, 0.1]
        
        return RLState(
            avg_confidence=case.avg_confidence,
            min_confidence=case.min_confidence,
            max_confidence=case.max_confidence,
            avg_risk=case.fall_risk_score,
            max_risk=min(1.0, case.fall_risk_score + 0.1),
            disagreement_score=case.disagreement_score,
            data_quality_score=0.2 if case.stride_length == 0 else 0.9,
            vote_execute=votes[0],
            vote_escalate=votes[1],
            vote_refuse=votes[2],
            vote_request_data=votes[3],
            has_veto=1.0 if case.has_veto else 0.0,
        )
    
    def action_from_name(self, name: str) -> Action:
        """Convert action name to Action enum."""
        name_map = {
            "EXECUTE": Action.EXECUTE,
            "ESCALATE": Action.ESCALATE,
            "REFUSE": Action.REFUSE,
            "REQUEST_MORE_DATA": Action.REQUEST_MORE_DATA,
        }
        return name_map[name]
    
    def train_on_environment(self, episodes: Optional[int] = None) -> List[TrainingMetrics]:
        """
        Train using the simulated environment.
        
        Uses epsilon-greedy exploration with decay.
        """
        episodes = episodes or self.config.episodes
        epsilon = self.config.initial_exploration
        
        rewards = []
        
        for ep in range(episodes):
            # Reset environment
            state = self.env.reset()
            
            # Select action with exploration
            action, _ = self.policy.select_action(state, explore=True)
            
            # Get outcome and reward
            outcome, reward = self.env.step(action)
            rewards.append(reward)
            
            # Update policy
            self.policy.update(state, action, reward, None)
            
            # Decay exploration
            epsilon = max(
                self.config.final_exploration,
                epsilon * self.config.exploration_decay
            )
            self.policy.epsilon = epsilon
            
            # Record metrics periodically
            if (ep + 1) % 100 == 0:
                metrics = TrainingMetrics(
                    episode=ep + 1,
                    avg_reward=np.mean(rewards[-100:]),
                    exploration_rate=epsilon,
                    q_table_size=len(self.policy.q_table),
                    accuracy=self._evaluate_accuracy(),
                )
                self.metrics_history.append(metrics)
                print(f"Episode {ep+1}: avg_reward={metrics.avg_reward:.3f}, accuracy={metrics.accuracy:.2%}, q_size={metrics.q_table_size}")
        
        return self.metrics_history
    
    def train_on_dataset(self, cases: List[TrainingCase]) -> List[TrainingMetrics]:
        """
        Train using labeled training cases.
        
        This is imitation learning / supervised pretraining.
        """
        print(f"Training on {len(cases)} labeled cases...")
        
        # Multiple passes through the dataset
        for epoch in range(10):
            np.random.shuffle(cases)
            epoch_rewards = []
            
            for case in cases:
                state = self.case_to_state(case)
                optimal = self.action_from_name(case.optimal_action)
                
                # Compute reward for the correct action
                reward = case.reward_if_correct
                
                # Update policy with the correct action
                self.policy.update(state, optimal, reward, None)
                epoch_rewards.append(reward)
            
            # Record metrics
            metrics = TrainingMetrics(
                episode=(epoch + 1) * len(cases),
                avg_reward=np.mean(epoch_rewards),
                exploration_rate=self.policy.epsilon,
                q_table_size=len(self.policy.q_table),
                accuracy=self._evaluate_on_cases(cases),
            )
            self.metrics_history.append(metrics)
            print(f"Epoch {epoch+1}: avg_reward={metrics.avg_reward:.3f}, accuracy={metrics.accuracy:.2%}")
        
        return self.metrics_history
    
    def train_combined(
        self,
        cases: List[TrainingCase],
        env_episodes: int = 2000,
    ) -> List[TrainingMetrics]:
        """
        Combined training: supervised pretraining + RL fine-tuning.
        
        1. First, train on labeled cases (imitation learning)
        2. Then, fine-tune with environment exploration
        """
        print("=" * 60)
        print("Phase 1: Supervised Pretraining on Labeled Cases")
        print("=" * 60)
        self.train_on_dataset(cases)
        
        print("\n" + "=" * 60)
        print("Phase 2: RL Fine-tuning with Environment Exploration")
        print("=" * 60)
        self.train_on_environment(env_episodes)
        
        return self.metrics_history
    
    def _evaluate_accuracy(self, n_samples: int = 100) -> float:
        """Evaluate policy accuracy on random samples."""
        correct = 0
        
        for _ in range(n_samples):
            state = self.env.reset()
            action, _ = self.policy.select_action(state, explore=False)
            
            # Get optimal action based on ground truth
            risk = self.env.ground_truth_risk
            if risk > 0.75:
                optimal = Action.REFUSE
            elif risk > 0.5:
                optimal = Action.ESCALATE
            elif state.data_quality_score < 0.5:
                optimal = Action.REQUEST_MORE_DATA
            else:
                optimal = Action.EXECUTE
            
            if action == optimal:
                correct += 1
        
        return correct / n_samples
    
    def _evaluate_on_cases(self, cases: List[TrainingCase]) -> float:
        """Evaluate policy accuracy on labeled cases."""
        correct = 0
        
        for case in cases:
            state = self.case_to_state(case)
            action, _ = self.policy.select_action(state, explore=False)
            optimal = self.action_from_name(case.optimal_action)
            
            if action == optimal:
                correct += 1
        
        return correct / len(cases)
    
    def save_policy(self, path: str):
        """Save trained policy."""
        self.policy.save(path)
        print(f"Policy saved to: {path}")
    
    def save_metrics(self, path: str):
        """Save training metrics."""
        data = [m.to_dict() for m in self.metrics_history]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Metrics saved to: {path}")


def run_training(
    output_dir: Optional[str] = None,
    n_cases: int = 55,
    env_episodes: int = 3000,
    seed: int = 42,
):
    """
    Run the complete training pipeline.
    
    1. Generate training data
    2. Train policy with combined approach
    3. Evaluate and save results
    """
    output_dir = output_dir or str(Path(__file__).parent.parent / "data" / "models")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CLINICAL WAR ROOM - RL AGENT TRAINING")
    print("=" * 60)
    
    # Generate training data
    print("\n📊 Generating training dataset...")
    cases = generate_training_dataset(seed=seed)
    print(f"Generated {len(cases)} cases:")
    action_counts = {}
    for c in cases:
        action_counts[c.optimal_action] = action_counts.get(c.optimal_action, 0) + 1
    for action, count in sorted(action_counts.items()):
        print(f"   {action}: {count}")
    
    # Save training data
    data_path = Path(output_dir).parent / "training" / "training_cases.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    from rl.training_data import save_training_dataset
    save_training_dataset(cases, str(data_path))
    print(f"📁 Training data saved to: {data_path}")
    
    # Create trainer
    config = TrainingConfig(
        episodes=env_episodes,
        seed=seed,
    )
    trainer = RLTrainer(config)
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train_combined(cases, env_episodes)
    
    # Evaluate
    print("\n📈 Final Evaluation:")
    final_accuracy = trainer._evaluate_accuracy(n_samples=200)
    print(f"   Environment accuracy: {final_accuracy:.2%}")
    case_accuracy = trainer._evaluate_on_cases(cases)
    print(f"   Dataset accuracy: {case_accuracy:.2%}")
    
    # Save results
    policy_path = os.path.join(output_dir, "rl_policy.pkl")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    
    trainer.save_policy(policy_path)
    trainer.save_metrics(metrics_path)
    
    print("\n✅ Training complete!")
    print(f"   Policy: {policy_path}")
    print(f"   Metrics: {metrics_path}")
    
    return trainer


if __name__ == "__main__":
    run_training()
