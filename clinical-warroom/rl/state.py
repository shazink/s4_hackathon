"""
Clinical War Room - RL State Encoding

Encodes debate/safety outputs into a fixed-size numeric state vector.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from safety.evaluator import SafetyInput


@dataclass
class RLState:
    """
    Fixed-size numeric state for RL agent.
    
    All values normalized to [0, 1] range.
    """
    # Confidence metrics
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    
    # Risk metrics
    avg_risk: float
    max_risk: float
    
    # Debate metrics
    disagreement_score: float
    
    # Data quality
    data_quality_score: float
    
    # Vote distribution (normalized counts)
    vote_execute: float
    vote_escalate: float
    vote_refuse: float
    vote_request_data: float
    
    # Ethics
    has_veto: float  # 0 or 1
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size numpy array."""
        return np.array([
            self.avg_confidence,
            self.min_confidence,
            self.max_confidence,
            self.avg_risk,
            self.max_risk,
            self.disagreement_score,
            self.data_quality_score,
            self.vote_execute,
            self.vote_escalate,
            self.vote_refuse,
            self.vote_request_data,
            self.has_veto,
        ], dtype=np.float32)
    
    @classmethod
    def state_dim(cls) -> int:
        """Dimension of state vector."""
        return 12
    
    def to_dict(self) -> dict:
        return {
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "avg_risk": self.avg_risk,
            "max_risk": self.max_risk,
            "disagreement_score": self.disagreement_score,
            "data_quality_score": self.data_quality_score,
            "vote_execute": self.vote_execute,
            "vote_escalate": self.vote_escalate,
            "vote_refuse": self.vote_refuse,
            "vote_request_data": self.vote_request_data,
            "has_veto": self.has_veto,
        }
    
    @classmethod
    def from_debate_result(
        cls,
        debate_result: Dict[str, Any],
        data_quality: float = 1.0,
    ) -> "RLState":
        """
        Create RLState from debate output.
        
        Args:
            debate_result: DebateResult.to_dict() output
            data_quality: Data quality score from MCP tools
        """
        # Extract opinions
        opinions = debate_result.get("revised_opinions", [])
        if not opinions:
            opinions = debate_result.get("initial_opinions", [])
        
        confidences = [op.get("confidence", 0.5) for op in opinions]
        risks = [op.get("risk", 0.5) for op in opinions]
        
        avg_conf = np.mean(confidences) if confidences else 0.5
        min_conf = min(confidences) if confidences else 0.5
        max_conf = max(confidences) if confidences else 0.5
        avg_risk = np.mean(risks) if risks else 0.5
        max_risk = max(risks) if risks else 0.5
        
        # Vote distribution
        votes = debate_result.get("votes", [])
        vote_counts = {"execute": 0, "escalate": 0, "refuse": 0, "request_more_data": 0}
        
        for vote in votes:
            choice = vote.get("vote_choice", "").lower()
            if choice in vote_counts:
                vote_counts[choice] += 1
        
        total_votes = max(len(votes), 1)
        
        return cls(
            avg_confidence=float(avg_conf),
            min_confidence=float(min_conf),
            max_confidence=float(max_conf),
            avg_risk=float(avg_risk),
            max_risk=float(max_risk),
            disagreement_score=debate_result.get("disagreement_score", 0.0),
            data_quality_score=data_quality,
            vote_execute=vote_counts["execute"] / total_votes,
            vote_escalate=vote_counts["escalate"] / total_votes,
            vote_refuse=vote_counts["refuse"] / total_votes,
            vote_request_data=vote_counts["request_more_data"] / total_votes,
            has_veto=1.0 if debate_result.get("has_veto", False) else 0.0,
        )


def discretize_state(state: RLState, bins: int = 5) -> tuple:
    """
    Discretize state for tabular Q-learning.
    
    Args:
        state: RLState to discretize
        bins: Number of bins per dimension
        
    Returns:
        Tuple of discretized values
    """
    vector = state.to_vector()
    # Clip to [0, 1] and discretize
    clipped = np.clip(vector, 0.0, 1.0)
    discretized = tuple((clipped * (bins - 1)).astype(int))
    return discretized
