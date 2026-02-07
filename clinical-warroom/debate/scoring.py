"""
Clinical War Room - Disagreement Scoring

Computes disagreement metrics for debates.
"""

from typing import List, Dict, Any
import math

from debate.schemas import Vote, VoteChoice, AgentPosition


def compute_variance(values: List[float]) -> float:
    """Compute variance of a list of values."""
    if not values or len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance


def compute_vote_entropy(votes: List[Vote]) -> float:
    """
    Compute entropy of vote distribution.
    
    Higher entropy = more disagreement.
    Returns value between 0 and 1.
    """
    if not votes:
        return 0.0
    
    # Count vote choices
    vote_counts: Dict[VoteChoice, int] = {}
    for vote in votes:
        choice = vote.vote_choice
        vote_counts[choice] = vote_counts.get(choice, 0) + 1
    
    # Compute probabilities
    n = len(votes)
    probabilities = [count / n for count in vote_counts.values()]
    
    # Compute entropy (base 2)
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by max entropy (log2 of number of vote options)
    max_entropy = math.log2(len(VoteChoice))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


def compute_disagreement_score(
    initial_positions: List[AgentPosition],
    revised_positions: List[AgentPosition],
    votes: List[Vote],
) -> float:
    """
    Compute overall disagreement score for a debate.
    
    Score is between 0.0 (complete consensus) and 1.0 (complete disagreement).
    
    Components:
    - Confidence variance (25%)
    - Risk variance (25%)
    - Vote entropy (40%)
    - Ethics concerns (10%)
    
    Args:
        initial_positions: Agent positions before revision
        revised_positions: Agent positions after revision
        votes: Final votes from all agents
        
    Returns:
        Disagreement score between 0 and 1
    """
    scores = {}
    
    # 1. Confidence variance (25%)
    confidences = [p.confidence for p in revised_positions]
    confidence_variance = compute_variance(confidences)
    # Normalize: max variance is 0.25 (when values are 0 and 1)
    scores["confidence"] = min(1.0, confidence_variance / 0.25)
    
    # 2. Risk variance (25%)
    risks = [p.risk for p in revised_positions]
    risk_variance = compute_variance(risks)
    scores["risk"] = min(1.0, risk_variance / 0.25)
    
    # 3. Vote entropy (40%)
    scores["vote"] = compute_vote_entropy(votes)
    
    # 4. Ethics/veto presence (10%)
    has_veto = any(p.has_veto for p in revised_positions)
    has_high_risk = any(p.risk > 0.7 for p in revised_positions)
    scores["ethics"] = 1.0 if has_veto else (0.5 if has_high_risk else 0.0)
    
    # Weighted combination
    disagreement = (
        0.25 * scores["confidence"] +
        0.25 * scores["risk"] +
        0.40 * scores["vote"] +
        0.10 * scores["ethics"]
    )
    
    return min(1.0, max(0.0, disagreement))


def compute_revision_delta(
    initial: AgentPosition,
    revised: AgentPosition,
) -> Dict[str, float]:
    """
    Compute how much an agent revised their position.
    
    Returns:
        Dict with confidence and risk deltas
    """
    return {
        "confidence_delta": revised.confidence - initial.confidence,
        "risk_delta": revised.risk - initial.risk,
        "added_concerns": revised.concerns_count - initial.concerns_count,
    }


def summarize_disagreement(
    disagreement_score: float,
    votes: List[Vote],
) -> str:
    """Generate human-readable disagreement summary."""
    if disagreement_score < 0.2:
        level = "LOW"
        desc = "Agents largely agree on this case."
    elif disagreement_score < 0.4:
        level = "MODERATE"
        desc = "Some disagreement exists, but majority consensus reached."
    elif disagreement_score < 0.6:
        level = "ELEVATED"
        desc = "Significant disagreement among agents."
    elif disagreement_score < 0.8:
        level = "HIGH"
        desc = "Major disagreement, human review strongly recommended."
    else:
        level = "CRITICAL"
        desc = "Extreme disagreement, case likely out of scope."
    
    # Vote breakdown
    counts = {}
    for vote in votes:
        choice = vote.vote_choice.value
        counts[choice] = counts.get(choice, 0) + 1
    
    vote_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
    
    return f"Disagreement: {level} ({disagreement_score:.0%}). {desc} Votes: {vote_str}"
