"""
Clinical War Room - Debate Engine

Phase 4: Multi-agent debate for clinical decision support.
"""

from debate.schemas import (
    Critique,
    Vote,
    VoteChoice,
    AgentPosition,
    DebateResult,
    DebateRound,
)
from debate.critique import CritiqueGenerator, AGENT_BIASES
from debate.revision import RevisionHandler
from debate.voting import VotingHandler
from debate.scoring import (
    compute_disagreement_score,
    compute_variance,
    compute_vote_entropy,
    summarize_disagreement,
)
from debate.orchestrator import DebateOrchestrator


__all__ = [
    # Schemas
    "Critique",
    "Vote",
    "VoteChoice",
    "AgentPosition",
    "DebateResult",
    "DebateRound",
    # Handlers
    "CritiqueGenerator",
    "RevisionHandler",
    "VotingHandler",
    # Scoring
    "compute_disagreement_score",
    "compute_variance",
    "compute_vote_entropy",
    "summarize_disagreement",
    # Orchestrator
    "DebateOrchestrator",
    # Constants
    "AGENT_BIASES",
]
