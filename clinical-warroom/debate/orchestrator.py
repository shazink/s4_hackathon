"""
Clinical War Room - Debate Orchestrator

Orchestrates the 4-round debate between specialist agents.
"""

from typing import List, Dict, Any, Optional
import json

from agents.schemas import AgentOutput, CaseContext
from debate.schemas import (
    Critique, Vote, VoteChoice, AgentPosition, DebateResult
)
from debate.critique import CritiqueGenerator
from debate.revision import RevisionHandler
from debate.voting import VotingHandler
from debate.scoring import (
    compute_disagreement_score, summarize_disagreement
)
from core.logging import logger


class DebateOrchestrator:
    """
    Orchestrates the Clinical War Room debate.
    
    FOUR ROUNDS:
    1. Opinion Collection - Gather agent positions
    2. Cross-Critique - Agents critique each other
    3. Revision - Agents revise based on critiques
    4. Voting - Agents vote on recommended action
    
    RULES:
    - Does NOT call MCP tools
    - Does NOT enforce rules
    - Does NOT make final decisions
    - Agent identities remain distinct
    """
    
    def __init__(
        self,
        critique_generator: Optional[CritiqueGenerator] = None,
        revision_handler: Optional[RevisionHandler] = None,
        voting_handler: Optional[VotingHandler] = None,
    ):
        self.critique_gen = critique_generator or CritiqueGenerator()
        self.revision = revision_handler or RevisionHandler()
        self.voting = voting_handler or VotingHandler()
        self.log = logger.with_context(phase="debate")
    
    def run_debate(
        self,
        case_id: str,
        agent_outputs: Dict[str, AgentOutput],
        case_summary: str = "",
    ) -> DebateResult:
        """
        Run a complete 4-round debate.
        
        Args:
            case_id: Identifier for the case
            agent_outputs: Dict mapping agent name to AgentOutput
            case_summary: Optional summary of the case for voting context
            
        Returns:
            DebateResult with all rounds, votes, and metrics
        """
        self.log.info(f"Starting debate on case {case_id} with {len(agent_outputs)} agents")
        
        # Initialize result
        result = DebateResult(
            case_id=case_id,
            total_agents=len(agent_outputs),
        )
        
        # ROUND 1: Opinion Collection
        self.log.info("ROUND 1: Opinion Collection")
        initial_positions = self._round1_collect_opinions(agent_outputs)
        result.initial_opinions = [p.to_dict() for p in initial_positions]
        result.rounds_completed = 1
        
        # ROUND 2: Cross-Critique
        self.log.info("ROUND 2: Cross-Critique")
        critiques = self._round2_critique(agent_outputs)
        result.critiques = [c.to_dict() for c in critiques]
        result.rounds_completed = 2
        
        # ROUND 3: Revision
        self.log.info("ROUND 3: Revision")
        revised_outputs = self._round3_revision(agent_outputs, critiques)
        revised_positions = [
            AgentPosition.from_agent_output(o) for o in revised_outputs.values()
        ]
        result.revised_opinions = [p.to_dict() for p in revised_positions]
        result.rounds_completed = 3
        
        # ROUND 4: Voting
        self.log.info("ROUND 4: Voting")
        votes = self._round4_voting(revised_outputs, case_summary)
        result.votes = [v.to_dict() for v in votes]
        result.rounds_completed = 4
        
        # Compute metrics
        result.disagreement_score = compute_disagreement_score(
            initial_positions, revised_positions, votes
        )
        
        # Check for veto
        for agent_name, output in revised_outputs.items():
            if output.veto:
                result.has_veto = True
                result.veto_agent = agent_name
                break
        
        self.log.info(f"Debate complete: {result.summary()}")
        return result
    
    def _round1_collect_opinions(
        self,
        agent_outputs: Dict[str, AgentOutput],
    ) -> List[AgentPosition]:
        """
        ROUND 1: Collect structured outputs as initial board positions.
        """
        positions = []
        for agent_name, output in agent_outputs.items():
            position = AgentPosition.from_agent_output(output)
            positions.append(position)
            self.log.debug(
                f"  {agent_name}: confidence={output.confidence:.0%}, "
                f"risk={output.risk:.0%}"
            )
        return positions
    
    def _round2_critique(
        self,
        agent_outputs: Dict[str, AgentOutput],
    ) -> List[Critique]:
        """
        ROUND 2: Each agent critiques at least one other agent.
        
        Uses predefined critique pairs for meaningful disagreement.
        """
        return self.critique_gen.generate_all_critiques(agent_outputs)
    
    def _round3_revision(
        self,
        agent_outputs: Dict[str, AgentOutput],
        critiques: List[Critique],
    ) -> Dict[str, AgentOutput]:
        """
        ROUND 3: Agents revise positions based on critiques.
        """
        return self.revision.revise_all(agent_outputs, critiques)
    
    def _round4_voting(
        self,
        revised_outputs: Dict[str, AgentOutput],
        case_summary: str,
    ) -> List[Vote]:
        """
        ROUND 4: Agents cast votes on recommended action.
        """
        return self.voting.collect_all_votes(revised_outputs, case_summary)
    
    def format_debate_report(self, result: DebateResult) -> str:
        """
        Format a human-readable debate report.
        """
        lines = [
            "=" * 70,
            f"  DEBATE REPORT: {result.case_id}",
            "=" * 70,
            "",
        ]
        
        # Initial Opinions
        lines.append("ROUND 1: INITIAL OPINIONS")
        lines.append("-" * 40)
        for opinion in result.initial_opinions:
            lines.append(
                f"  {opinion['agent_name']:20s} | "
                f"Conf: {opinion['confidence']:.0%} | "
                f"Risk: {opinion['risk']:.0%}"
            )
        lines.append("")
        
        # Critiques
        lines.append("ROUND 2: CROSS-CRITIQUES")
        lines.append("-" * 40)
        for critique in result.critiques:
            lines.append(
                f"  {critique['critic_agent']} → {critique['target_agent']}:"
            )
            lines.append(f"    [{critique['critique_type']}] {critique['critique_text'][:80]}...")
        lines.append("")
        
        # Revised Opinions
        lines.append("ROUND 3: REVISED OPINIONS")
        lines.append("-" * 40)
        for i, revised in enumerate(result.revised_opinions):
            initial = result.initial_opinions[i]
            conf_delta = revised['confidence'] - initial['confidence']
            delta_str = f"{conf_delta:+.0%}" if conf_delta != 0 else "unchanged"
            lines.append(
                f"  {revised['agent_name']:20s} | "
                f"Conf: {revised['confidence']:.0%} ({delta_str}) | "
                f"Risk: {revised['risk']:.0%}"
            )
        lines.append("")
        
        # Votes
        lines.append("ROUND 4: VOTES")
        lines.append("-" * 40)
        for vote in result.votes:
            lines.append(
                f"  {vote['agent_name']:20s} | "
                f"{vote['vote_choice']:15s} | "
                f"Conf: {vote['confidence']:.0%}"
            )
            if vote['reasoning']:
                lines.append(f"    → {vote['reasoning'][:60]}...")
        lines.append("")
        
        # Summary
        lines.append("DEBATE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Disagreement Score: {result.disagreement_score:.0%}")
        
        # Vote breakdown
        vote_counts = {}
        for v in result.votes:
            choice = v['vote_choice']
            vote_counts[choice] = vote_counts.get(choice, 0) + 1
        lines.append(f"  Vote Distribution: {vote_counts}")
        
        if result.has_veto:
            lines.append(f"  ⚠️  VETO by {result.veto_agent}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
