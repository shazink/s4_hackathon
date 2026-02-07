"""
Clinical War Room - Voting Handler

Handles agent voting on recommended actions.
"""

from typing import List, Dict, Any, Optional
import json

from agents.schemas import AgentOutput
from agents.llm_client import LLMClient, get_llm_client
from debate.schemas import Vote, VoteChoice
from core.logging import logger


VOTING_SYSTEM_PROMPT = """You are a specialist agent voting on a clinical case.

## YOUR AGENT: {agent_name}

## YOUR FINAL POSITION
- Claim: {claim}
- Confidence: {confidence:.0%}
- Risk: {risk:.0%}
- Concerns: {concerns_count}

## CASE SUMMARY
{case_summary}

## VOTING OPTIONS
- EXECUTE: Proceed with automated assessment and recommendations
- ESCALATE: Escalate to human clinical review
- REFUSE: Refuse to proceed (serious safety concern)
- REQUEST_MORE_DATA: Need additional information before deciding

## YOUR TASK
Cast your vote based on your analysis and the debate.

Consider:
- Your confidence level
- The risk assessment
- Any safety concerns
- Data quality issues

Output JSON:
{{
    "vote": "execute|escalate|refuse|request_more_data",
    "confidence": 0.0-1.0,
    "reasoning": "Why you voted this way"
}}
"""


class VotingHandler:
    """
    Handles agent voting on recommended actions.
    
    Each agent casts a vote with confidence and reasoning.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self.log = logger.with_context(phase="voting")
    
    def cast_vote(
        self,
        agent_output: AgentOutput,
        case_summary: str = "",
    ) -> Vote:
        """
        Have an agent cast a vote.
        
        Args:
            agent_output: The agent's final position
            case_summary: Summary of the case
            
        Returns:
            Vote object
        """
        system_prompt = VOTING_SYSTEM_PROMPT.format(
            agent_name=agent_output.agent_name,
            claim=agent_output.claim,
            confidence=agent_output.confidence,
            risk=agent_output.risk,
            concerns_count=len(agent_output.concerns),
            case_summary=case_summary or "No summary available",
        )
        
        user_prompt = "Cast your vote."
        
        # Use fallback if LLM is not available
        if not self.llm.is_available:
            return self._fallback_vote(agent_output)
        
        parsed_json, raw_content, error = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        if error or parsed_json is None:
            self.log.warning(f"Vote generation failed: {error}")
            return self._fallback_vote(agent_output)
        
        # Parse vote choice
        vote_str = parsed_json.get("vote", "escalate").lower().replace("-", "_")
        try:
            vote_choice = VoteChoice(vote_str)
        except ValueError:
            vote_choice = VoteChoice.ESCALATE
        
        return Vote(
            agent_name=agent_output.agent_name,
            vote_choice=vote_choice,
            confidence=min(1.0, max(0.0, parsed_json.get("confidence", 0.5))),
            reasoning=parsed_json.get("reasoning", ""),
        )
    
    def _fallback_vote(self, agent_output: AgentOutput) -> Vote:
        """
        Deterministic fallback voting based on agent position.
        
        Uses agent type, confidence, and risk to determine vote.
        """
        # Veto always means refuse
        if agent_output.veto:
            return Vote(
                agent_name=agent_output.agent_name,
                vote_choice=VoteChoice.REFUSE,
                confidence=0.9,
                reasoning=f"Veto issued: {agent_output.veto_reason or 'Safety concern'}",
            )
        
        # Determine vote based on confidence and risk
        agent_name = agent_output.agent_name
        confidence = agent_output.confidence
        risk = agent_output.risk
        
        # Agent-specific voting logic
        if "Ethics" in agent_name:
            # Ethics agent is cautious
            if risk > 0.6 or confidence < 0.5:
                vote = VoteChoice.ESCALATE
                reasoning = "High risk or low confidence requires human review"
            else:
                vote = VoteChoice.EXECUTE
                reasoning = "Case appears safe for automated assessment"
                
        elif "Data Quality" in agent_name:
            # Data Quality agent cares about reliability
            if confidence < 0.6:
                vote = VoteChoice.REQUEST_MORE_DATA
                reasoning = "Data quality concerns warrant additional data collection"
            elif risk > 0.7:
                vote = VoteChoice.ESCALATE
                reasoning = "Risk level requires human verification"
            else:
                vote = VoteChoice.EXECUTE
                reasoning = "Data quality sufficient for automated assessment"
                
        elif "Risk" in agent_name:
            # Risk agent is pessimistic
            if risk > 0.5:
                vote = VoteChoice.ESCALATE
                reasoning = "Elevated risk requires human clinical judgment"
            else:
                vote = VoteChoice.EXECUTE
                reasoning = "Risk within acceptable automated thresholds"
                
        elif "Diagnostic" in agent_name:
            # Diagnostic agent favors action
            if confidence > 0.6:
                vote = VoteChoice.EXECUTE
                reasoning = "Sufficient confidence in diagnostic findings"
            else:
                vote = VoteChoice.ESCALATE
                reasoning = "Unclear findings require clinical review"
                
        elif "Evidence" in agent_name:
            # Evidence agent is conservative
            if confidence < 0.7:
                vote = VoteChoice.ESCALATE
                reasoning = "Insufficient evidence for automated decision"
            else:
                vote = VoteChoice.EXECUTE
                reasoning = "Findings consistent with established guidelines"
                
        else:
            # Default: escalate if uncertain
            if confidence < 0.6 or risk > 0.6:
                vote = VoteChoice.ESCALATE
                reasoning = "Default to human review for uncertain cases"
            else:
                vote = VoteChoice.EXECUTE
                reasoning = "Case within automated thresholds"
        
        return Vote(
            agent_name=agent_output.agent_name,
            vote_choice=vote,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def collect_all_votes(
        self,
        revised_opinions: Dict[str, AgentOutput],
        case_summary: str = "",
    ) -> List[Vote]:
        """
        Collect votes from all agents.
        
        Args:
            revised_opinions: Dict mapping agent name to revised AgentOutput
            case_summary: Summary of the case
            
        Returns:
            List of Vote objects
        """
        votes = []
        for agent_name, output in revised_opinions.items():
            vote = self.cast_vote(output, case_summary)
            votes.append(vote)
            self.log.info(f"{agent_name} votes: {vote.vote_choice.value}")
        
        return votes
