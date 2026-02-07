"""
Clinical War Room - Revision Handler

Handles agent revisions after receiving critiques.
"""

from typing import List, Dict, Any, Optional
import json

from agents.schemas import AgentOutput, EvidenceItem, Concern
from agents.llm_client import LLMClient, get_llm_client
from debate.schemas import Critique, AgentPosition
from core.logging import logger


REVISION_SYSTEM_PROMPT = """You are an agent in a clinical decision support system who has received critiques of your position.

## YOUR AGENT: {agent_name}

## YOUR ORIGINAL POSITION
- Claim: {original_claim}
- Confidence: {original_confidence:.0%}
- Risk: {original_risk:.0%}
- Reasoning: {original_reasoning}

## CRITIQUES RECEIVED
{critiques_text}

## YOUR TASK
Respond to the critiques. You may:
1. DEFEND your position if critiques are unfounded
2. REVISE your confidence downward if critiques are valid
3. ADD CONCERNS you may have overlooked
4. ACKNOWLEDGE limitations

Be specific about what you're changing and why.

Output JSON:
{{
    "revised_confidence": 0.0-1.0,
    "revised_risk": 0.0-1.0,
    "revision_note": "Explanation of changes made",
    "new_concerns": ["any new concerns to add"],
    "defense": "Any defense of your original position"
}}
"""


class RevisionHandler:
    """
    Handles agent response to critiques.
    
    Agents can defend, revise, or add concerns based on critiques.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self.log = logger.with_context(phase="revision")
    
    def revise_opinion(
        self,
        original_output: AgentOutput,
        critiques: List[Critique],
    ) -> AgentOutput:
        """
        Revise an agent's opinion based on critiques received.
        
        Args:
            original_output: The original AgentOutput
            critiques: List of critiques targeting this agent
            
        Returns:
            Revised AgentOutput
        """
        if not critiques:
            return original_output
        
        critiques_text = "\n".join([
            f"From {c.critic_agent} ({c.critique_type}, {c.severity}):\n"
            f"  {c.critique_text}\n"
            f"  Suggestion: {c.suggested_adjustment or 'None'}"
            for c in critiques
        ])
        
        system_prompt = REVISION_SYSTEM_PROMPT.format(
            agent_name=original_output.agent_name,
            original_claim=original_output.claim,
            original_confidence=original_output.confidence,
            original_risk=original_output.risk,
            original_reasoning=original_output.reasoning[:500],
            critiques_text=critiques_text,
        )
        
        user_prompt = "Respond to the critiques and provide your revised position."
        
        # Use fallback if LLM is not available
        if not self.llm.is_available:
            return self._fallback_revision(original_output, critiques)
        
        parsed_json, raw_content, error = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        if error or parsed_json is None:
            self.log.warning(f"Revision generation failed: {error}")
            return self._fallback_revision(original_output, critiques)
        
        return self._apply_revision(original_output, parsed_json, critiques)
    
    def _apply_revision(
        self,
        original: AgentOutput,
        revision_data: Dict[str, Any],
        critiques: List[Critique],
    ) -> AgentOutput:
        """Apply revision data to create revised output."""
        # Get revised values with bounds
        revised_confidence = min(1.0, max(0.0, 
            revision_data.get("revised_confidence", original.confidence)
        ))
        revised_risk = min(1.0, max(0.0,
            revision_data.get("revised_risk", original.risk)
        ))
        
        # Add new concerns
        new_concerns = list(original.concerns)
        for concern_text in revision_data.get("new_concerns", []):
            if concern_text:
                new_concerns.append(Concern(
                    description=concern_text,
                    severity="moderate",
                ))
        
        # Build revised reasoning
        revision_note = revision_data.get("revision_note", "")
        defense = revision_data.get("defense", "")
        
        revised_reasoning = original.reasoning
        if revision_note:
            revised_reasoning += f"\n\n[REVISED] {revision_note}"
        if defense:
            revised_reasoning += f"\n\n[DEFENSE] {defense}"
        
        # Create revised output
        return AgentOutput(
            agent_name=original.agent_name,
            claim=original.claim,
            confidence=revised_confidence,
            risk=revised_risk,
            evidence=original.evidence,
            concerns=new_concerns,
            reasoning=revised_reasoning,
            veto=original.veto,
            veto_reason=original.veto_reason,
        )
    
    def _fallback_revision(
        self,
        original: AgentOutput,
        critiques: List[Critique],
    ) -> AgentOutput:
        """
        Deterministic fallback revision when LLM fails.
        
        Applies conservative adjustments based on critique severity.
        """
        confidence_adjustment = 0.0
        risk_adjustment = 0.0
        new_concerns = list(original.concerns)
        
        for critique in critiques:
            # Adjust based on critique type and severity
            severity_factor = {"low": 0.05, "moderate": 0.10, "high": 0.15}.get(
                critique.severity, 0.10
            )
            
            if critique.critique_type == "overconfidence":
                confidence_adjustment -= severity_factor
            elif critique.critique_type == "missing_evidence":
                confidence_adjustment -= severity_factor * 0.5
            elif critique.critique_type == "logic_flaw":
                confidence_adjustment -= severity_factor
                risk_adjustment += severity_factor * 0.5
            elif critique.critique_type == "bias":
                confidence_adjustment -= severity_factor * 0.5
            
            # Add concern for each critique
            new_concerns.append(Concern(
                description=f"Critique from {critique.critic_agent}: {critique.critique_text[:100]}...",
                severity=critique.severity,
            ))
        
        revised_confidence = max(0.1, original.confidence + confidence_adjustment)
        revised_risk = min(1.0, original.risk + risk_adjustment)
        
        revision_note = f"Revised based on {len(critiques)} critiques."
        
        return AgentOutput(
            agent_name=original.agent_name,
            claim=original.claim,
            confidence=revised_confidence,
            risk=revised_risk,
            evidence=original.evidence,
            concerns=new_concerns,
            reasoning=original.reasoning + f"\n\n[REVISED] {revision_note}",
            veto=original.veto,
            veto_reason=original.veto_reason,
        )
    
    def revise_all(
        self,
        opinions: Dict[str, AgentOutput],
        critiques: List[Critique],
    ) -> Dict[str, AgentOutput]:
        """
        Revise all agent opinions based on critiques.
        
        Args:
            opinions: Dict mapping agent name to AgentOutput
            critiques: All critiques from the critique round
            
        Returns:
            Dict mapping agent name to revised AgentOutput
        """
        # Group critiques by target
        critiques_by_target: Dict[str, List[Critique]] = {}
        for critique in critiques:
            target = critique.target_agent
            if target not in critiques_by_target:
                critiques_by_target[target] = []
            critiques_by_target[target].append(critique)
        
        # Revise each opinion
        revised = {}
        for agent_name, output in opinions.items():
            agent_critiques = critiques_by_target.get(agent_name, [])
            revised[agent_name] = self.revise_opinion(output, agent_critiques)
            
            if agent_critiques:
                self.log.info(
                    f"Revised {agent_name}: "
                    f"confidence {output.confidence:.0%} → {revised[agent_name].confidence:.0%}"
                )
        
        return revised
