"""
Clinical War Room - Critique Generator

Generates cross-agent critiques using LLM reasoning.
"""

from typing import List, Dict, Any, Optional
import json

from agents.schemas import AgentOutput
from agents.llm_client import LLMClient, get_llm_client
from debate.schemas import Critique
from core.logging import logger


CRITIQUE_SYSTEM_PROMPT = """You are a debate moderator in a clinical decision support system.

Your role is to generate CRITIQUES of agent opinions to expose:
1. LOGIC FLAWS - reasoning that doesn't follow from evidence
2. OVERCONFIDENCE - confidence higher than evidence supports
3. MISSING EVIDENCE - claims made without supporting data
4. BIAS - unacknowledged assumptions or preferences

## CRITIC AGENT: {critic_agent}
The agent generating this critique has the following bias: {critic_bias}

## TARGET AGENT: {target_agent}
The agent being critiqued has the following output:
- Claim: {target_claim}
- Confidence: {target_confidence:.0%}
- Risk: {target_risk:.0%}
- Evidence count: {evidence_count}
- Reasoning: {target_reasoning}

## YOUR TASK
Generate a critique from {critic_agent}'s perspective, challenging {target_agent}'s position.

The critique should:
- Be specific and actionable
- Reference the evidence (or lack thereof)
- Suggest how the target should revise their position

Output JSON:
{{
    "critique_type": "logic_flaw|overconfidence|missing_evidence|bias",
    "critique_text": "Your detailed critique...",
    "severity": "low|moderate|high",
    "suggested_adjustment": "What should target agent change?"
}}
"""


AGENT_BIASES = {
    "Diagnostic Agent": "optimistic (high sensitivity, favors detection)",
    "Risk Agent": "pessimistic (assumes worst case, high false positives ok)",
    "Data Quality Agent": "skeptical (questions data reliability)",
    "Ethics Agent": "safety-first (prioritizes patient welfare)",
    "Evidence Agent": "conservative (strictly evidence-based)",
}


class CritiqueGenerator:
    """
    Generates critiques between agents.
    
    Uses LLM to create meaningful critiques that expose
    weaknesses in agent reasoning.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self.log = logger.with_context(phase="critique")
    
    def generate_critique(
        self,
        critic_agent: str,
        target_output: AgentOutput,
    ) -> Critique:
        """
        Generate a critique from one agent about another.
        
        Args:
            critic_agent: Name of the agent doing the critiquing
            target_output: The AgentOutput being critiqued
            
        Returns:
            Critique object
        """
        critic_bias = AGENT_BIASES.get(critic_agent, "neutral")
        
        system_prompt = CRITIQUE_SYSTEM_PROMPT.format(
            critic_agent=critic_agent,
            critic_bias=critic_bias,
            target_agent=target_output.agent_name,
            target_claim=target_output.claim,
            target_confidence=target_output.confidence,
            target_risk=target_output.risk,
            evidence_count=len(target_output.evidence),
            target_reasoning=target_output.reasoning[:500],
        )
        
        user_prompt = f"Generate critique of {target_output.agent_name}'s position."
        
        # Use fallback if LLM is not available
        if not self.llm.is_available:
            return self._fallback_critique(critic_agent, target_output)
        
        parsed_json, raw_content, error = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        if error or parsed_json is None:
            self.log.warning(f"Critique generation failed: {error}")
            return self._fallback_critique(critic_agent, target_output)
        
        return Critique(
            critic_agent=critic_agent,
            target_agent=target_output.agent_name,
            critique_type=parsed_json.get("critique_type", "general"),
            critique_text=parsed_json.get("critique_text", "No critique generated"),
            severity=parsed_json.get("severity", "moderate"),
            suggested_adjustment=parsed_json.get("suggested_adjustment"),
        )
    
    def _fallback_critique(
        self,
        critic_agent: str,
        target_output: AgentOutput,
    ) -> Critique:
        """Generate fallback critique when LLM fails."""
        # Deterministic critique based on agent types
        critique_map = {
            ("Risk Agent", "Diagnostic Agent"): (
                "overconfidence",
                "Diagnostic Agent may be underestimating cumulative risk factors. "
                "The optimistic bias toward detection does not adequately account "
                "for worst-case scenarios.",
                "Recommend increasing risk assessment by 15-20%.",
            ),
            ("Data Quality Agent", "Risk Agent"): (
                "missing_evidence",
                "Risk Agent's high risk assessment may not be fully supported "
                "by the data quality. Conclusions should be tempered by data reliability.",
                "Consider noting data quality limitations in risk assessment.",
            ),
            ("Evidence Agent", "Diagnostic Agent"): (
                "overconfidence",
                "The diagnostic claim lacks specific literature citations. "
                "Confidence should be reduced without published threshold references.",
                "Add specific guideline references to support claims.",
            ),
            ("Ethics Agent", "Risk Agent"): (
                "bias",
                "Risk Agent's pessimistic framing may create unnecessary anxiety. "
                "Patient communication should balance risk with actionable guidance.",
                "Include recommendations for patient communication.",
            ),
            ("Diagnostic Agent", "Evidence Agent"): (
                "logic_flaw",
                "Evidence Agent's conservative stance may delay necessary intervention. "
                "Early detection requires accepting some uncertainty.",
                "Consider clinical urgency alongside evidence thresholds.",
            ),
        }
        
        key = (critic_agent, target_output.agent_name)
        if key in critique_map:
            ctype, text, adjustment = critique_map[key]
        else:
            ctype = "general"
            text = f"{critic_agent} questions {target_output.agent_name}'s confidence level of {target_output.confidence:.0%}."
            adjustment = "Consider revising confidence based on available evidence."
        
        return Critique(
            critic_agent=critic_agent,
            target_agent=target_output.agent_name,
            critique_type=ctype,
            critique_text=text,
            severity="moderate",
            suggested_adjustment=adjustment,
        )
    
    def generate_all_critiques(
        self,
        opinions: Dict[str, AgentOutput],
        critique_pairs: Optional[List[tuple]] = None,
    ) -> List[Critique]:
        """
        Generate critiques for multiple agent pairs.
        
        Args:
            opinions: Dict mapping agent name to AgentOutput
            critique_pairs: Optional list of (critic, target) pairs.
                           If None, uses default pairing.
        
        Returns:
            List of Critique objects
        """
        if critique_pairs is None:
            # Default critique pairs for meaningful disagreement
            critique_pairs = [
                ("Risk Agent", "Diagnostic Agent"),
                ("Data Quality Agent", "Risk Agent"),
                ("Evidence Agent", "Diagnostic Agent"),
                ("Ethics Agent", "Risk Agent"),
                ("Diagnostic Agent", "Evidence Agent"),
            ]
        
        critiques = []
        for critic, target in critique_pairs:
            if critic in opinions and target in opinions:
                target_output = opinions[target]
                critique = self.generate_critique(critic, target_output)
                critiques.append(critique)
                self.log.info(f"Generated critique: {critic} → {target}")
        
        return critiques
