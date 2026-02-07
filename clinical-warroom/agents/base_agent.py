"""
Clinical War Room - Base Agent

Abstract base class for all specialist agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import json

from agents.schemas import AgentOutput, CaseContext, EvidenceItem, Concern, RiskLevel
from agents.llm_client import LLMClient, get_llm_client
from core.logging import logger


class BaseAgent(ABC):
    """
    Abstract base class for specialist agents.
    
    Design principles:
    - Agents analyze cases INDEPENDENTLY
    - Agents use LLM for reasoning ONLY
    - Agents NEVER compute numeric features (use MCP tools)
    - Agents NEVER override MCP tool outputs
    - All outputs must conform to AgentOutput schema
    """
    
    # Must be set by subclasses
    agent_name: str = "BaseAgent"
    agent_role: str = "Base specialist agent"
    agent_bias: str = "neutral"
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.log = logger.with_context(agent_name=self.agent_name)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        
        Must explicitly encode the agent's bias and role.
        """
        pass
    
    def build_user_prompt(self, context: CaseContext) -> str:
        """
        Build the user prompt from case context.
        
        Override for agent-specific formatting.
        """
        return f"""
Analyze the following clinical case and provide your assessment.

{context.format_for_prompt()}

---

Provide your response as a JSON object with this exact structure:
{{
    "agent_name": "{self.agent_name}",
    "claim": "Your primary claim or assessment (be specific)",
    "confidence": 0.0-1.0,
    "risk": 0.0-1.0,
    "evidence": [
        {{"source": "...", "content": "...", "relevance": 0.0-1.0}}
    ],
    "concerns": [
        {{"description": "...", "severity": "low|moderate|high|critical", "mitigation": "..."}}
    ],
    "reasoning": "Your detailed reasoning",
    "veto": false
}}

IMPORTANT:
- Base your assessment on the MCP tool outputs and RAG knowledge provided
- Do NOT invent or compute numeric values - use ONLY what is given
- Explain your reasoning clearly
"""
    
    def run(self, context: CaseContext) -> AgentOutput:
        """
        Run the agent on a case.
        
        Args:
            context: Case context with patient info, tool outputs, knowledge
            
        Returns:
            Validated AgentOutput
        """
        self.log.info(f"Running agent on case {context.case_id}")
        
        system_prompt = self.system_prompt
        user_prompt = self.build_user_prompt(context)
        
        # Generate LLM response
        parsed_json, raw_content, error = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        if error or parsed_json is None:
            self.log.error(f"LLM generation failed: {error}")
            return self._fallback_output(context, error or "Unknown error")
        
        # Validate and convert to AgentOutput
        try:
            output = self._parse_output(parsed_json)
            self.log.info(f"Agent output: {output.summary()}")
            return output
        except Exception as e:
            self.log.error(f"Output parsing failed: {e}")
            return self._fallback_output(context, str(e))
    
    def _parse_output(self, json_data: Dict[str, Any]) -> AgentOutput:
        """Parse JSON response into AgentOutput."""
        # Ensure agent_name is set
        json_data["agent_name"] = json_data.get("agent_name", self.agent_name)
        
        # Parse evidence items
        evidence = []
        for e in json_data.get("evidence", []):
            if isinstance(e, dict):
                evidence.append(EvidenceItem(
                    source=e.get("source", "unknown"),
                    content=e.get("content", ""),
                    relevance=float(e.get("relevance", 1.0)),
                ))
        
        # Parse concerns
        concerns = []
        for c in json_data.get("concerns", []):
            if isinstance(c, dict):
                concerns.append(Concern(
                    description=c.get("description", ""),
                    severity=c.get("severity", "moderate"),
                    mitigation=c.get("mitigation"),
                ))
        
        # Determine risk level
        risk = float(json_data.get("risk", 0.5))
        if risk >= 0.75:
            risk_level = RiskLevel.CRITICAL
        elif risk >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk >= 0.25:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return AgentOutput(
            agent_name=json_data["agent_name"],
            claim=json_data.get("claim", "No claim provided"),
            confidence=float(json_data.get("confidence", 0.5)),
            risk=risk,
            risk_level=risk_level,
            evidence=evidence,
            concerns=concerns,
            reasoning=json_data.get("reasoning", ""),
            veto=bool(json_data.get("veto", False)),
            veto_reason=json_data.get("veto_reason"),
        )
    
    def _fallback_output(self, context: CaseContext, error: str) -> AgentOutput:
        """Generate fallback output when LLM fails."""
        return AgentOutput(
            agent_name=self.agent_name,
            claim=f"Unable to complete analysis: {error}",
            confidence=0.0,
            risk=0.5,
            risk_level=RiskLevel.UNKNOWN,
            evidence=[],
            concerns=[
                Concern(
                    description=f"Agent analysis failed: {error}",
                    severity="high",
                    mitigation="Manual review required",
                )
            ],
            reasoning="Analysis could not be completed due to an error.",
            veto=False,
        )
