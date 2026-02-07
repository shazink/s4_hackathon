"""
Clinical War Room - Chat Query Engine

Fully integrated engine using:
- Real LLM agents (Groq API)
- Trained RL policy for decision-making
- RAG for patient context retrieval
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid
import os

from agents.llm_client import get_llm_client, LLMClient
from rl.coordinator import RLCoordinator
from rl.state import RLState
from rl.environment import Action, ACTION_NAMES


@dataclass
class AgentMessage:
    """A message from an agent during deliberation."""
    agent_name: str
    message: str
    timestamp: str
    message_type: str  # "opinion", "critique", "revision", "vote"
    confidence: float = 0.0
    risk_score: float = 0.0
    target_agent: Optional[str] = None


@dataclass
class ChatResponse:
    """Response to a doctor's query."""
    query_id: str
    query: str
    patient_id: str
    patient_name: str
    
    # The final answer
    answer: str
    confidence: float
    risk_level: str  # "low", "moderate", "high", "critical"
    recommendation: str  # "EXECUTE", "ESCALATE", "REFUSE"
    
    # Agent deliberation
    agent_messages: List[AgentMessage] = field(default_factory=list)
    
    # Risk assessment summary
    risk_factors: List[str] = field(default_factory=list)
    
    # RL decision info
    rl_explanation: str = ""
    rl_was_overridden: bool = False
    
    timestamp: str = ""


# Agent system prompts
AGENT_PROMPTS = {
    "Diagnostic": """You are the Diagnostic Agent in a clinical decision support system.
Analyze the patient data and provide your clinical assessment.
Focus on: diagnosis, symptom analysis, clinical findings.
Respond in 2-3 sentences with your key observations.""",
    
    "Risk": """You are the Risk Assessment Agent in a clinical decision support system.
Analyze the patient's fall risk and safety factors.
Focus on: fall risk score, mobility issues, risk factors.
Respond in 2-3 sentences with your risk assessment.""",
    
    "Evidence": """You are the Evidence-Based Medicine Agent in a clinical decision support system.
Reference relevant clinical guidelines and research evidence.
Focus on: applicable guidelines, evidence-based recommendations.
Respond in 2-3 sentences citing relevant guidelines.""",
    
    "Ethics": """You are the Ethics Agent in a clinical decision support system.
Consider patient autonomy, informed consent, and ethical implications.
Focus on: patient rights, ethical considerations, quality of life.
Respond in 2-3 sentences with ethical guidance.""",
    
    "Data Quality": """You are the Data Quality Agent in a clinical decision support system.
Assess the completeness and reliability of the available data.
Focus on: missing data, data quality issues, confidence in findings.
Respond in 2-3 sentences about data reliability.""",
}


class ChatQueryEngine:
    """
    Fully integrated query engine using real LLM and trained RL.
    """
    
    def __init__(self):
        self.llm_client = get_llm_client()
        
        # Load trained RL policy
        policy_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "models", "rl_policy.pkl"
        )
        self.rl_coordinator = RLCoordinator(policy_path=policy_path)
        
        self.using_llm = self.llm_client.is_available
        print(f"ChatQueryEngine initialized: LLM={'available' if self.using_llm else 'mock mode'}")
    
    def query(
        self,
        question: str,
        patient_context: str,
        patient_id: str,
        patient_name: str,
        gait_data: Optional[Dict[str, float]] = None,
    ) -> ChatResponse:
        """Process a doctor's query using real agents and RL."""
        query_id = f"Q-{uuid.uuid4().hex[:8].upper()}"
        messages: List[AgentMessage] = []
        
        # Compute fall risk from gait data
        gait_data = gait_data or {}
        fall_risk = self._compute_fall_risk(gait_data, patient_context)
        
        # Build context for agents
        context = f"""
Patient: {patient_name}
Question: {question}

Medical History:
{patient_context}

Gait Metrics:
- Stride Length: {gait_data.get('stride_length', 'N/A')} m
- Gait Speed: {gait_data.get('gait_speed', 'N/A')} m/s
- Symmetry Index: {gait_data.get('symmetry_index', 'N/A')}
- Computed Fall Risk: {fall_risk:.0%}
"""
        
        # Run all agents
        agent_confidences = []
        agent_risks = []
        
        for agent_name, system_prompt in AGENT_PROMPTS.items():
            opinion, conf, risk = self._run_agent(agent_name, system_prompt, context)
            agent_confidences.append(conf)
            agent_risks.append(risk if risk else fall_risk)
            
            messages.append(AgentMessage(
                agent_name=f"{agent_name} Agent",
                message=opinion,
                timestamp=datetime.now().isoformat(),
                message_type="opinion",
                confidence=conf,
                risk_score=risk if risk else fall_risk,
            ))
        
        # Compute aggregate state for RL
        avg_conf = sum(agent_confidences) / len(agent_confidences)
        min_conf = min(agent_confidences)
        max_conf = max(agent_confidences)
        avg_risk = sum(agent_risks) / len(agent_risks)
        max_risk = max(agent_risks)
        
        # Estimate disagreement from confidence spread
        disagreement = max_conf - min_conf
        
        # Data quality based on gait data completeness
        data_quality = 0.9 if len(gait_data) >= 3 else (0.5 if gait_data else 0.2)
        
        # Build RL state
        rl_state = RLState(
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            avg_risk=avg_risk,
            max_risk=max_risk,
            disagreement_score=disagreement,
            data_quality_score=data_quality,
            vote_execute=0.3 if fall_risk < 0.4 else 0.1,
            vote_escalate=0.4,
            vote_refuse=0.2 if fall_risk > 0.7 else 0.1,
            vote_request_data=0.1 if data_quality > 0.5 else 0.3,
            has_veto=1.0 if fall_risk > 0.85 else 0.0,
        )
        
        # Get RL decision using trained policy
        from safety.evaluator import SafetyOutput
        safety_output = SafetyOutput(
            allowed=fall_risk < 0.85,
            triggered_rule=None,
            forced_action=None,
            explanation="Safety check passed" if fall_risk < 0.85 else "High risk detected",
        )
        
        rl_result = self.rl_coordinator.decide(rl_state, safety_output)
        
        # Add RL decision as final message
        messages.append(AgentMessage(
            agent_name="RL Coordinator",
            message=f"Decision: {rl_result.selected_action}. {rl_result.explanation}",
            timestamp=datetime.now().isoformat(),
            message_type="vote",
            confidence=rl_result.policy_confidence,
        ))
        
        # Determine risk level
        if fall_risk > 0.75:
            risk_level = "critical"
        elif fall_risk > 0.5:
            risk_level = "high"
        elif fall_risk > 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        # Generate final answer
        answer = self._generate_answer(question, patient_context, fall_risk, rl_result.selected_action)
        risk_factors = self._extract_risk_factors(patient_context, fall_risk)
        
        return ChatResponse(
            query_id=query_id,
            query=question,
            patient_id=patient_id,
            patient_name=patient_name,
            answer=answer,
            confidence=rl_result.policy_confidence,
            risk_level=risk_level,
            recommendation=rl_result.selected_action,
            agent_messages=messages,
            risk_factors=risk_factors,
            rl_explanation=rl_result.explanation,
            rl_was_overridden=rl_result.was_overridden,
            timestamp=datetime.now().isoformat(),
        )
    
    def _run_agent(self, agent_name: str, system_prompt: str, context: str) -> tuple:
        """Run a single agent and get its opinion."""
        response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=context,
        )
        
        # Parse confidence from response or estimate
        content = response.content
        confidence = 0.75  # Default
        risk = None
        
        # Try to extract any mentioned percentages
        import re
        percent_match = re.search(r'(\d+)%', content)
        if percent_match:
            value = int(percent_match.group(1)) / 100
            if 'risk' in content.lower():
                risk = min(value, 1.0)
            elif 'confidence' in content.lower():
                confidence = min(value, 1.0)
        
        return content, confidence, risk
    
    def _compute_fall_risk(self, gait_data: Dict[str, float], context: str) -> float:
        """Compute fall risk score deterministically."""
        base_risk = 0.3
        
        if gait_data:
            stride = gait_data.get("stride_length", 0.7)
            speed = gait_data.get("gait_speed", 1.0)
            symmetry = gait_data.get("symmetry_index", 0.9)
            
            if stride < 0.5:
                base_risk += 0.2
            if speed < 0.8:
                base_risk += 0.15
            if symmetry < 0.8:
                base_risk += 0.15
        
        context_lower = context.lower()
        if "fall" in context_lower:
            base_risk += 0.2
        if "diabetes" in context_lower:
            base_risk += 0.1
        if any(f"{age}" in context_lower for age in range(75, 100)):
            base_risk += 0.1
        
        return min(base_risk, 1.0)
    
    def _generate_answer(self, question: str, context: str, fall_risk: float, recommendation: str) -> str:
        """Generate the final answer using LLM."""
        prompt = f"""Based on the multi-agent analysis:
- Fall Risk: {fall_risk:.0%}
- Recommendation: {recommendation}

Patient Context: {context[:500]}

Question: {question}

Provide a concise, professional answer to the doctor's question in 2-3 sentences."""

        response = self.llm_client.generate(
            system_prompt="You are a clinical decision support system summarizing multi-agent findings.",
            user_prompt=prompt,
        )
        
        return response.content
    
    def _extract_risk_factors(self, context: str, fall_risk: float) -> List[str]:
        """Extract key risk factors using LLM."""
        prompt = f"""Analyze this patient context and list the key risk factors for falls.
Return only a comma-separated list of risk factors, nothing else.

Patient Context:
{context[:800]}

Computed Fall Risk: {fall_risk:.0%}

Risk factors:"""

        response = self.llm_client.generate(
            system_prompt="You are a clinical analyst. Extract risk factors concisely.",
            user_prompt=prompt,
            max_tokens=200,
        )
        
        # Parse comma-separated response
        factors = [f.strip() for f in response.content.split(",") if f.strip()]
        
        if fall_risk > 0.5:
            factors.append(f"Elevated fall risk ({fall_risk:.0%})")
        
        return factors if factors else ["Unable to extract risk factors"]


# Global engine instance
_engine: Optional[ChatQueryEngine] = None


def get_chat_engine() -> ChatQueryEngine:
    """Get the global chat engine instance."""
    global _engine
    if _engine is None:
        _engine = ChatQueryEngine()
    return _engine
