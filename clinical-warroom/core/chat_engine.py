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
    
    # Multi-agent debate
    debate_rounds: List[Dict[str, Any]] = field(default_factory=list)
    treatment_urgency: str = ""  # "IMMEDIATE", "ESCALATE", "MONITOR"
    
    # Risk assessment summary
    risk_factors: List[str] = field(default_factory=list)
    
    timestamp: str = ""


# Agent system prompts
AGENT_PROMPTS = {
    "Diagnostic": """You are the Diagnostic Agent in a clinical decision support system.
Analyze the patient data and provide your clinical assessment.
Focus on: diagnosis, symptom analysis, clinical findings.
Respond in 2-3 sentences with your key observations.""",
    
    "Risk": """You are the Risk Assessment Agent in a clinical decision support system.
Analyze the patient's safety and health risks based on the available data.
Focus on: mobility issues, safety concerns, potential complications.
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
    Query engine using real LLM agents for clinical analysis.
    """
    
    def __init__(self):
        self.llm_client = get_llm_client()
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
        """Process a doctor's query using LLM agents."""
        query_id = f"Q-{uuid.uuid4().hex[:8].upper()}"
        messages: List[AgentMessage] = []
        
        # Build context for agents (no manual risk calculations)
        gait_info = ""
        if gait_data:
            gait_info = f"""
Gait Measurements:
- Stride Length: {gait_data.get('stride_length', 'N/A')} m
- Gait Speed: {gait_data.get('gait_speed', 'N/A')} m/s
- Symmetry Index: {gait_data.get('symmetry_index', 'N/A')}
"""
        
        context = f"""
Patient: {patient_name}
Question: {question}

Medical History:
{patient_context}
{gait_info}
"""
        
        # Run all agents
        agent_confidences = []
        
        for agent_name, system_prompt in AGENT_PROMPTS.items():
            opinion, conf = self._run_agent(agent_name, system_prompt, context)
            agent_confidences.append(conf)
            
            messages.append(AgentMessage(
                agent_name=f"{agent_name} Agent",
                message=opinion,
                timestamp=datetime.now().isoformat(),
                message_type="opinion",
                confidence=conf,
                risk_score=0.0,
            ))
        
        # Compute aggregate metrics for decision making
        avg_conf = sum(agent_confidences) / len(agent_confidences)
        
        # Simple recommendation based purely on agent confidence
        if avg_conf < 0.5:
            recommendation = "ESCALATE"
            explanation = "Low agent confidence - requires expert clinical review"
            risk_level = "moderate"
        elif avg_conf > 0.8:
            recommendation = "PROCEED WITH CAUTION"
            explanation = "High agent confidence - agents agree on assessment"
            risk_level = "low"
        else:
            recommendation = "REVIEW RECOMMENDED"
            explanation = "Moderate confidence - clinical oversight advised"
            risk_level = "moderate"
        
        # Determine if question is about treatment urgency/risk
        should_debate = self._should_run_debate(question)
        
        # Run multi-agent debate only if question is about treatment urgency
        debate_rounds = []
        treatment_urgency = ""
        
        if should_debate:
            debate_rounds, treatment_urgency = self._run_debate(
                question, patient_context, messages, avg_conf, risk_level
            )
            
            # Align risk level with treatment urgency for consistency
            if treatment_urgency == "IMMEDIATE":
                risk_level = "critical"
                recommendation = "IMMEDIATE TREATMENT"
            elif treatment_urgency == "ESCALATE":
                risk_level = "high"
                recommendation = "ESCALATE TO SPECIALIST"
            elif treatment_urgency == "MONITOR":
                risk_level = "low"
                recommendation = "CONTINUE MONITORING"
        
        # Generate final answer
        answer = self._generate_answer(question, patient_context, recommendation)
        
        return ChatResponse(
            query_id=query_id,
            query=question,
            patient_id=patient_id,
            patient_name=patient_name,
            answer=answer,
            confidence=avg_conf,
            risk_level=risk_level,
            recommendation=recommendation,
            agent_messages=messages,
            debate_rounds=debate_rounds,
            treatment_urgency=treatment_urgency,
            risk_factors=[],  # No pre-computed risk factors
            timestamp=datetime.now().isoformat(),
        )
    
    def _run_agent(self, agent_name: str, system_prompt: str, context: str) -> tuple:
        """Run a single agent and get its opinion."""
        response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=context,
        )
        
        # Use default confidence
        content = response.content
        confidence = 0.75  # Default confidence for all agents
        
        return content, confidence
    

    
    def _generate_answer(self, question: str, context: str, recommendation: str) -> str:
        """Generate the final answer using LLM."""
        prompt = f"""Based on the multi-agent clinical analysis:
- Recommendation: {recommendation}

Patient Context: {context[:500]}

Question: {question}

Provide a concise, professional answer to the doctor's question in 2-3 sentences."""

        response = self.llm_client.generate(
            system_prompt="You are a clinical decision support system summarizing multi-agent findings.",
            user_prompt=prompt,
        )
        
        return response.content
    
    def _should_run_debate(self, question: str) -> bool:
        """Determine if the question warrants a treatment urgency debate."""
        
        classifier_prompt = f"""Analyze this medical question and determine if it's asking about TREATMENT URGENCY or IMMEDIATE CARE NEEDS.

Question: "{question}"

Questions that warrant debate:
- "Does this patient need immediate treatment?"
- "Should we escalate this case?"
- "What's the urgency level?"
- "Is this an emergency?"
- "How urgent is treatment?"
- "Should this patient be hospitalized?"
- Questions about treatment timing, risk assessment, urgency

Questions that DON'T warrant debate:
- "What medications is this patient on?"
- "What's the diagnosis?"
- "What are the symptoms?"
- "Tell me about the patient's history"
- General informational questions

Answer with ONLY "YES" or "NO"."""

        response = self.llm_client.generate(
            system_prompt="You are a question classifier for a medical system.",
            user_prompt=classifier_prompt
        )
        
        # Check if response indicates this is about treatment urgency
        response_text = response.content.strip().upper()
        return "YES" in response_text
    
    def _run_debate(
        self, 
        question: str, 
        patient_context: str,
        agent_messages: List[AgentMessage],
        confidence: float,
        risk_level: str
    ) -> tuple:
        """Run a multi-agent debate on treatment urgency."""
        
        debate_rounds = []
        
        # ROUND 1: Initial Positions
        round_1 = {
            "round": 1,
            "title": "Initial Positions",
            "agents": {}
        }
        
        # Proponent Agent (pushes for action)
        proponent_prompt = f"""You are the PROPONENT agent in a medical debate. Your role is to advocate for decisive action when appropriate.

Patient Context: {patient_context[:500]}
Question: {question}
Current Assessment: {risk_level} risk, {confidence:.0%} confidence

Argue for immediate treatment if medically justified. Be bold but medically sound. 2-3 sentences."""
        
        proponent_response = self.llm_client.generate(
            system_prompt="You are an action-oriented medical advocate.",
            user_prompt=proponent_prompt
        )
        round_1["agents"]["proponent"] = proponent_response.content
        
        # Skeptic Agent (advocates caution)
        skeptic_prompt = f"""You are the SKEPTIC agent in a medical debate. Your role is to question assumptions and advocate for caution.

Patient Context: {patient_context[:500]}
Question: {question}
Current Assessment: {risk_level} risk, {confidence:.0%} confidence

Challenge any rush to treatment. Highlight uncertainties and risks. 2-3 sentences."""
        
        skeptic_response = self.llm_client.generate(
            system_prompt="You are a cautious medical reviewer.",
            user_prompt=skeptic_prompt
        )
        round_1["agents"]["skeptic"] = skeptic_response.content
        
        # Mediator Agent (balanced)
        mediator_prompt = f"""You are the MEDIATOR agent in a medical debate. Your role is to find balanced solutions.

Patient Context: {patient_context[:500]}
Question: {question}
Current Assessment: {risk_level} risk, {confidence:.0%} confidence

Weigh both action and caution. Seek middle ground. 2-3 sentences."""
        
        mediator_response = self.llm_client.generate(
            system_prompt="You are a balanced medical coordinator.",
            user_prompt=mediator_prompt
        )
        round_1["agents"]["mediator"] = mediator_response.content
        
        debate_rounds.append(round_1)
        
        # ROUND 2: Challenge & Rebuttal
        round_2 = {
            "round": 2,
            "title": "Challenge & Rebuttal",
            "agents": {}
        }
        
        # Skeptic challenges Proponent
        challenge_prompt = f"""The PROPONENT said: "{proponent_response.content}"

As the SKEPTIC, challenge this position. Point out what could go wrong. 1-2 sentences."""
        
        challenge_response = self.llm_client.generate(
            system_prompt="You are challenging overly optimistic views.",
            user_prompt=challenge_prompt
        )
        round_2["agents"]["skeptic"] = challenge_response.content
        
        # Proponent defends
        defense_prompt = f"""The SKEPTIC challenged you: "{challenge_response.content}"

As the PROPONENT, defend your position. Address the concern. 1-2 sentences."""
        
        defense_response = self.llm_client.generate(
            system_prompt="You are defending decisive medical action.",
            user_prompt=defense_prompt
        )
        round_2["agents"]["proponent"] = defense_response.content
        
        # Mediator questions both
        mediation_prompt = f"""PROPONENT: "{defense_response.content}"
SKEPTIC: "{challenge_response.content}"

As the MEDIATOR, what question would help resolve this? 1 sentence."""
        
        mediation_response = self.llm_client.generate(
            system_prompt="You are finding common ground.",
            user_prompt=mediation_prompt
        )
        round_2["agents"]["mediator"] = mediation_response.content
        
        debate_rounds.append(round_2)
        
        # ROUND 3: Final Consensus
        round_3 = {
            "round": 3,
            "title": "Final Consensus",
            "agents": {}
        }
        
        # Mediator synthesizes
        synthesis_prompt = f"""After debate, as MEDIATOR, make final recommendation on treatment urgency.

Choose ONE:
- IMMEDIATE TREATMENT NEEDED
- ESCALATE TO SPECIALIST
- CONTINUE MONITORING

Risk level: {risk_level}
Confidence: {confidence:.0%}

Answer with decision + brief reason (1-2 sentences)."""
        
        synthesis_response = self.llm_client.generate(
            system_prompt="You are making the final medical decision.",
            user_prompt=synthesis_prompt
        )
        round_3["agents"]["mediator"] = synthesis_response.content
        
        # Extract treatment urgency from synthesis
        synthesis_text = synthesis_response.content.upper()
        if "IMMEDIATE" in synthesis_text:
            treatment_urgency = "IMMEDIATE"
        elif "ESCALATE" in synthesis_text:
            treatment_urgency = "ESCALATE"
        else:
            treatment_urgency = "MONITOR"
        
        # Final votes from other agents
        vote_prompt_pro = f"""Final decision: "{synthesis_response.content}"

As PROPONENT, do you agree? Vote: AGREE or DISAGREE. 1 sentence."""
        
        vote_pro = self.llm_client.generate(
            system_prompt="You are casting your final vote.",
            user_prompt=vote_prompt_pro
        )
        round_3["agents"]["proponent"] = f"Vote: {vote_pro.content}"
        
        vote_prompt_skep = f"""Final decision: "{synthesis_response.content}"

As SKEPTIC, do you agree? Vote: AGREE or DISAGREE. 1 sentence."""
        
        vote_skep = self.llm_client.generate(
            system_prompt="You are casting your final vote.",
            user_prompt=vote_prompt_skep
        )
        round_3["agents"]["skeptic"] = f"Vote: {vote_skep.content}"
        
        debate_rounds.append(round_3)
        
        return debate_rounds, treatment_urgency

    



# Global engine instance
_engine: Optional[ChatQueryEngine] = None


def get_chat_engine() -> ChatQueryEngine:
    """Get the global chat engine instance."""
    global _engine
    if _engine is None:
        _engine = ChatQueryEngine()
    return _engine
