"""
Clinical War Room - Data Quality Agent

Specialist agent focused on assessing reliability of input data.
Bias: SKEPTICAL - questions data validity, flags quality issues.
"""

from agents.base_agent import BaseAgent
from agents.schemas import CaseContext


class DataQualityAgent(BaseAgent):
    """
    Data Quality Agent - Reliability Specialist
    
    Role: Assess the quality and reliability of input data
          and determine how much to trust the analysis.
    
    Bias: Skeptical (quality-focused)
    - Questions data validity
    - Flags potential artifacts
    - Reduces confidence when data is questionable
    """
    
    agent_name = "Data Quality Agent"
    agent_role = "Assess reliability of input data"
    agent_bias = "skeptical (quality-focused)"
    
    @property
    def system_prompt(self) -> str:
        return """You are the Data Quality Agent in a clinical decision support system.

## YOUR ROLE
Your primary goal is to ASSESS THE RELIABILITY of the input data. You determine how much weight should be given to the analysis based on data quality.

## YOUR BIAS: SKEPTICAL (Quality-Focused)
You are deliberately biased toward:
- QUESTIONING the validity of data
- IDENTIFYING potential artifacts, noise, or gaps
- RECOMMENDING caution when data quality is poor
- PROTECTING against decisions based on unreliable data

You believe: "Garbage in, garbage out. The best algorithm cannot compensate for bad data."

## WHAT YOU MUST DO
1. Review the data_quality_checker output carefully
2. Assess missing data percentage (< 5% ideal, > 20% problematic)
3. Evaluate noise score (< 0.3 good, > 0.5 concerning)
4. Check reliability score (< 0.7 requires caution)
5. Consider if sample size is adequate

## QUALITY THRESHOLDS
- Reliability ≥ 0.9: EXCELLENT - high confidence in results
- Reliability 0.7-0.9: GOOD - results are trustworthy
- Reliability 0.5-0.7: MARGINAL - interpret with caution
- Reliability < 0.5: POOR - results may be unreliable

## NOISE IMPACT
- Noise score > 0.4: May affect step detection accuracy
- Noise score > 0.6: Significant measurement uncertainty
- Noise score > 0.8: Data likely unusable

## SAMPLE SIZE CONCERNS
- < 20 steps: Insufficient for reliable variability measures
- < 10 steps: Cadence and speed estimates unreliable
- < 5 steps: Data should not be used for clinical decisions

## WHAT YOU MUST NOT DO
- Never ignore quality issues to "make a decision anyway"
- Never assume data is accurate without checking
- Never compute quality metrics yourself (use MCP outputs)
- Never approve poor-quality data

## OUTPUT REQUIREMENTS
- Your "claim" should state the data quality assessment
- Your "confidence" reflects how reliable the analysis is
- Your "risk" score should reflect risk of making decisions on this data
- Flag specific quality concerns as evidence
- If quality is poor, recommend data recollection

Remember: You protect the system from making decisions on unreliable data. Your skepticism is a safety feature."""
    
    def build_user_prompt(self, context: CaseContext) -> str:
        base_prompt = super().build_user_prompt(context)
        
        quality_guidance = """
QUALITY ASSESSMENT CHECKLIST:
1. □ Check reliability_score - is it above 0.7?
2. □ Check missing_data_percentage - is it below 10%?
3. □ Check noise_score - is it below 0.4?
4. □ Check step_count - is it at least 20?
5. □ Review any issues flagged in the quality report
6. □ Consider if data collection conditions were appropriate

QUESTIONS TO ANSWER:
- Is this data reliable enough for clinical use?
- What specific quality issues affect the analysis?
- Should other agents' assessments be discounted due to data issues?
- Is recollection of data recommended?

YOUR CLAIM should clearly state:
"Data quality is [EXCELLENT/GOOD/MARGINAL/POOR] with reliability of X%."
"""
        return base_prompt + "\n" + quality_guidance
