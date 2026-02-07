"""
Clinical War Room - LLM Client

Wrapper for LLM API calls.
Uses Groq for fast inference.
"""

import os
import json
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

from core.logging import logger


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    success: bool
    error: Optional[str] = None


class LLMClient:
    """
    Client for LLM API calls.
    
    Rules:
    - LLM is used ONLY for reasoning and explanation
    - LLM must NEVER compute numeric features
    - LLM must NEVER override MCP tool outputs
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log = logger.with_context(phase="llm_client")
        
        self._client = None
        if HAS_GROQ and self.api_key:
            self._client = Groq(api_key=self.api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if LLM client is configured and available."""
        return self._client is not None
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            LLMResponse with content or error
        """
        if not self.is_available:
            self.log.warning("LLM client not available, returning mock response")
            return self._mock_response(system_prompt, user_prompt)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            self.log.info(f"LLM response generated ({usage['total_tokens']} tokens)")
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                success=True,
            )
            
        except Exception as e:
            self.log.error(f"LLM generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                usage={},
                success=False,
                error=str(e),
            )
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> tuple:
        """
        Generate a JSON response from the LLM.
        
        Returns:
            Tuple of (parsed_json, raw_content, error)
        """
        response = self.generate(system_prompt, user_prompt, **kwargs)
        
        if not response.success:
            return None, response.content, response.error
        
        # Try to extract JSON from response
        content = response.content
        
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        
        try:
            parsed = json.loads(content)
            return parsed, response.content, None
        except json.JSONDecodeError as e:
            # Try to find JSON object in content
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end])
                    return parsed, response.content, None
            except:
                pass
            
            return None, response.content, f"JSON parse error: {e}"
    
    def _mock_response(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> LLMResponse:
        """Generate a mock response when LLM is unavailable."""
        # Extract agent name from system prompt - match multi-word names
        agent_match = re.search(r'You are the ([\w\s]+?) Agent', system_prompt)
        agent_name = agent_match.group(1).strip() if agent_match else "Unknown"
        
        mock_json = {
            "agent_name": f"{agent_name} Agent",
            "claim": f"Based on the provided data, this case requires careful clinical review. [MOCK RESPONSE - LLM not available]",
            "confidence": 0.5,
            "risk": 0.5,
            "evidence": [
                {"source": "MCP tools", "content": "Tool outputs analyzed", "relevance": 1.0}
            ],
            "concerns": [
                {"description": "LLM unavailable - mock response generated", "severity": "moderate"}
            ],
            "reasoning": "This is a mock response because the LLM client is not configured. Configure GROQ_API_KEY for real analysis.",
            "veto": False,
        }
        
        return LLMResponse(
            content=json.dumps(mock_json, indent=2),
            model="mock",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            success=True,
        )


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
