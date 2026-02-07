"""
Clinical War Room - Domain-Safe Exceptions

Hierarchical exception system for clear error handling.
All exceptions are designed for safe, informative error reporting.
"""

from typing import Optional, Any


class WarRoomException(Exception):
    """
    Base exception for all Clinical War Room errors.
    
    All exceptions include:
    - Human-readable message
    - Optional context for debugging
    - Safe defaults (never expose sensitive data)
    """
    
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Safe serialization for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "context": {k: v for k, v in self.context.items() 
                       if not k.startswith("_")}  # Filter private context
        }


# =============================================================================
# MCP Tool Exceptions
# =============================================================================

class MCPException(WarRoomException):
    """Base exception for MCP-related errors."""
    pass


class ToolNotFoundError(MCPException):
    """Raised when a requested tool is not registered."""
    
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' is not registered in MCP registry",
            {"tool_name": tool_name}
        )


class ToolExecutionError(MCPException):
    """Raised when a tool fails during execution."""
    
    def __init__(self, tool_name: str, reason: str):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {reason}",
            {"tool_name": tool_name, "reason": reason}
        )


class ToolSchemaError(MCPException):
    """Raised when tool input/output doesn't match schema."""
    
    def __init__(self, tool_name: str, field: str, expected: str, received: str):
        super().__init__(
            f"Schema validation failed for '{tool_name}': {field} expected {expected}, got {received}",
            {"tool_name": tool_name, "field": field, "expected": expected, "received": received}
        )


# =============================================================================
# Agent Exceptions
# =============================================================================

class AgentException(WarRoomException):
    """Base exception for agent-related errors."""
    pass


class AgentTimeoutError(AgentException):
    """Raised when an agent exceeds time limit."""
    
    def __init__(self, agent_name: str, timeout_seconds: int):
        super().__init__(
            f"Agent '{agent_name}' timed out after {timeout_seconds}s",
            {"agent_name": agent_name, "timeout_seconds": timeout_seconds}
        )


class AgentAnalysisError(AgentException):
    """Raised when agent analysis fails."""
    
    def __init__(self, agent_name: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' analysis failed: {reason}",
            {"agent_name": agent_name, "reason": reason}
        )


# =============================================================================
# Safety & Decision Exceptions
# =============================================================================

class SafetyException(WarRoomException):
    """Base exception for safety-related issues."""
    pass


class SafetyVetoError(SafetyException):
    """
    Raised when safety rules mandate a veto.
    
    This is NOT an error - it's a deliberate safety mechanism.
    The system is working as designed when this is raised.
    """
    
    def __init__(self, rule_name: str, reason: str, recommendation: str = "ESCALATE"):
        super().__init__(
            f"Safety veto triggered by '{rule_name}': {reason}",
            {"rule_name": rule_name, "reason": reason, "recommendation": recommendation}
        )
        self.recommendation = recommendation


class InsufficientConfidenceError(SafetyException):
    """Raised when confidence is too low for a decision."""
    
    def __init__(self, current_confidence: float, required_confidence: float):
        super().__init__(
            f"Insufficient confidence: {current_confidence:.2f} < {required_confidence:.2f}",
            {"current": current_confidence, "required": required_confidence}
        )


class HighRiskError(SafetyException):
    """Raised when risk exceeds acceptable threshold."""
    
    def __init__(self, current_risk: float, threshold: float):
        super().__init__(
            f"Risk too high: {current_risk:.2f} > {threshold:.2f}",
            {"current": current_risk, "threshold": threshold}
        )


# =============================================================================
# Case & Data Exceptions
# =============================================================================

class CaseException(WarRoomException):
    """Base exception for case-related errors."""
    pass


class CaseNotFoundError(CaseException):
    """Raised when a case cannot be found."""
    
    def __init__(self, case_id: str):
        super().__init__(
            f"Case '{case_id}' not found",
            {"case_id": case_id}
        )


class InvalidCaseDataError(CaseException):
    """Raised when case data is invalid or incomplete."""
    
    def __init__(self, case_id: str, missing_fields: list):
        super().__init__(
            f"Case '{case_id}' has invalid data: missing {missing_fields}",
            {"case_id": case_id, "missing_fields": missing_fields}
        )


class InsufficientDataError(CaseException):
    """Raised when there's not enough data for analysis."""
    
    def __init__(self, case_id: str, reason: str):
        super().__init__(
            f"Insufficient data for case '{case_id}': {reason}",
            {"case_id": case_id, "reason": reason}
        )


# =============================================================================
# RAG Exceptions
# =============================================================================

class RAGException(WarRoomException):
    """Base exception for RAG-related errors."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Failed to retrieve documents: {reason}",
            {"query_preview": query[:100], "reason": reason}
        )


# =============================================================================
# Debate Exceptions
# =============================================================================

class DebateException(WarRoomException):
    """Base exception for debate-related errors."""
    pass


class ConsensusNotReachedError(DebateException):
    """Raised when agents cannot reach consensus."""
    
    def __init__(self, disagreement_score: float, max_rounds: int):
        super().__init__(
            f"Consensus not reached after {max_rounds} rounds (disagreement: {disagreement_score:.2f})",
            {"disagreement_score": disagreement_score, "max_rounds": max_rounds}
        )
