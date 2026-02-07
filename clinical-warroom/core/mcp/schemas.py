"""
Clinical War Room - MCP Schemas

Model Context Protocol contracts for tool input/output.
All tools MUST conform to these schemas.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from enum import Enum
from datetime import datetime


class ToolCategory(str, Enum):
    """Categories of MCP tools."""
    GAIT = "gait"
    QUALITY = "quality"
    RISK = "risk"
    REPORTING = "reporting"


class ToolStatus(str, Enum):
    """Execution status of a tool."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolInput:
    """
    Standard input contract for all MCP tools.
    
    Tools receive structured input and MUST NOT
    access external data directly.
    """
    case_id: str
    patient_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def validate(self) -> bool:
        """Validate input has required fields."""
        if not self.case_id:
            return False
        if not self.patient_data:
            return False
        return True


@dataclass
class ToolOutput:
    """
    Standard output contract for all MCP tools.
    
    Tools produce evidence, NEVER decisions.
    All outputs include confidence and metadata.
    """
    tool_name: str
    status: ToolStatus
    result: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    execution_time_ms: float
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "warnings": self.warnings,
            "error_message": self.error_message,
        }


@dataclass
class ToolDefinition:
    """
    Schema definition for a registered tool.
    
    Used by MCP registry to validate and document tools.
    """
    name: str
    category: ToolCategory
    description: str
    version: str
    input_schema: Dict[str, Any]  # JSON Schema format
    output_schema: Dict[str, Any]  # JSON Schema format
    requires_patient_data: bool = True
    timeout_seconds: int = 30
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "requires_patient_data": self.requires_patient_data,
            "timeout_seconds": self.timeout_seconds,
        }


# =============================================================================
# Pre-defined Tool Schemas (Placeholders for Phase 1)
# =============================================================================

GAIT_FEATURE_SCHEMA = ToolDefinition(
    name="gait_feature_extractor",
    category=ToolCategory.GAIT,
    description="Extract gait features from sensor data",
    version="0.1.0",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_data": {"type": "array", "items": {"type": "number"}},
            "sampling_rate": {"type": "number"},
        },
        "required": ["sensor_data", "sampling_rate"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "stride_length": {"type": "number"},
            "cadence": {"type": "number"},
            "symmetry_index": {"type": "number"},
            "variability": {"type": "number"},
        }
    }
)

DATA_QUALITY_SCHEMA = ToolDefinition(
    name="data_quality_checker",
    category=ToolCategory.QUALITY,
    description="Assess quality and completeness of patient data",
    version="0.1.0",
    input_schema={
        "type": "object",
        "properties": {
            "patient_data": {"type": "object"},
        },
        "required": ["patient_data"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "completeness_score": {"type": "number"},
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "data_quality_flags": {"type": "array", "items": {"type": "string"}},
        }
    }
)

FALL_RISK_SCHEMA = ToolDefinition(
    name="fall_risk_predictor",
    category=ToolCategory.RISK,
    description="Predict fall risk from patient features",
    version="0.1.0",
    input_schema={
        "type": "object",
        "properties": {
            "gait_features": {"type": "object"},
            "patient_history": {"type": "object"},
        },
        "required": ["gait_features"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "risk_score": {"type": "number"},
            "risk_level": {"type": "string", "enum": ["low", "moderate", "high", "critical"]},
            "contributing_factors": {"type": "array", "items": {"type": "string"}},
        }
    }
)
