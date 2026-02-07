"""
Clinical War Room - MCP Tool Registration

Registers all deterministic tools with the MCP registry.
This is the ONLY place tools are made available to agents.
"""

import time
from typing import Dict, Any
from core.mcp.schemas import (
    ToolDefinition,
    ToolCategory,
    ToolInput,
    ToolOutput,
    ToolStatus,
)
from core.mcp.registry import registry
from core.logging import logger

# Import tool implementations
from tools.gait.feature_extractor import (
    extract_gait_features,
    parse_sensor_data,
)
from tools.quality.data_quality import check_data_quality
from tools.risk.fall_risk_model import predict_fall_risk
from tools.reporting.report_generator import generate_report


# =============================================================================
# Tool Definitions (Schemas)
# =============================================================================

GAIT_FEATURE_EXTRACTOR_DEF = ToolDefinition(
    name="gait_feature_extractor",
    category=ToolCategory.GAIT,
    description="Extract gait features (cadence, stride length, speed, asymmetry, variability) from IMU sensor data",
    version="1.0.0",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_data": {
                "type": "array",
                "description": "List of sensor samples with timestamp_ms, accel_x/y/z",
                "items": {"type": "object"}
            },
            "patient_height_m": {
                "type": "number",
                "description": "Patient height in meters for stride estimation",
                "default": 1.7
            },
            "sampling_rate_hz": {
                "type": "number",
                "description": "Sensor sampling rate in Hz",
                "default": 100
            }
        },
        "required": ["sensor_data"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "cadence": {"type": "number"},
            "stride_length": {"type": "number"},
            "gait_speed": {"type": "number"},
            "asymmetry_index": {"type": "number"},
            "variability": {"type": "number"},
            "step_count": {"type": "integer"},
            "analysis_duration_ms": {"type": "number"},
        }
    },
    timeout_seconds=10,
)

DATA_QUALITY_CHECKER_DEF = ToolDefinition(
    name="data_quality_checker",
    category=ToolCategory.QUALITY,
    description="Assess quality and reliability of raw gait sensor data",
    version="1.0.0",
    input_schema={
        "type": "object",
        "properties": {
            "raw_data": {
                "type": "array",
                "description": "Raw sensor data samples",
                "items": {"type": "object"}
            },
            "required_fields": {
                "type": "array",
                "description": "Fields required in each sample",
                "items": {"type": "string"},
                "default": ["timestamp_ms", "accel_x", "accel_y", "accel_z"]
            }
        },
        "required": ["raw_data"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "missing_data_percentage": {"type": "number"},
            "noise_score": {"type": "number"},
            "reliability_score": {"type": "number"},
            "total_samples": {"type": "integer"},
            "valid_samples": {"type": "integer"},
            "issues": {"type": "array", "items": {"type": "string"}},
        }
    },
    timeout_seconds=5,
)

FALL_RISK_PREDICTOR_DEF = ToolDefinition(
    name="fall_risk_predictor",
    category=ToolCategory.RISK,
    description="Predict fall risk probability from gait features using deterministic model",
    version="1.0.0",
    input_schema={
        "type": "object",
        "properties": {
            "gait_features": {
                "type": "object",
                "description": "Output from gait_feature_extractor",
            },
            "patient_age": {
                "type": "integer",
                "description": "Patient age for risk adjustment",
            },
            "medical_history": {
                "type": "array",
                "description": "List of relevant medical conditions",
                "items": {"type": "string"}
            },
            "data_quality_score": {
                "type": "number",
                "description": "Reliability score from data_quality_checker",
            }
        },
        "required": ["gait_features"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "fall_risk_probability": {"type": "number"},
            "risk_level": {"type": "string"},
            "contributing_factors": {"type": "array", "items": {"type": "string"}},
            "protective_factors": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        }
    },
    timeout_seconds=5,
)

REPORT_GENERATOR_DEF = ToolDefinition(
    name="report_generator",
    category=ToolCategory.REPORTING,
    description="Generate structured clinical report from computed metrics",
    version="1.0.0",
    input_schema={
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "patient_id": {"type": "string"},
            "patient_info": {"type": "object"},
            "gait_features": {"type": "object"},
            "data_quality": {"type": "object"},
            "fall_risk": {"type": "object"},
        },
        "required": ["case_id", "patient_id", "patient_info"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "patient_id": {"type": "string"},
            "report_type": {"type": "string"},
            "generated_at": {"type": "string"},
            "sections": {"type": "array"},
            "summary": {"type": "string"},
            "risk_level": {"type": "string"},
            "text_report": {"type": "string"},
        }
    },
    timeout_seconds=5,
)


# =============================================================================
# Tool Handlers
# =============================================================================

def _handle_gait_feature_extractor(input: ToolInput) -> ToolOutput:
    """Handle gait feature extraction."""
    start = time.time()
    
    try:
        params = input.parameters
        sensor_data = params.get("sensor_data", [])
        patient_height = params.get("patient_height_m", 1.7)
        sampling_rate = params.get("sampling_rate_hz", 100)
        
        # Parse and extract features
        samples = parse_sensor_data(sensor_data)
        features = extract_gait_features(
            samples=samples,
            patient_height_m=patient_height,
            sampling_rate_hz=sampling_rate,
        )
        
        return ToolOutput(
            tool_name="gait_feature_extractor",
            status=ToolStatus.SUCCESS,
            result=features.to_dict(),
            confidence=1.0 if features.step_count >= 10 else 0.5,
            execution_time_ms=(time.time() - start) * 1000,
        )
        
    except Exception as e:
        return ToolOutput(
            tool_name="gait_feature_extractor",
            status=ToolStatus.ERROR,
            result={},
            confidence=0.0,
            execution_time_ms=(time.time() - start) * 1000,
            error_message=str(e),
        )


def _handle_data_quality_checker(input: ToolInput) -> ToolOutput:
    """Handle data quality assessment."""
    start = time.time()
    
    try:
        params = input.parameters
        raw_data = params.get("raw_data", [])
        required_fields = params.get("required_fields")
        
        result = check_data_quality(
            raw_data=raw_data,
            required_fields=required_fields,
        )
        
        return ToolOutput(
            tool_name="data_quality_checker",
            status=ToolStatus.SUCCESS,
            result=result.to_dict(),
            confidence=1.0,  # Quality check is deterministic
            execution_time_ms=(time.time() - start) * 1000,
        )
        
    except Exception as e:
        return ToolOutput(
            tool_name="data_quality_checker",
            status=ToolStatus.ERROR,
            result={},
            confidence=0.0,
            execution_time_ms=(time.time() - start) * 1000,
            error_message=str(e),
        )


def _handle_fall_risk_predictor(input: ToolInput) -> ToolOutput:
    """Handle fall risk prediction."""
    start = time.time()
    
    try:
        params = input.parameters
        gait_features = params.get("gait_features", {})
        patient_age = params.get("patient_age")
        medical_history = params.get("medical_history")
        data_quality_score = params.get("data_quality_score")
        
        result = predict_fall_risk(
            gait_features=gait_features,
            patient_age=patient_age,
            medical_history=medical_history,
            data_quality_score=data_quality_score,
        )
        
        return ToolOutput(
            tool_name="fall_risk_predictor",
            status=ToolStatus.SUCCESS,
            result=result.to_dict(),
            confidence=result.confidence,
            execution_time_ms=(time.time() - start) * 1000,
        )
        
    except Exception as e:
        return ToolOutput(
            tool_name="fall_risk_predictor",
            status=ToolStatus.ERROR,
            result={},
            confidence=0.0,
            execution_time_ms=(time.time() - start) * 1000,
            error_message=str(e),
        )


def _handle_report_generator(input: ToolInput) -> ToolOutput:
    """Handle report generation."""
    start = time.time()
    
    try:
        params = input.parameters
        
        report = generate_report(
            case_id=params.get("case_id", input.case_id),
            patient_id=params.get("patient_id", ""),
            patient_info=params.get("patient_info", {}),
            gait_features=params.get("gait_features"),
            data_quality=params.get("data_quality"),
            fall_risk=params.get("fall_risk"),
        )
        
        result = report.to_dict()
        result["text_report"] = report.to_text()
        
        return ToolOutput(
            tool_name="report_generator",
            status=ToolStatus.SUCCESS,
            result=result,
            confidence=1.0,
            execution_time_ms=(time.time() - start) * 1000,
        )
        
    except Exception as e:
        return ToolOutput(
            tool_name="report_generator",
            status=ToolStatus.ERROR,
            result={},
            confidence=0.0,
            execution_time_ms=(time.time() - start) * 1000,
            error_message=str(e),
        )


# =============================================================================
# Registration Function
# =============================================================================

def register_all_tools() -> None:
    """Register all MCP tools. Call once at startup."""
    log = logger.with_context(phase="tool_registration")
    
    # Check if already registered
    if registry.is_registered("gait_feature_extractor"):
        log.info("Tools already registered, skipping")
        return
    
    log.info("Registering MCP tools...")
    
    # Register each tool
    registry.register(GAIT_FEATURE_EXTRACTOR_DEF, _handle_gait_feature_extractor)
    log.info("Registered: gait_feature_extractor")
    
    registry.register(DATA_QUALITY_CHECKER_DEF, _handle_data_quality_checker)
    log.info("Registered: data_quality_checker")
    
    registry.register(FALL_RISK_PREDICTOR_DEF, _handle_fall_risk_predictor)
    log.info("Registered: fall_risk_predictor")
    
    registry.register(REPORT_GENERATOR_DEF, _handle_report_generator)
    log.info("Registered: report_generator")
    
    log.info(f"Registered {len(registry.list_tools())} MCP tools")


# Auto-register on import
register_all_tools()
