"""
Clinical War Room - MCP Client

Tool invocation interface.
This is the ONLY way agents can execute tools.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from core.mcp.registry import registry, RegisteredTool
from core.mcp.schemas import ToolInput, ToolOutput, ToolStatus
from core.exceptions import ToolNotFoundError, ToolExecutionError
from core.logging import logger


@dataclass
class InvocationResult:
    """Result of a tool invocation."""
    success: bool
    output: Optional[ToolOutput]
    error: Optional[str] = None


class MCPClient:
    """
    Client for invoking MCP tools.
    
    Design principles:
    - ALL tool invocations go through this client
    - Client handles validation, timing, and error handling
    - Client logs all invocations for audit trail
    - Agents NEVER call tools directly
    """
    
    def __init__(self):
        self.registry = registry
        self.log = logger.with_context(phase="mcp_client")
    
    def call(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        case_id: str = "unknown",
    ) -> ToolOutput:
        """
        Call a registered tool by name.
        
        This is the primary interface for tool invocation.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool-specific arguments
            case_id: Case identifier for tracking
            
        Returns:
            ToolOutput with result or error
            
        Raises:
            ToolNotFoundError: If tool is not registered
        """
        # Check if tool exists
        if not self.registry.is_registered(tool_name):
            self.log.error(f"Tool not found: {tool_name}")
            raise ToolNotFoundError(tool_name)
        
        tool = self.registry.get(tool_name)
        
        # Check if tool is enabled
        if not tool.enabled:
            self.log.warning(f"Tool disabled: {tool_name}")
            return ToolOutput(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                result={},
                confidence=0.0,
                execution_time_ms=0.0,
                error_message=f"Tool '{tool_name}' is currently disabled",
            )
        
        # Build input
        tool_input = ToolInput(
            case_id=case_id,
            patient_data={},  # Not used in direct call
            parameters=arguments,
        )
        
        # Execute with timing
        self.log.info(f"Calling tool: {tool_name} for case {case_id}")
        start_time = time.time()
        
        try:
            output = tool.handler(tool_input)
            execution_time = (time.time() - start_time) * 1000
            
            # Update execution time if not set
            if output.execution_time_ms == 0:
                output.execution_time_ms = execution_time
            
            self.log.info(
                f"Tool {tool_name} completed in {execution_time:.2f}ms "
                f"(status: {output.status.value}, confidence: {output.confidence:.2f})"
            )
            
            return output
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            self.log.error(f"Tool {tool_name} failed: {error_msg}")
            
            return ToolOutput(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                result={},
                confidence=0.0,
                execution_time_ms=execution_time,
                error_message=error_msg,
            )
    
    def invoke(
        self, 
        tool_name: str, 
        case_id: str,
        patient_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> InvocationResult:
        """
        Invoke a registered tool with full context.
        
        Args:
            tool_name: Name of the tool to invoke
            case_id: Case identifier for tracking
            patient_data: Patient data to process
            parameters: Additional tool parameters
            
        Returns:
            InvocationResult with output or error
        """
        # Use call() method internally
        try:
            output = self.call(
                tool_name=tool_name,
                arguments=parameters or {},
                case_id=case_id,
            )
            
            return InvocationResult(
                success=output.is_success(),
                output=output,
                error=output.error_message,
            )
            
        except ToolNotFoundError as e:
            return InvocationResult(
                success=False,
                output=None,
                error=str(e),
            )
    
    def call_pipeline(
        self,
        case_id: str,
        sensor_data: list,
        patient_info: Dict[str, Any],
    ) -> Dict[str, ToolOutput]:
        """
        Execute the complete gait analysis pipeline.
        
        Calls tools in order:
        1. data_quality_checker
        2. gait_feature_extractor
        3. fall_risk_predictor
        4. report_generator
        
        Args:
            case_id: Case identifier
            sensor_data: Raw IMU sensor samples
            patient_info: Patient demographics and history
            
        Returns:
            Dict mapping tool names to their outputs
        """
        results = {}
        
        # Step 1: Check data quality
        quality_output = self.call(
            "data_quality_checker",
            {"raw_data": sensor_data},
            case_id=case_id,
        )
        results["data_quality_checker"] = quality_output
        
        # Step 2: Extract gait features
        feature_output = self.call(
            "gait_feature_extractor",
            {
                "sensor_data": sensor_data,
                "patient_height_m": patient_info.get("height_m", 1.7),
            },
            case_id=case_id,
        )
        results["gait_feature_extractor"] = feature_output
        
        # Step 3: Predict fall risk
        risk_output = self.call(
            "fall_risk_predictor",
            {
                "gait_features": feature_output.result,
                "patient_age": patient_info.get("age"),
                "medical_history": patient_info.get("medical_history", []),
                "data_quality_score": quality_output.result.get("reliability_score", 1.0),
            },
            case_id=case_id,
        )
        results["fall_risk_predictor"] = risk_output
        
        # Step 4: Generate report
        report_output = self.call(
            "report_generator",
            {
                "case_id": case_id,
                "patient_id": patient_info.get("patient_id", "unknown"),
                "patient_info": patient_info,
                "gait_features": feature_output.result,
                "data_quality": quality_output.result,
                "fall_risk": risk_output.result,
            },
            case_id=case_id,
        )
        results["report_generator"] = report_output
        
        return results
    
    def list_available_tools(self) -> list:
        """List all available tools."""
        return self.registry.list_tools()


# Global client instance
mcp_client = MCPClient()
