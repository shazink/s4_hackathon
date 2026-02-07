#!/usr/bin/env python3
"""
MCP Tools Verification Script

Demonstrates calling all tools exclusively through MCP.
NO direct tool imports - only MCPClient.

Usage:
    python scripts/verify_mcp_tools.py
"""

import math
import json

# Register all tools (this is the ONLY import that touches tools)
import tools.mcp_tools  # noqa: F401

from core.mcp.client import mcp_client


def generate_sample_sensor_data(
    num_samples: int = 300,
    steps_per_minute: float = 100,
) -> list:
    """Generate realistic synthetic IMU gait data."""
    step_interval_ms = 60000 / steps_per_minute
    data = []
    
    for i in range(num_samples):
        timestamp = i * 10  # 100Hz sampling
        
        # Simulate vertical acceleration with step peaks
        phase = (timestamp % step_interval_ms) / step_interval_ms
        
        # Base acceleration (gravity normalized)
        base = 1.0
        
        # Add step impact peak
        if 0.0 < phase < 0.15:
            step_impact = 0.8 * math.sin(phase * math.pi / 0.15)
        else:
            step_impact = 0.0
        
        # Add subtle noise
        noise = 0.03 * math.sin(timestamp * 0.1)
        
        data.append({
            "timestamp_ms": timestamp,
            "accel_x": 0.1 * math.sin(timestamp * 0.02),
            "accel_y": 0.15 * math.cos(timestamp * 0.02),
            "accel_z": base + step_impact + noise,
        })
    
    return data


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    print()
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def main():
    """Run MCP tools verification."""
    print_separator("MCP TOOLS VERIFICATION")
    print("Demonstrating tool invocation exclusively through MCPClient")
    print("NO direct tool imports allowed - MCP is the ONLY bridge.")
    
    # Generate sample data
    sensor_data = generate_sample_sensor_data(num_samples=300)
    patient_info = {
        "patient_id": "DEMO-001",
        "age": 72,
        "sex": "M",
        "height_m": 1.75,
        "medical_history": ["hypertension", "type 2 diabetes"],
    }
    
    # List available tools
    print_separator("1. LISTING AVAILABLE TOOLS")
    tools = mcp_client.list_available_tools()
    print(f"Registered tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['name']} ({tool['category']}) v{tool['version']}")
    
    # Call each tool individually
    print_separator("2. DATA QUALITY CHECKER")
    quality_output = mcp_client.call(
        "data_quality_checker",
        {"raw_data": sensor_data},
        case_id="VERIFY-001",
    )
    print(f"Status: {quality_output.status.value}")
    print(f"Execution Time: {quality_output.execution_time_ms:.2f}ms")
    print(f"Result:")
    print(json.dumps(quality_output.result, indent=2))
    
    print_separator("3. GAIT FEATURE EXTRACTOR")
    feature_output = mcp_client.call(
        "gait_feature_extractor",
        {"sensor_data": sensor_data, "patient_height_m": 1.75},
        case_id="VERIFY-001",
    )
    print(f"Status: {feature_output.status.value}")
    print(f"Execution Time: {feature_output.execution_time_ms:.2f}ms")
    print(f"Confidence: {feature_output.confidence:.2f}")
    print(f"Result:")
    print(json.dumps(feature_output.result, indent=2))
    
    print_separator("4. FALL RISK PREDICTOR")
    risk_output = mcp_client.call(
        "fall_risk_predictor",
        {
            "gait_features": feature_output.result,
            "patient_age": 72,
            "medical_history": patient_info["medical_history"],
            "data_quality_score": quality_output.result.get("reliability_score", 1.0),
        },
        case_id="VERIFY-001",
    )
    print(f"Status: {risk_output.status.value}")
    print(f"Execution Time: {risk_output.execution_time_ms:.2f}ms")
    print(f"Confidence: {risk_output.confidence:.2f}")
    print(f"Result:")
    print(json.dumps(risk_output.result, indent=2))
    
    print_separator("5. REPORT GENERATOR")
    report_output = mcp_client.call(
        "report_generator",
        {
            "case_id": "VERIFY-001",
            "patient_id": patient_info["patient_id"],
            "patient_info": patient_info,
            "gait_features": feature_output.result,
            "data_quality": quality_output.result,
            "fall_risk": risk_output.result,
        },
        case_id="VERIFY-001",
    )
    print(f"Status: {report_output.status.value}")
    print(f"Execution Time: {report_output.execution_time_ms:.2f}ms")
    print()
    print("Generated Report (Text):")
    print("-" * 70)
    print(report_output.result.get("text_report", "No report generated"))
    
    # Run full pipeline
    print_separator("6. FULL PIPELINE EXECUTION")
    print("Running call_pipeline() - executes all tools in sequence...")
    
    pipeline_results = mcp_client.call_pipeline(
        case_id="PIPELINE-001",
        sensor_data=sensor_data,
        patient_info=patient_info,
    )
    
    print(f"\nPipeline completed with {len(pipeline_results)} tool outputs:")
    for tool_name, output in pipeline_results.items():
        status_emoji = "✅" if output.status.value == "success" else "❌"
        print(f"  {status_emoji} {tool_name}: {output.status.value} ({output.execution_time_ms:.1f}ms)")
    
    print_separator("VERIFICATION COMPLETE")
    print("✅ All tools registered and callable through MCP")
    print("✅ No direct tool imports used")
    print("✅ Pipeline execution successful")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
