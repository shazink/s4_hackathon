"""
Clinical War Room - Gait Feature Extractor

Deterministic tool for extracting gait features from IMU sensor data.
NO LLM usage. NO decision making. Evidence only.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
from statistics import mean, stdev


@dataclass
class GaitSample:
    """Single IMU sensor sample."""
    timestamp_ms: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


@dataclass
class GaitFeatures:
    """Extracted gait features."""
    cadence: float  # steps per minute
    stride_length: float  # meters
    gait_speed: float  # meters per second
    asymmetry_index: float  # 0-1, 0 = symmetric
    variability: float  # coefficient of variation
    step_count: int
    analysis_duration_ms: float
    
    def to_dict(self) -> dict:
        return {
            "cadence": round(self.cadence, 2),
            "stride_length": round(self.stride_length, 3),
            "gait_speed": round(self.gait_speed, 3),
            "asymmetry_index": round(self.asymmetry_index, 3),
            "variability": round(self.variability, 3),
            "step_count": self.step_count,
            "analysis_duration_ms": round(self.analysis_duration_ms, 2),
        }


def _detect_steps(samples: List[GaitSample], threshold: float = 1.2) -> List[int]:
    """
    Detect step events from accelerometer data.
    
    Uses simple peak detection on vertical acceleration magnitude.
    Returns indices of detected steps.
    """
    if len(samples) < 3:
        return []
    
    # Calculate acceleration magnitude
    magnitudes = [
        math.sqrt(s.accel_x**2 + s.accel_y**2 + s.accel_z**2)
        for s in samples
    ]
    
    # Simple peak detection
    steps = []
    avg_mag = mean(magnitudes)
    
    for i in range(1, len(magnitudes) - 1):
        # Peak if higher than neighbors and above threshold
        if (magnitudes[i] > magnitudes[i-1] and 
            magnitudes[i] > magnitudes[i+1] and
            magnitudes[i] > avg_mag * threshold):
            steps.append(i)
    
    # Filter out steps too close together (< 200ms apart typically)
    if len(steps) < 2:
        return steps
    
    filtered_steps = [steps[0]]
    min_step_interval_ms = 200
    
    for i in range(1, len(steps)):
        time_diff = samples[steps[i]].timestamp_ms - samples[steps[i-1]].timestamp_ms
        if time_diff >= min_step_interval_ms:
            filtered_steps.append(steps[i])
    
    return filtered_steps


def _calculate_step_intervals(samples: List[GaitSample], step_indices: List[int]) -> List[float]:
    """Calculate time intervals between consecutive steps in milliseconds."""
    if len(step_indices) < 2:
        return []
    
    intervals = []
    for i in range(1, len(step_indices)):
        interval = samples[step_indices[i]].timestamp_ms - samples[step_indices[i-1]].timestamp_ms
        intervals.append(interval)
    
    return intervals


def extract_gait_features(
    samples: List[GaitSample],
    patient_height_m: float = 1.7,
    sampling_rate_hz: float = 100.0
) -> GaitFeatures:
    """
    Extract gait features from IMU sensor samples.
    
    Args:
        samples: List of IMU sensor samples
        patient_height_m: Patient height in meters (for stride estimation)
        sampling_rate_hz: Sensor sampling rate
        
    Returns:
        GaitFeatures dataclass with computed metrics
    """
    if len(samples) < 10:
        return GaitFeatures(
            cadence=0.0,
            stride_length=0.0,
            gait_speed=0.0,
            asymmetry_index=1.0,  # Maximum asymmetry = unknown
            variability=1.0,  # Maximum variability = unknown
            step_count=0,
            analysis_duration_ms=0.0,
        )
    
    # Calculate analysis duration
    duration_ms = samples[-1].timestamp_ms - samples[0].timestamp_ms
    duration_sec = duration_ms / 1000.0
    
    # Detect steps
    step_indices = _detect_steps(samples)
    step_count = len(step_indices)
    
    if step_count < 2:
        return GaitFeatures(
            cadence=0.0,
            stride_length=0.0,
            gait_speed=0.0,
            asymmetry_index=1.0,
            variability=1.0,
            step_count=step_count,
            analysis_duration_ms=duration_ms,
        )
    
    # Calculate step intervals
    intervals = _calculate_step_intervals(samples, step_indices)
    
    # Cadence (steps per minute)
    avg_interval_sec = mean(intervals) / 1000.0
    cadence = 60.0 / avg_interval_sec if avg_interval_sec > 0 else 0.0
    
    # Stride length estimation (empirical formula based on height and cadence)
    # stride_length ≈ height * 0.415 at comfortable pace
    stride_factor = 0.415 * (1.0 + (cadence - 100) * 0.002)  # Adjust for pace
    stride_factor = max(0.35, min(0.50, stride_factor))  # Clamp to realistic range
    stride_length = patient_height_m * stride_factor
    
    # Gait speed (m/s)
    gait_speed = stride_length * cadence / 60.0
    
    # Asymmetry index (compare odd vs even step intervals)
    if len(intervals) >= 4:
        odd_intervals = intervals[0::2]  # Left steps (assumed)
        even_intervals = intervals[1::2]  # Right steps (assumed)
        
        avg_odd = mean(odd_intervals) if odd_intervals else 0
        avg_even = mean(even_intervals) if even_intervals else 0
        
        if avg_odd + avg_even > 0:
            asymmetry_index = abs(avg_odd - avg_even) / (avg_odd + avg_even)
        else:
            asymmetry_index = 0.0
    else:
        asymmetry_index = 0.5  # Unknown with insufficient data
    
    # Variability (coefficient of variation of step intervals)
    if len(intervals) >= 3:
        std_interval = stdev(intervals)
        avg_interval = mean(intervals)
        variability = std_interval / avg_interval if avg_interval > 0 else 0.0
    else:
        variability = 0.5  # Unknown with insufficient data
    
    return GaitFeatures(
        cadence=cadence,
        stride_length=stride_length,
        gait_speed=gait_speed,
        asymmetry_index=asymmetry_index,
        variability=variability,
        step_count=step_count,
        analysis_duration_ms=duration_ms,
    )


def parse_sensor_data(raw_data: List[Dict[str, Any]]) -> List[GaitSample]:
    """Parse raw sensor data dictionaries into GaitSample objects."""
    samples = []
    for i, row in enumerate(raw_data):
        try:
            sample = GaitSample(
                timestamp_ms=float(row.get("timestamp_ms", i * 10)),
                accel_x=float(row.get("accel_x", row.get("ax", 0))),
                accel_y=float(row.get("accel_y", row.get("ay", 0))),
                accel_z=float(row.get("accel_z", row.get("az", 0))),
                gyro_x=float(row.get("gyro_x", row.get("gx", 0))) if "gyro_x" in row or "gx" in row else None,
                gyro_y=float(row.get("gyro_y", row.get("gy", 0))) if "gyro_y" in row or "gy" in row else None,
                gyro_z=float(row.get("gyro_z", row.get("gz", 0))) if "gyro_z" in row or "gz" in row else None,
            )
            samples.append(sample)
        except (ValueError, TypeError):
            continue  # Skip malformed samples
    
    return samples
