"""
Clinical War Room - Data Quality Checker

Deterministic tool for assessing quality of gait sensor data.
NO LLM usage. NO decision making. Evidence only.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import math
from statistics import mean, stdev


@dataclass
class DataQualityResult:
    """Data quality assessment results."""
    missing_data_percentage: float  # 0-100
    noise_score: float  # 0-1, higher = noisier
    reliability_score: float  # 0-1, higher = more reliable
    total_samples: int
    valid_samples: int
    issues: List[str]
    
    def to_dict(self) -> dict:
        return {
            "missing_data_percentage": round(self.missing_data_percentage, 2),
            "noise_score": round(self.noise_score, 3),
            "reliability_score": round(self.reliability_score, 3),
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "issues": self.issues,
        }


def _calculate_missing_data(samples: List[Dict[str, Any]], required_fields: List[str]) -> float:
    """Calculate percentage of missing required fields."""
    if not samples:
        return 100.0
    
    total_fields = len(samples) * len(required_fields)
    missing = 0
    
    for sample in samples:
        for field in required_fields:
            if field not in sample or sample[field] is None:
                missing += 1
    
    return (missing / total_fields) * 100 if total_fields > 0 else 100.0


def _calculate_noise_score(values: List[float]) -> float:
    """
    Calculate noise score based on signal-to-noise ratio.
    
    Uses high-frequency variation detection.
    Returns 0-1, where 1 is extremely noisy.
    """
    if len(values) < 3:
        return 0.5  # Unknown with insufficient data
    
    # Calculate first-order differences
    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    
    if not diffs:
        return 0.0
    
    avg_diff = mean(diffs)
    
    # Calculate expected natural variation based on signal range
    signal_range = max(values) - min(values)
    
    if signal_range < 0.01:
        return 0.0  # Effectively constant signal
    
    # Noise ratio: how much of the change is high-frequency noise
    noise_ratio = avg_diff / signal_range
    
    # Normalize to 0-1 range
    noise_score = min(1.0, noise_ratio * 2)
    
    return noise_score


def _detect_outliers(values: List[float], z_threshold: float = 3.0) -> int:
    """Count outliers using z-score method."""
    if len(values) < 3:
        return 0
    
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0
    
    if std < 0.001:
        return 0
    
    outliers = sum(1 for v in values if abs(v - avg) / std > z_threshold)
    return outliers


def _check_sampling_consistency(timestamps: List[float]) -> tuple:
    """Check if sampling rate is consistent."""
    if len(timestamps) < 2:
        return (True, 0.0, [])
    
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    if not intervals:
        return (True, 0.0, [])
    
    avg_interval = mean(intervals)
    
    if len(intervals) > 1:
        interval_std = stdev(intervals)
        cv = interval_std / avg_interval if avg_interval > 0 else 0
    else:
        cv = 0.0
    
    issues = []
    
    # Check for gaps (intervals > 3x average)
    gaps = sum(1 for i in intervals if i > avg_interval * 3)
    if gaps > 0:
        issues.append(f"Found {gaps} sampling gaps (>3x normal interval)")
    
    # Check for negative intervals (time going backwards)
    negatives = sum(1 for i in intervals if i < 0)
    if negatives > 0:
        issues.append(f"Found {negatives} negative time intervals")
    
    is_consistent = cv < 0.2 and gaps == 0 and negatives == 0
    
    return (is_consistent, cv, issues)


def check_data_quality(
    raw_data: List[Dict[str, Any]],
    required_fields: List[str] = None
) -> DataQualityResult:
    """
    Assess quality of raw gait sensor data.
    
    Args:
        raw_data: List of sensor sample dictionaries
        required_fields: Fields required in each sample
        
    Returns:
        DataQualityResult with quality metrics
    """
    if required_fields is None:
        required_fields = ["timestamp_ms", "accel_x", "accel_y", "accel_z"]
    
    if not raw_data:
        return DataQualityResult(
            missing_data_percentage=100.0,
            noise_score=1.0,
            reliability_score=0.0,
            total_samples=0,
            valid_samples=0,
            issues=["No data provided"],
        )
    
    issues = []
    total_samples = len(raw_data)
    
    # Check missing data
    missing_pct = _calculate_missing_data(raw_data, required_fields)
    
    if missing_pct > 50:
        issues.append(f"High missing data: {missing_pct:.1f}%")
    elif missing_pct > 20:
        issues.append(f"Moderate missing data: {missing_pct:.1f}%")
    
    # Extract valid samples
    valid_samples = []
    for sample in raw_data:
        if all(field in sample and sample[field] is not None for field in required_fields):
            valid_samples.append(sample)
    
    valid_count = len(valid_samples)
    
    if valid_count < 10:
        return DataQualityResult(
            missing_data_percentage=missing_pct,
            noise_score=1.0,
            reliability_score=0.0,
            total_samples=total_samples,
            valid_samples=valid_count,
            issues=issues + ["Insufficient valid samples for analysis"],
        )
    
    # Calculate noise scores for each axis
    accel_x = [float(s.get("accel_x", 0)) for s in valid_samples]
    accel_y = [float(s.get("accel_y", 0)) for s in valid_samples]
    accel_z = [float(s.get("accel_z", 0)) for s in valid_samples]
    
    noise_x = _calculate_noise_score(accel_x)
    noise_y = _calculate_noise_score(accel_y)
    noise_z = _calculate_noise_score(accel_z)
    
    avg_noise = (noise_x + noise_y + noise_z) / 3
    
    if avg_noise > 0.7:
        issues.append("High sensor noise detected")
    elif avg_noise > 0.4:
        issues.append("Moderate sensor noise detected")
    
    # Check for outliers
    outliers_x = _detect_outliers(accel_x)
    outliers_y = _detect_outliers(accel_y)
    outliers_z = _detect_outliers(accel_z)
    total_outliers = outliers_x + outliers_y + outliers_z
    
    outlier_pct = (total_outliers / (valid_count * 3)) * 100
    if outlier_pct > 5:
        issues.append(f"High outlier rate: {outlier_pct:.1f}%")
    
    # Check sampling consistency
    timestamps = [float(s.get("timestamp_ms", 0)) for s in valid_samples]
    is_consistent, timing_cv, timing_issues = _check_sampling_consistency(timestamps)
    issues.extend(timing_issues)
    
    if not is_consistent:
        issues.append(f"Inconsistent sampling rate (CV={timing_cv:.2f})")
    
    # Calculate overall reliability score
    # Based on: missing data, noise, outliers, timing consistency
    reliability = 1.0
    reliability -= min(0.4, missing_pct / 100)  # Max 0.4 penalty
    reliability -= min(0.3, avg_noise * 0.5)  # Max 0.3 penalty
    reliability -= min(0.2, outlier_pct / 20)  # Max 0.2 penalty
    reliability -= 0.1 if not is_consistent else 0  # 0.1 penalty
    
    reliability = max(0.0, min(1.0, reliability))
    
    return DataQualityResult(
        missing_data_percentage=missing_pct,
        noise_score=avg_noise,
        reliability_score=reliability,
        total_samples=total_samples,
        valid_samples=valid_count,
        issues=issues,
    )
