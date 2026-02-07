"""
Clinical War Room - Fall Risk Model

Deterministic tool for predicting fall risk from gait features.
NO LLM usage. NO learning. Uses fixed rule-based model.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class FallRiskResult:
    """Fall risk prediction results."""
    fall_risk_probability: float  # 0-1
    risk_level: str  # "low", "moderate", "high", "critical"
    contributing_factors: List[str]
    protective_factors: List[str]
    confidence: float  # 0-1, confidence in this prediction
    
    def to_dict(self) -> dict:
        return {
            "fall_risk_probability": round(self.fall_risk_probability, 3),
            "risk_level": self.risk_level,
            "contributing_factors": self.contributing_factors,
            "protective_factors": self.protective_factors,
            "confidence": round(self.confidence, 3),
        }


# =============================================================================
# Normative Values for Gait Parameters
# Based on literature: normal ranges for healthy adults
# =============================================================================

NORMATIVE_VALUES = {
    "cadence": {
        "normal_min": 90,
        "normal_max": 120,
        "concern_low": 75,
        "concern_high": 140,
    },
    "gait_speed": {
        "normal_min": 1.0,
        "normal_max": 1.4,
        "concern_low": 0.6,
        "critical_low": 0.4,
    },
    "asymmetry_index": {
        "normal_max": 0.1,
        "concern": 0.2,
        "high_risk": 0.3,
    },
    "variability": {
        "normal_max": 0.05,
        "concern": 0.08,
        "high_risk": 0.12,
    },
    "stride_length": {
        "normal_min": 0.55,
        "normal_max": 0.85,
        "concern_low": 0.45,
    },
}

# =============================================================================
# Age-based adjustment factors
# =============================================================================

AGE_ADJUSTMENT = {
    "60-69": {"multiplier": 1.1, "speed_tolerance": 0.15},
    "70-79": {"multiplier": 1.25, "speed_tolerance": 0.25},
    "80+": {"multiplier": 1.4, "speed_tolerance": 0.35},
}


def _get_age_adjustment(age: Optional[int]) -> tuple:
    """Get risk multiplier and speed tolerance based on age."""
    if age is None:
        return (1.2, 0.2)  # Default moderate adjustment
    
    if age >= 80:
        adj = AGE_ADJUSTMENT["80+"]
    elif age >= 70:
        adj = AGE_ADJUSTMENT["70-79"]
    elif age >= 60:
        adj = AGE_ADJUSTMENT["60-69"]
    else:
        return (1.0, 0.0)  # No adjustment for younger adults
    
    return (adj["multiplier"], adj["speed_tolerance"])


def _calculate_gait_speed_risk(gait_speed: float, age: Optional[int]) -> tuple:
    """
    Calculate fall risk contribution from gait speed.
    
    Returns (risk_score, factors)
    """
    _, speed_tolerance = _get_age_adjustment(age)
    norms = NORMATIVE_VALUES["gait_speed"]
    
    adjusted_normal_min = norms["normal_min"] - speed_tolerance
    adjusted_concern_low = norms["concern_low"] - speed_tolerance
    adjusted_critical_low = norms["critical_low"] - speed_tolerance
    
    factors = []
    
    if gait_speed < adjusted_critical_low:
        risk = 0.9
        factors.append(f"Critically low gait speed ({gait_speed:.2f} m/s)")
    elif gait_speed < adjusted_concern_low:
        risk = 0.6
        factors.append(f"Low gait speed ({gait_speed:.2f} m/s)")
    elif gait_speed < adjusted_normal_min:
        risk = 0.3
        factors.append(f"Below-normal gait speed ({gait_speed:.2f} m/s)")
    elif gait_speed > norms["normal_max"]:
        risk = 0.2
        factors.append(f"High gait speed may indicate rushed walking")
    else:
        risk = 0.0
    
    return (risk, factors)


def _calculate_asymmetry_risk(asymmetry_index: float) -> tuple:
    """Calculate fall risk contribution from gait asymmetry."""
    norms = NORMATIVE_VALUES["asymmetry_index"]
    factors = []
    
    if asymmetry_index >= norms["high_risk"]:
        risk = 0.8
        factors.append(f"Severe gait asymmetry ({asymmetry_index:.2f})")
    elif asymmetry_index >= norms["concern"]:
        risk = 0.5
        factors.append(f"Moderate gait asymmetry ({asymmetry_index:.2f})")
    elif asymmetry_index >= norms["normal_max"]:
        risk = 0.2
        factors.append(f"Mild gait asymmetry ({asymmetry_index:.2f})")
    else:
        risk = 0.0
    
    return (risk, factors)


def _calculate_variability_risk(variability: float) -> tuple:
    """Calculate fall risk contribution from step variability."""
    norms = NORMATIVE_VALUES["variability"]
    factors = []
    
    if variability >= norms["high_risk"]:
        risk = 0.7
        factors.append(f"High step-to-step variability ({variability:.2f})")
    elif variability >= norms["concern"]:
        risk = 0.4
        factors.append(f"Elevated step-to-step variability ({variability:.2f})")
    elif variability >= norms["normal_max"]:
        risk = 0.15
        factors.append(f"Slightly increased variability ({variability:.2f})")
    else:
        risk = 0.0
    
    return (risk, factors)


def _identify_protective_factors(
    gait_speed: float,
    asymmetry_index: float,
    variability: float,
    cadence: float,
) -> List[str]:
    """Identify factors that reduce fall risk."""
    protective = []
    
    norms_speed = NORMATIVE_VALUES["gait_speed"]
    norms_asym = NORMATIVE_VALUES["asymmetry_index"]
    norms_var = NORMATIVE_VALUES["variability"]
    norms_cad = NORMATIVE_VALUES["cadence"]
    
    if norms_speed["normal_min"] <= gait_speed <= norms_speed["normal_max"]:
        protective.append("Normal gait speed")
    
    if asymmetry_index < norms_asym["normal_max"]:
        protective.append("Symmetric gait pattern")
    
    if variability < norms_var["normal_max"]:
        protective.append("Consistent step timing")
    
    if norms_cad["normal_min"] <= cadence <= norms_cad["normal_max"]:
        protective.append("Normal cadence")
    
    return protective


def predict_fall_risk(
    gait_features: Dict[str, Any],
    patient_age: Optional[int] = None,
    medical_history: Optional[List[str]] = None,
    data_quality_score: Optional[float] = None,
) -> FallRiskResult:
    """
    Predict fall risk from gait features using deterministic model.
    
    Args:
        gait_features: Dict with cadence, stride_length, gait_speed,
                      asymmetry_index, variability
        patient_age: Optional patient age for risk adjustment
        medical_history: Optional list of conditions affecting fall risk
        data_quality_score: Optional quality score (0-1) for confidence adjustment
        
    Returns:
        FallRiskResult with risk probability and factors
    """
    # Extract features with defaults
    gait_speed = float(gait_features.get("gait_speed", 0))
    asymmetry_index = float(gait_features.get("asymmetry_index", 0.5))
    variability = float(gait_features.get("variability", 0.5))
    cadence = float(gait_features.get("cadence", 0))
    stride_length = float(gait_features.get("stride_length", 0))
    step_count = int(gait_features.get("step_count", 0))
    
    # Check for sufficient data
    if step_count < 5 or gait_speed == 0:
        return FallRiskResult(
            fall_risk_probability=0.5,  # Unknown
            risk_level="unknown",
            contributing_factors=["Insufficient gait data for analysis"],
            protective_factors=[],
            confidence=0.1,
        )
    
    # Calculate risk components
    all_factors = []
    
    speed_risk, speed_factors = _calculate_gait_speed_risk(gait_speed, patient_age)
    all_factors.extend(speed_factors)
    
    asym_risk, asym_factors = _calculate_asymmetry_risk(asymmetry_index)
    all_factors.extend(asym_factors)
    
    var_risk, var_factors = _calculate_variability_risk(variability)
    all_factors.extend(var_factors)
    
    # Combine risks with weights
    # Speed is most predictive, followed by variability, then asymmetry
    base_risk = (
        speed_risk * 0.45 +
        var_risk * 0.30 +
        asym_risk * 0.25
    )
    
    # Apply age adjustment
    age_multiplier, _ = _get_age_adjustment(patient_age)
    adjusted_risk = base_risk * age_multiplier
    
    if patient_age and patient_age >= 75:
        all_factors.append("Age-related increased risk")
    
    # Apply medical history modifiers
    if medical_history:
        history_lower = [h.lower() for h in medical_history]
        
        if any(term in " ".join(history_lower) for term in ["stroke", "cva"]):
            adjusted_risk += 0.15
            all_factors.append("History of stroke")
        
        if any(term in " ".join(history_lower) for term in ["parkinson", "parkinsonian"]):
            adjusted_risk += 0.2
            all_factors.append("Parkinson's disease")
        
        if any(term in " ".join(history_lower) for term in ["diabetes", "diabetic"]):
            adjusted_risk += 0.1
            all_factors.append("Diabetes (neuropathy risk)")
        
        if any(term in " ".join(history_lower) for term in ["fall", "fallen"]):
            adjusted_risk += 0.25
            all_factors.append("Previous falls")
    
    # Clamp to 0-1 range
    final_risk = max(0.0, min(1.0, adjusted_risk))
    
    # Determine risk level
    if final_risk >= 0.7:
        risk_level = "critical"
    elif final_risk >= 0.5:
        risk_level = "high"
    elif final_risk >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Calculate confidence in prediction
    confidence = 0.8  # Base confidence
    
    if data_quality_score is not None:
        confidence *= data_quality_score
    
    if step_count < 20:
        confidence *= 0.7  # Reduced confidence with fewer steps
    
    if step_count > 50:
        confidence = min(1.0, confidence * 1.1)  # Slightly higher confidence
    
    # Identify protective factors
    protective = _identify_protective_factors(
        gait_speed, asymmetry_index, variability, cadence
    )
    
    return FallRiskResult(
        fall_risk_probability=final_risk,
        risk_level=risk_level,
        contributing_factors=all_factors if all_factors else ["No significant risk factors identified"],
        protective_factors=protective,
        confidence=max(0.0, min(1.0, confidence)),
    )
