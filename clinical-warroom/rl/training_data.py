"""
Clinical War Room - Training Data Generator

Generates diverse clinical cases for RL training.
Each case has labeled ground truth for supervised/imitation learning.
"""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path


class RiskCategory(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class OptimalAction(Enum):
    EXECUTE = "EXECUTE"
    ESCALATE = "ESCALATE"
    REFUSE = "REFUSE"
    REQUEST_MORE_DATA = "REQUEST_MORE_DATA"


@dataclass
class TrainingCase:
    """A labeled training case for RL."""
    case_id: str
    
    # Patient features
    age: int
    gender: str
    medical_history: str
    chief_complaint: str
    
    # Gait metrics (MCP outputs)
    stride_length: float
    gait_speed: float
    symmetry_index: float
    fall_risk_score: float
    
    # Agent analysis
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    disagreement_score: float
    has_veto: bool
    
    # Risk features
    risk_category: str
    ground_truth_risk: float
    
    # Ground truth label
    optimal_action: str
    reward_if_correct: float
    penalty_if_wrong: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Templates for generating diverse cases
MEDICAL_HISTORIES = [
    "No significant medical history",
    "History of falls, 2 in past year",
    "History of falls, 4 in past 6 months",
    "Type 2 diabetes, controlled with medication",
    "Uncontrolled diabetes, A1C > 9",
    "Hypertension, well controlled",
    "Severe hypertension, recent hospitalization",
    "Parkinson's disease, early stage",
    "Parkinson's disease, advanced",
    "Recent hip replacement surgery",
    "Multiple sclerosis",
    "Prior stroke with residual weakness",
    "Peripheral neuropathy",
    "Vestibular disorder",
    "Severe arthritis limiting mobility",
    "Dementia, mild cognitive impairment",
    "History of syncope",
    "Osteoporosis with prior fracture",
    "Morbid obesity",
    "Cardiac arrhythmia",
]

CHIEF_COMPLAINTS = [
    "Annual wellness checkup",
    "Gait instability reported by family",
    "Frequent falls at home",
    "Difficulty walking, progressive",
    "Balance problems when turning",
    "Fear of falling limiting activity",
    "Dizziness when standing",
    "Weakness in legs",
    "Foot drop noted",
    "Pre-surgical fitness assessment",
    "Post-surgery rehabilitation evaluation",
    "Medication review for fall risk",
]


def generate_low_risk_case(case_num: int) -> TrainingCase:
    """Generate a low-risk case - optimal action is EXECUTE."""
    return TrainingCase(
        case_id=f"TRAIN-LOW-{case_num:03d}",
        age=random.randint(45, 65),
        gender=random.choice(["Male", "Female"]),
        medical_history=random.choice([
            "No significant medical history",
            "Type 2 diabetes, controlled with medication",
            "Hypertension, well controlled",
        ]),
        chief_complaint=random.choice([
            "Annual wellness checkup",
            "Pre-surgical fitness assessment",
        ]),
        stride_length=random.uniform(0.65, 0.80),
        gait_speed=random.uniform(1.1, 1.4),
        symmetry_index=random.uniform(0.88, 0.95),
        fall_risk_score=random.uniform(0.05, 0.25),
        avg_confidence=random.uniform(0.78, 0.92),
        min_confidence=random.uniform(0.70, 0.80),
        max_confidence=random.uniform(0.85, 0.95),
        disagreement_score=random.uniform(0.05, 0.15),
        has_veto=False,
        risk_category="low",
        ground_truth_risk=random.uniform(0.05, 0.25),
        optimal_action="EXECUTE",
        reward_if_correct=2.0,
        penalty_if_wrong=-0.5,
    )


def generate_moderate_risk_case(case_num: int) -> TrainingCase:
    """Generate a moderate-risk case - optimal action is ESCALATE."""
    return TrainingCase(
        case_id=f"TRAIN-MOD-{case_num:03d}",
        age=random.randint(65, 78),
        gender=random.choice(["Male", "Female"]),
        medical_history=random.choice([
            "History of falls, 2 in past year",
            "Parkinson's disease, early stage",
            "Peripheral neuropathy",
            "Recent hip replacement surgery",
        ]),
        chief_complaint=random.choice([
            "Gait instability reported by family",
            "Balance problems when turning",
            "Weakness in legs",
        ]),
        stride_length=random.uniform(0.50, 0.65),
        gait_speed=random.uniform(0.8, 1.0),
        symmetry_index=random.uniform(0.75, 0.85),
        fall_risk_score=random.uniform(0.35, 0.55),
        avg_confidence=random.uniform(0.60, 0.75),
        min_confidence=random.uniform(0.45, 0.60),
        max_confidence=random.uniform(0.70, 0.85),
        disagreement_score=random.uniform(0.25, 0.45),
        has_veto=False,
        risk_category="moderate",
        ground_truth_risk=random.uniform(0.35, 0.55),
        optimal_action="ESCALATE",
        reward_if_correct=1.5,
        penalty_if_wrong=-1.0,
    )


def generate_high_risk_case(case_num: int) -> TrainingCase:
    """Generate a high-risk case - optimal action is ESCALATE or REFUSE."""
    return TrainingCase(
        case_id=f"TRAIN-HIGH-{case_num:03d}",
        age=random.randint(75, 90),
        gender=random.choice(["Male", "Female"]),
        medical_history=random.choice([
            "History of falls, 4 in past 6 months",
            "Parkinson's disease, advanced",
            "Prior stroke with residual weakness",
            "Vestibular disorder",
            "Dementia, mild cognitive impairment",
        ]),
        chief_complaint=random.choice([
            "Frequent falls at home",
            "Difficulty walking, progressive",
            "Fear of falling limiting activity",
            "Dizziness when standing",
        ]),
        stride_length=random.uniform(0.35, 0.50),
        gait_speed=random.uniform(0.5, 0.8),
        symmetry_index=random.uniform(0.60, 0.75),
        fall_risk_score=random.uniform(0.60, 0.80),
        avg_confidence=random.uniform(0.50, 0.65),
        min_confidence=random.uniform(0.35, 0.50),
        max_confidence=random.uniform(0.60, 0.75),
        disagreement_score=random.uniform(0.40, 0.60),
        has_veto=random.choice([True, False]),
        risk_category="high",
        ground_truth_risk=random.uniform(0.60, 0.80),
        optimal_action="ESCALATE",
        reward_if_correct=1.5,
        penalty_if_wrong=-2.0,
    )


def generate_critical_risk_case(case_num: int) -> TrainingCase:
    """Generate a critical-risk case - optimal action is REFUSE."""
    return TrainingCase(
        case_id=f"TRAIN-CRIT-{case_num:03d}",
        age=random.randint(80, 95),
        gender=random.choice(["Male", "Female"]),
        medical_history=random.choice([
            "History of falls, 4 in past 6 months",
            "Parkinson's disease, advanced",
            "Multiple sclerosis",
            "Severe arthritis limiting mobility",
            "History of syncope",
            "Osteoporosis with prior fracture",
        ]),
        chief_complaint=random.choice([
            "Frequent falls at home",
            "Cannot walk without assistance",
            "Acute gait deterioration",
        ]),
        stride_length=random.uniform(0.20, 0.35),
        gait_speed=random.uniform(0.2, 0.5),
        symmetry_index=random.uniform(0.45, 0.60),
        fall_risk_score=random.uniform(0.80, 0.98),
        avg_confidence=random.uniform(0.75, 0.90),  # Agents confident it's dangerous
        min_confidence=random.uniform(0.65, 0.80),
        max_confidence=random.uniform(0.85, 0.95),
        disagreement_score=random.uniform(0.10, 0.25),  # High consensus on danger
        has_veto=True,
        risk_category="critical",
        ground_truth_risk=random.uniform(0.85, 0.98),
        optimal_action="REFUSE",
        reward_if_correct=2.0,
        penalty_if_wrong=-10.0,  # Very bad to execute on critical risk
    )


def generate_incomplete_data_case(case_num: int) -> TrainingCase:
    """Generate a case with incomplete data - optimal action is REQUEST_MORE_DATA."""
    return TrainingCase(
        case_id=f"TRAIN-DATA-{case_num:03d}",
        age=random.randint(50, 80),
        gender=random.choice(["Male", "Female"]),
        medical_history="Limited records available",
        chief_complaint=random.choice([
            "Gait assessment requested",
            "New patient evaluation",
        ]),
        stride_length=0.0,  # Missing
        gait_speed=0.0,  # Missing
        symmetry_index=0.0,  # Missing
        fall_risk_score=random.uniform(0.30, 0.50),  # Uncertain
        avg_confidence=random.uniform(0.35, 0.50),  # Low confidence
        min_confidence=random.uniform(0.20, 0.35),
        max_confidence=random.uniform(0.45, 0.60),
        disagreement_score=random.uniform(0.50, 0.70),  # High disagreement
        has_veto=False,
        risk_category="moderate",
        ground_truth_risk=random.uniform(0.30, 0.50),
        optimal_action="REQUEST_MORE_DATA",
        reward_if_correct=1.0,
        penalty_if_wrong=-0.5,
    )


def generate_training_dataset(
    n_low: int = 15,
    n_moderate: int = 15,
    n_high: int = 10,
    n_critical: int = 7,
    n_incomplete: int = 8,
    seed: Optional[int] = 42,
) -> List[TrainingCase]:
    """
    Generate a complete training dataset.
    
    Default: 55 cases total with distribution:
    - 15 low risk (EXECUTE)
    - 15 moderate risk (ESCALATE)
    - 10 high risk (ESCALATE)
    - 7 critical risk (REFUSE)
    - 8 incomplete data (REQUEST_MORE_DATA)
    """
    if seed:
        random.seed(seed)
    
    cases = []
    
    for i in range(n_low):
        cases.append(generate_low_risk_case(i + 1))
    
    for i in range(n_moderate):
        cases.append(generate_moderate_risk_case(i + 1))
    
    for i in range(n_high):
        cases.append(generate_high_risk_case(i + 1))
    
    for i in range(n_critical):
        cases.append(generate_critical_risk_case(i + 1))
    
    for i in range(n_incomplete):
        cases.append(generate_incomplete_data_case(i + 1))
    
    # Shuffle
    random.shuffle(cases)
    
    return cases


def save_training_dataset(cases: List[TrainingCase], path: str):
    """Save training dataset to JSON."""
    data = [c.to_dict() for c in cases]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_training_dataset(path: str) -> List[TrainingCase]:
    """Load training dataset from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [TrainingCase(**c) for c in data]


if __name__ == "__main__":
    # Generate and save training data
    cases = generate_training_dataset()
    
    output_dir = Path(__file__).parent.parent / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "training_cases.json"
    save_training_dataset(cases, str(output_path))
    
    print(f"Generated {len(cases)} training cases:")
    action_counts = {}
    for c in cases:
        action_counts[c.optimal_action] = action_counts.get(c.optimal_action, 0) + 1
    
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")
    
    print(f"\nSaved to: {output_path}")
