"""
Clinical War Room - UI Package
"""

from ui.demo_data import get_demo_scenarios
from ui.components import (
    render_patient_panel,
    render_agent_panel,
    render_debate_timeline,
    render_disagreement_meter,
    render_decision_panel,
    render_human_review_panel,
    get_action_color,
)


__all__ = [
    "get_demo_scenarios",
    "render_patient_panel",
    "render_agent_panel",
    "render_debate_timeline",
    "render_disagreement_meter",
    "render_decision_panel",
    "render_human_review_panel",
    "get_action_color",
]
