import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.recommendation.engine import generate_agent_recommendation


def test_tool_failure_recommendation():
    diagnosis = {
        "root_cause": "tool_execution_failure",
        "severity": "high",
    }

    failure_analysis = {
        "signals": ["tool_failure", "execution_failure"],
    }

    recommendation = generate_agent_recommendation(
        diagnosis,
        failure_analysis,
    )

    assert recommendation["recommendation_detected"] is True
    assert recommendation["priority"] == "high"
    assert recommendation["action"] == "retry_or_validate_tool"


def test_no_failure_recommendation():
    diagnosis = {
        "root_cause": "no_failure_detected",
        "severity": "low",
    }

    failure_analysis = {
        "signals": [],
    }

    recommendation = generate_agent_recommendation(
        diagnosis,
        failure_analysis,
    )

    assert recommendation["recommendation_detected"] is False
    assert recommendation["priority"] == "low"
    assert recommendation["action"] == "continue_monitoring"