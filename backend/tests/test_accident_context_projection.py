"""Regression coverage for event fields consumed through the role projection."""

import asyncio

import pytest

from backend.agent.collaboration.context_projection import project_context_for_agent
from backend.agent.collaboration.executor import _call_agent_function


def analyze_projected_event(event):
    context = project_context_for_agent(
        {"normalized_event": event}, "AccidentAgent"
    )
    return context, asyncio.run(_call_agent_function("AccidentAgent", context))


@pytest.mark.parametrize("duration", [0, 60, 1860])
def test_known_duration_reaches_accident_analysis(duration):
    context, result = analyze_projected_event(
        {"eventType": "accident", "roadName": "test road", "duration": duration}
    )

    assert context["duration"] == duration
    assert not any("持续时间未知" in finding for finding in result["findings"])
    if duration == 1860:
        assert "已持续 31min，建议优先清障" in result["findings"]


@pytest.mark.parametrize("duration_fields", [{}, {"duration": None}])
def test_unknown_duration_stays_unknown(duration_fields):
    context, result = analyze_projected_event(
        {"eventType": "accident", "roadName": "test road", **duration_fields}
    )

    assert context.get("duration") is None
    assert any("持续时间未知" in finding for finding in result["findings"])


@pytest.mark.parametrize("risk_level", ["高风险", "重大风险"])
def test_risk_level_reaches_multi_department_rule(risk_level):
    context, result = analyze_projected_event(
        {
            "eventType": "accident",
            "roadName": "test road",
            "duration": 60,
            "riskLevel": risk_level,
            "unrelatedInternalState": "must not be projected",
        }
    )

    assert context["riskLevel"] == risk_level
    assert f"风险 {risk_level}，需多部门联动" in result["findings"]
    assert result["urgency"] == "critical"
    assert "unrelatedInternalState" not in context
