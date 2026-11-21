"""All Planning channels obey the frozen review, including action candidates."""
import json
import sqlite3

import pytest

from backend.tests.validation_human_review_business_quality import (
    env, quality_case, SIGNAL, EDIT, ACCESS, UNKNOWN,
)
from backend.planning.agent_planning_adapter import (
    build_planning_input_from_agent, AgentPlanningAdapterError,
)
from backend.planning.analysis_handoff import build_analysis_snapshot, AnalysisHandoffError


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("quality_case", [{
    "SignalAgent": {"proposed_actions": [{"actionType": "notify_dingtalk"}]},
    "PublicSafetyAgent": {"proposed_actions": [{"actionType": "notify_dingtalk"}]},
    # Dispatch was computed using the original domain outputs, before review.
    "DispatchAgent": {"suggestion": SIGNAL, "findings": [SIGNAL],
                      "proposed_actions": [{"actionType": "notify_dingtalk"}]},
    "AccidentAgent": {"proposed_actions": [{"actionType": "notify_wechat"}],
                      "assumptions": ["The injury report still requires verification"]},
}], indirect=True)
def test_reviewed_inputs_exclude_superseded_actions_and_preserve_independent_results(quality_case, reuse, edit):
    case = quality_case
    assert case["resolve"](EDIT if edit else None).status_code == 200
    value = build_planning_input_from_agent("handoff-event", "session", "quality-run", reuse_analysis=reuse)
    constraints = value.constraints
    final = constraints["agentFinalDecision"]
    projection = final["effectiveAnalysis"]
    assert all(SIGNAL not in json.dumps(channel, ensure_ascii=False) for channel in (
        constraints["agentRecommendations"], constraints["agentFindings"],
        constraints["agentRecommendationAudit"], final,
    ))
    assert {r["agentName"] for r in constraints["agentRecommendations"]} == {"AccidentAgent", "HumanReview"}
    assert [a["actionType"] for a in constraints["agentRecommendationAudit"]["accepted"]] == ["notify_wechat"]
    assert value.planMetadata["agentRecommendationAudit"] == constraints["agentRecommendationAudit"]
    assert ACCESS in final["fusionSummary"] and UNKNOWN in final["fusionSummary"]
    assert "The injury report still requires verification" in final["fusionSummary"]
    retained = next(o for o in projection["outputs"] if o["agent_name"] == "AccidentAgent")
    assert retained["task_id"] and retained["sourceVersion"] and retained["evidence_refs"] == ["chunk"]
    resolutions = [o for o in projection["outputs"] if o["agent_name"] == "HumanReview"]
    assert len(resolutions) == 3
    assert all(o["sourceTaskIds"] and o["sourceVersions"] and not o["proposed_actions"] for o in resolutions)
    assert final["humanReview"]["externalActionsApproved"] is False
    assert final["actionPlan"] == [] and len(case["calls"]) == 4
    if reuse:
        snapshot = constraints["analysisSnapshot"]
        assert snapshot["finalDecision"] == final
        assert json.loads(snapshot["agentOutputs"]["SourceFusion"]["summary"]) == final
    # Audit remains durable and unchanged, but is outside current Planning inputs.
    tasks = case["collaboration"].list_tasks("quality-run")
    original = next(t for t in tasks if t["agent_name"] == "SignalAgent")
    assert json.loads(original["output_snapshot"])["proposed_actions"] == [{"actionType": "notify_dingtalk"}]
    assert SIGNAL in original["output_snapshot"]


@pytest.mark.parametrize("reuse", [False, True])
def test_pending_review_cannot_fall_back_to_raw_planning_inputs(quality_case, reuse):
    with pytest.raises(AgentPlanningAdapterError) as error:
        build_planning_input_from_agent("handoff-event", "session", "quality-run", reuse_analysis=reuse)
    assert error.value.code == "analysis_review_invalid"


@pytest.mark.parametrize("legacy", [False, True])
def test_inconsistent_or_legacy_review_requires_fresh_analysis(quality_case, legacy):
    case = quality_case
    case["resolve"](EDIT)
    run = case["collaboration"].get_run("quality-run")
    final = json.loads(run["final_decision"])
    final.pop("effectiveAnalysis")
    with sqlite3.connect(case["db"]) as conn:
        conn.execute("UPDATE collaboration_runs SET final_decision=? WHERE run_id='quality-run'", (json.dumps(final),))
        if legacy:
            row = conn.execute("SELECT decision_json FROM collaboration_reviews WHERE run_id='quality-run'").fetchone()
            decision = json.loads(row[0])
            decision["finalDecision"] = final
            conn.execute("UPDATE collaboration_reviews SET decision_json=? WHERE run_id='quality-run'", (json.dumps(decision),))
    code = "analysis_review_projection_missing" if legacy else "analysis_review_invalid"
    for reuse in (False, True):
        with pytest.raises(AgentPlanningAdapterError) as error:
            build_planning_input_from_agent("handoff-event", "session", "quality-run", reuse_analysis=reuse)
        assert error.value.code == code
    with pytest.raises(AnalysisHandoffError) as error:
        build_analysis_snapshot("handoff-event", "session", "quality-run")
    assert error.value.code == code
