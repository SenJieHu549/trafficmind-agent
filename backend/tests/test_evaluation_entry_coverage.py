"""Entry coverage and non-vacuous assertions, with real isolated child workers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.evaluation.v2.checks import evaluate
from backend.evaluation.v2.runner import load_dataset, run_suite


DATASET = Path(__file__).resolve().parents[1] / "evaluation_data/trafficmind_eval_v2_entry_coverage.json"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root, summary = run_suite(DATASET, tmp_path_factory.mktemp("entry-coverage"), deadline=180)
    return root, summary


def saved(suite, case_id):
    return json.loads((suite[0] / f"{case_id}-llm_tools/record.json").read_text(encoding="utf-8"))


def failures(case_id, record):
    case = next(c for c in load_dataset(DATASET)["cases"] if c["caseId"] == case_id)
    return {c["name"] for c in evaluate(case, record) if not c["passed"]}


@pytest.mark.parametrize("case_id,stage", [("M01", "completed"), ("M02", "completed"),
    ("R01", "awaiting_review"), ("R02", "completed"), ("R03", "completed"), ("R04", "review_rejected")])
def test_entry_cases_reach_expected_stage(suite, case_id, stage):
    r = saved(suite, case_id)
    assert r["passed"], r["checks"]
    assert r["workflowStatus"] == stage
    assert r["profile"]["entryPoint"] == "routed_api"
    assert r["memoryRecall"]["run_id"] == r["ids"]["collaborationRunId"]
    assert r["llmPathCompleted"] and not r.get("error")
    if case_id.startswith("R"):
        assert r["profile"]["scriptedDomainRoles"] == ["PublicSafetyAgent", "SignalAgent"]
        assert r["humanReview"]["modelRequestsBefore"] == r["humanReview"]["modelRequestsAfter"]


def test_summary_distinguishes_acceptance_from_business_completion(suite):
    summary = suite[1]
    assert summary["passed"] == summary["total"] == 6
    assert summary["workflowCompleted"] == 4
    assert summary["awaitingHumanReview"] == summary["humanReviewRejected"] == 1


@pytest.mark.parametrize("missing", ["modelMemoryContexts", "memoryDelivery", "memoryRecall"])
def test_memory_display_or_trace_alone_cannot_pass(suite, missing):
    r = saved(suite, "M01")
    r[missing] = [] if missing == "modelMemoryContexts" else {}
    assert failures("M01", r)


def test_foreign_memory_in_actual_input_is_detected(suite):
    r = saved(suite, "M01")
    r["modelMemoryContexts"][0]["items"].append({"memoryId": "memory-foreign-session"})
    assert "forbidden_memory_not_delivered" in failures("M01", r)


def test_superseded_parallel_action_is_detected(suite):
    r = saved(suite, "R03")
    r["plan"]["constraints"]["agentRecommendationAudit"]["accepted"].append({"actionType": "notify_dingtalk"})
    assert "superseded_text_excluded:notify_dingtalk" in failures("R03", r)


def test_a_waiting_run_is_not_accepted_as_completed(suite):
    r = saved(suite, "R01")
    r["collaboration"]["status"] = "completed"
    assert "expected_collaboration_stage" in failures("R01", r)


def test_role_rerun_after_review_is_detected(suite):
    r = saved(suite, "R03")
    task = next(t for t in r["humanReview"]["tasksAfter"] if t["agent_name"] == "AccidentAgent")
    task["attempt"] += 1
    assert "review_did_not_rerun_roles_or_models" in failures("R03", r)


def test_missing_review_cannot_pass_a_declared_review_task(suite):
    r = saved(suite, "R02")
    r.pop("humanReview")
    assert "expected_human_review_status" in failures("R02", r)


def test_entry_memory_and_review_support_strict_replay(suite, tmp_path):
    _, summary = run_suite(DATASET, tmp_path, case_ids=["M01", "R03"], transport="replay", replay_from=suite[0], deadline=180)
    assert summary["passed"] == summary["total"] == 2, summary
    assert summary["comparisonPassed"], summary


def test_scripted_domain_or_routing_fixtures_are_rejected_before_live_execution(tmp_path):
    with pytest.raises(ValueError, match="fixtures_forbidden_in_live"):
        run_suite(DATASET, tmp_path, transport="live", case_ids=["R01"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["entry", "missing_entry", "review", "scope"])
def test_dataset_validation_rejects_uncovered_or_invalid_fixture_configuration(tmp_path, change):
    dataset = deepcopy(load_dataset(DATASET))
    case = dataset["cases"][0]
    if change == "entry": case["entryPoint"] = "uncovered"
    elif change == "missing_entry": case.pop("entryPoint")
    elif change == "review": case["reviewDecision"] = {"action": "approve_everything"}
    else: case["memoryItems"][0]["fixtureScope"] = "arbitrary"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError): load_dataset(path)
