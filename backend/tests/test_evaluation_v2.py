"""Real-chain acceptance and strict replay; no live API / model downloads."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.evaluation.v2.providers import EvaluationClient, content_hash, safe_value
from backend.evaluation.v2.runner import DEFAULT_DATASET, load_dataset, run_suite


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root, summary = run_suite(DEFAULT_DATASET, tmp_path_factory.mktemp("evaluation-v2"))
    return root, summary


@pytest.mark.parametrize("case_id,status", [("E01", "completed"), ("E02", "completed"),
    ("E03", "rejected"), ("E04", "failed"), ("E05", "planning_blocked")])
def test_real_chain_expected_behavior(suite, case_id, status):
    root, _ = suite
    record = json.loads((root / f"{case_id}-llm_tools/record.json").read_text(encoding="utf-8"))
    assert record["passed"], record["checks"]
    assert record["workflowStatus"] == status
    if case_id != "E05":
        assert record["llmPathCompleted"] and not record["degraded"]
        assert len(record["runtimeTrace"]["llmCalls"]) == 3
        assert record["runtimeTrace"]["toolRounds"] == 2
    else:
        assert not record["llmPathCompleted"] and record["actualMode"] == "rule_fallback"


def test_replay_same_inputs_recreates_real_workflow_outcomes(suite, tmp_path):
    source, _ = suite
    root, report = run_suite(DEFAULT_DATASET, tmp_path, transport="replay", replay_from=source)
    assert report["passed"] == report["total"] == 5
    assert report["comparisonPassed"]
    assert all(r["behaviorMatchesReference"] for r in report["results"])


@pytest.mark.parametrize("mode", ["rule", "llm_context"])
def test_modes_use_same_entry_points_without_extra_model_or_tools(mode, tmp_path):
    root, report = run_suite(DEFAULT_DATASET, tmp_path, modes=[mode], case_ids=["E02"])
    assert report["passed"] == (mode == "llm_context"), report
    record = json.loads((root / f"E02-{mode}/record.json").read_text(encoding="utf-8"))
    assert record["actualMode"] == mode and not record["runtimeTrace"]["toolCalls"]
    assert bool(record["modelRequests"]) == (mode != "rule")


def test_rule_cannot_pass_a_goal_requiring_an_unproduced_candidate(tmp_path):
    _, report = run_suite(DEFAULT_DATASET, tmp_path, modes=["rule"], case_ids=["E02"])
    assert report["passed"] == 0
    assert "required_candidate_generated" in report["results"][0]["failedChecks"]


def test_strict_replay_rejects_changed_model_request(tmp_path):
    request = {"messages": [{"role": "user", "content": "original"}], "tools": [], "allow_tools": False, "timeout": 90}
    cassette = tmp_path / "cassette.json"
    cassette.write_text(json.dumps([{"requestHash": content_hash(request), "reply": {"message": {"content": "{}"}}}]), encoding="utf-8")
    client = EvaluationClient("replay", {}, cassette=cassette)
    request["messages"][0]["content"] = "changed"
    with pytest.raises(RuntimeError, match="replay_request_mismatch"):
        asyncio.run(client.complete(**request))


def test_dataset_validation_blocks_path_traversal_and_duplicate_cases(tmp_path):
    data = load_dataset(DEFAULT_DATASET)
    for change in ("../outside", "E02"):
        changed = deepcopy(data)
        changed["cases"][0]["caseId"] = change
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid_case_id"): load_dataset(path)


def test_public_records_remove_credentials_but_preserve_usage(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-fixture-credential")
    result = safe_value({"api_key": "private-fixture-credential", "usage": {"total_tokens": 42},
                         "text": "provider included private-fixture-credential"})
    assert "private-fixture-credential" not in json.dumps(result)
    assert result["usage"]["total_tokens"] == 42


def test_action_registry_has_no_notification_or_dispatch_side_effects():
    from backend.evaluation.v2.sandbox import action_registry
    registry = action_registry([])
    from backend.workflow.action_execution import CreateDispatchTaskExecutor, SendNotificationExecutor
    assert not isinstance(registry.get("create_dispatch_task"), CreateDispatchTaskExecutor)
    assert not isinstance(registry.get("send_notification"), SendNotificationExecutor)
    assert registry.get("notify_wechat").action_type == "notify_wechat"
    with pytest.raises(RuntimeError, match="evaluation_action_not_allowed"): registry.get("send_email")


def test_unauthorized_or_missing_source_evidence_is_not_a_pass(suite):
    from backend.evaluation.v2.checks import evaluate
    root, _ = suite
    record = json.loads((root / "E01-llm_tools/record.json").read_text(encoding="utf-8"))
    record["evidenceSnapshot"]["citedEvidenceIds"] = ["forged-source"]
    scores = evaluate(load_dataset(DEFAULT_DATASET)["cases"][0], record)
    assert not next(c for c in scores if c["name"] == "evidence_refs_exist")["passed"]


def test_unrelated_early_failure_is_not_mistaken_for_injected_provider_failure(suite):
    from backend.evaluation.v2.checks import evaluate
    root, _ = suite
    record = json.loads((root / "E05-llm_tools/record.json").read_text(encoding="utf-8"))
    record.update(modelRequests=0, faultInjectionObserved=False)
    record["runtimeTrace"]["llmCalls"] = []
    scores = evaluate(load_dataset(DEFAULT_DATASET)["cases"][-1], record)
    assert not next(c for c in scores if c["name"] == "injected_fault_was_reached")["passed"]
