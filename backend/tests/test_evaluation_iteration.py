"""Campaign metrics preserve failure denominators and separate Judge errors."""
import json

import pytest

from backend.evaluation.v2.iteration import metrics, paired, trial_row, report, write


def execution(tmp_path, *, passed=True, degraded=False, fault=False):
    run = tmp_path / "run"
    path = run / "T01-llm_tools" / "record.json"
    path.parent.mkdir(parents=True)
    record = {"passed": passed, "scenario": "provider_failure" if fault else "normal",
              "degraded": degraded, "actualMode": "rule_fallback" if degraded else "llm_tools",
              "checks": [{"name": "no_actions_before_approval", "passed": True}],
              "runtimeTrace": {"llmCalls": [{}], "toolCalls": [], "usage": {"total_tokens": 100}},
              "durationMs": 150}
    path.write_text(json.dumps(record), encoding="utf-8")
    trial = {"caseId": "T01", "repetition": 1, "runDirectory": str(run)}
    return trial, str(path.resolve())


def test_judge_failure_never_turns_execution_success_into_task_failure(tmp_path):
    trial, path = execution(tmp_path)
    ratings = {path: {"hardPassed": True, "primaryEligible": True,
                      "assessment": {"status": "error", "verdict": None}}}
    row = trial_row(trial, {"version": "baseline", "split": "dev"}, ratings)
    value = metrics([row])
    assert value["taskPassRate"] == 1
    assert value["qualityMeanConditional"] is None
    assert value["qualityCoverage"] == 0 and value["judgeErrors"] == 1


def test_launch_failure_stays_in_denominator_and_known_fault_stays_out_of_fallback_denominator(tmp_path):
    trial, _ = execution(tmp_path, degraded=True, fault=True)
    row = trial_row(trial, {"version": "baseline", "split": "dev"}, {})
    missing = trial_row({"caseId": "T02", "repetition": 1, "launchError": True},
                        {"version": "baseline", "split": "dev"}, {})
    m = metrics([row, missing])
    assert m["taskPassRate"] == .5 and m["trials"] == 2
    assert m["nonInjectedTrials"] == 1 and m["unexpectedFallbacks"] == 0


def test_quality_mean_is_conditional_and_all_fallbacks_are_retained(tmp_path):
    trial, path = execution(tmp_path)
    verdict = {"unableToJudge": False, **{d: {"score": 4} for d in ("reasoning", "evidence", "feasibility")}}
    ratings = {path: {"hardPassed": True, "primaryEligible": True,
                      "assessment": {"status": "scored", "verdict": verdict}}}
    good = trial_row(trial, {"version": "baseline", "split": "dev"}, ratings)
    bad = {**good, "caseId": "T02", "taskPassed": False, "unexpectedFallback": True, "quality": None}
    m = metrics([good, bad])
    assert m["taskPassRate"] == .5 and m["qualityMeanConditional"] == 4
    assert m["qualityCoverage"] == .5 and m["unexpectedFallbackRate"] == .5


def test_repetitions_are_paired_and_scenario_count_is_not_inflated(tmp_path):
    trial, _ = execution(tmp_path)
    first = trial_row(trial, {"version": "baseline", "split": "dev"}, {})
    before = [first, {**first, "repetition": 2, "taskPassed": False}]
    after = [{**r, "version": "candidate", "taskPassed": True} for r in before]
    assert metrics(before)["scenarios"] == 1 and metrics(before)["trials"] == 2
    comparison = paired(before, after)
    assert comparison["recovered"] == [("T01", 2)] and comparison["passDeltaPercentagePoints"] == 50
    with pytest.raises(ValueError, match="unpaired"):
        paired(before, after[:1])


def test_no_adoption_decision_until_all_splits_have_complete_comparisons(tmp_path):
    write(tmp_path / "experiment.json", {"protocol": {}, "protocolHash": "test", "configuration": {},
          "datasets": {"dev": {}, "regression": {}, "holdout": {}}, "arms": {}})
    assert report(tmp_path)["adoptionGatePassed"] is None


def test_expected_guard_pass_is_visible_separately_from_workflow_completion(tmp_path):
    trial, _ = execution(tmp_path, degraded=True, fault=True)
    row = trial_row(trial, {"version": "baseline", "split": "holdout"}, {})
    row["workflowStatus"] = "planning_blocked"
    value = metrics([row])
    assert value["taskPassed"] == 1
    assert value["workflowStatusCounts"] == {"planning_blocked": 1}
    assert value["actualModeCounts"] == {"rule_fallback": 1}
    assert value["unexpectedFallbackRate"] is None
    assert value["qualityDimensionMeansConditional"] == {}
