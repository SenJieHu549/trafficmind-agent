"""Old notification actions remain inside the isolated Workflow boundary."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.evaluation.v2.runner import PROJECT, run_suite
from backend.evaluation.v2.sandbox import action_registry


@pytest.mark.parametrize("name", ["notify_wechat", "notify_dingtalk"])
def test_legacy_adapter_never_calls_real_notification_sender(monkeypatch, name):
    from backend.tools import notify_tools
    def forbidden(*args, **kwargs):
        pytest.fail("real notification sender was invoked")
    monkeypatch.setattr(notify_tools, "send_wechat_work", forbidden)
    monkeypatch.setattr(notify_tools, "send_dingtalk", forbidden)
    effects = []
    adapter = action_registry(effects).get(name)
    context = SimpleNamespace(action_type=name, event_id="event", workflow_run_id="run", node_id="node",
                              idempotency_key="key", params={})
    assert adapter.validate(context) is None
    result = asyncio.run(adapter.execute(context))
    assert result.status.value == "succeeded" and result.metadata["evaluationOnly"]
    assert effects == [{"actionType":name,"eventId":"event","runId":"run","nodeId":"node",
                        "idempotencyKey":"key","params":{}}]
    context.params = {"eventId":"another-event"}
    assert adapter.validate(context) is not None


@pytest.fixture(scope="module")
def legacy_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("h01-compatibility")
    data = json.loads((PROJECT / "backend/evaluation/v2/datasets/quality_holdout.json").read_text(encoding="utf-8"))
    original = next(c for c in data["cases"] if c["caseId"] == "H01")
    data["cases"] = []
    for scenario, status in (("approve_repeat","completed"),("reject","rejected"),("facts_changed","failed")):
        case = deepcopy(original)
        case.update(caseId="H01_" + scenario, scenario=scenario, split="regression")
        case["expected"]["workflowStatus"] = status
        data["cases"].append(case)
    multiple = deepcopy(original)
    multiple.update(caseId="H01_two_approvals", scenario="approve_repeat", split="regression")
    multiple["expected"]["requiresCandidate"] = True
    data["cases"].append(multiple)
    data.update(split="regression", purpose="h01_legacy_adapter_regression_not_fresh_holdout")
    dataset = directory / "h01-regression.json"
    dataset.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    root, summary = run_suite(dataset, directory, modes=["llm_tools"])
    assert summary["passed"] == summary["total"] == 4
    return {scenario:json.loads((root / f"H01_{scenario}-llm_tools/record.json").read_text(encoding="utf-8"))
            for scenario in ("approve_repeat","reject","facts_changed","two_approvals")}


@pytest.mark.parametrize("scenario", ["approve_repeat", "reject", "facts_changed"])
def test_h01_uses_real_approval_and_isolated_execution(legacy_runs, scenario):
    record = legacy_runs[scenario]
    assert record["statusBeforeDecision"] == "awaiting_approval"
    assert not record["actionsBeforeDecision"] and not record["effectsBeforeDecision"]
    assert record["approvals"]
    if scenario != "approve_repeat":
        assert not record["actions"] and not record["isolatedEffects"]
        return
    assert record["workflowStatus"] == "completed"
    assert [e["actionType"] for e in record["isolatedEffects"]] == ["notify_wechat"]
    assert len([a for a in record["actions"] if a["actionType"] == "notify_wechat"]) == 1
    assert record["repeatExecution"]["unchanged"]
    events = [e["eventType"] for e in record["workflowEvents"]]
    assert events.index("approval_approved") < events.index("action_started")


def test_h01_notification_and_dispatch_each_require_their_own_approval(legacy_runs):
    from backend.evaluation.v2.checks import approvals_precede_required_actions
    record = legacy_runs["two_approvals"]
    assert record["workflowStatus"] == "completed"
    assert len(record["approvalDecisions"]) == 2
    assert len({a["approvalId"] for a in record["approvalDecisions"]}) == 2
    assert [e["actionType"] for e in record["isolatedEffects"]] == ["notify_wechat", "create_dispatch_task"]
    assert approvals_precede_required_actions(record)
    assert record["repeatExecution"]["unchanged"]


def test_sequential_approvals_are_checked_per_action_not_globally():
    from backend.evaluation.v2.checks import approvals_precede_required_actions
    approvals = [{"approvalId":"a", "decision":"approved", "proposedActions":[{"actionType":"notify_wechat"}]},
                 {"approvalId":"b", "decision":"approved", "proposedActions":[{"actionType":"create_dispatch_task","actionStepId":"dispatch"}]}]
    events = [{"eventType":"approval_approved","payload":{"approvalId":"a"}},
              {"eventType":"action_started","nodeId":"notify","payload":{"actionType":"notify_wechat"}},
              {"eventType":"approval_approved","payload":{"approvalId":"b"}},
              {"eventType":"action_started","nodeId":"dispatch","payload":{"actionType":"create_dispatch_task"}}]
    assert approvals_precede_required_actions({"approvals":approvals,"workflowEvents":events})
    assert not approvals_precede_required_actions({"approvals":approvals,"workflowEvents":events[:2]+[events[3]]})


def test_unrelated_approval_does_not_authorize_another_action():
    from backend.evaluation.v2.checks import approvals_precede_required_actions
    record = {"approvals":[{"approvalId":"a","decision":"approved","proposedActions":[{"actionType":"notify_wechat"}]}],
              "workflowEvents":[{"eventType":"approval_approved","payload":{"approvalId":"a"}},
                                {"eventType":"action_started","nodeId":"dispatch","payload":{"actionType":"create_dispatch_task"}}]}
    assert not approvals_precede_required_actions(record)


def test_step_bound_approval_cannot_authorize_same_type_at_another_node():
    from backend.evaluation.v2.checks import approvals_precede_required_actions
    record = {"approvals":[{"approvalId":"a","decision":"approved","proposedActions":[{"actionType":"create_dispatch_task","actionStepId":"other-node"}]}],
              "workflowEvents":[{"eventType":"approval_approved","payload":{"approvalId":"a"}},
                                {"eventType":"action_started","nodeId":"dispatch","payload":{"actionType":"create_dispatch_task"}}]}
    assert not approvals_precede_required_actions(record)
