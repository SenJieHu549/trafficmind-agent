from copy import deepcopy
import json

import pytest

from backend.evaluation.v2.tool_recovery_benchmark import assess, inject_limit, is_recovered


def test_injected_limit_changes_only_selected_tool_arguments():
    reply = {"message": {"tool_calls": [
        {"id": "a", "function": {"name": "accident_road_context", "arguments": "{}"}},
        {"id": "b", "function": {"name": "accident_case_search", "arguments": '{"limit":3}'}}]}, "usage": {"total_tokens": 100}}
    before = deepcopy(reply)
    injected = inject_limit(reply, "accident_case_search")
    assert reply == before
    assert injected["message"]["tool_calls"][0] == reply["message"]["tool_calls"][0]
    assert json.loads(injected["message"]["tool_calls"][1]["function"]["arguments"])["limit"] == 99
    assert injected["usage"] == {}


@pytest.mark.parametrize("kind,code", [("invalid_limit", "invalid_tool_arguments"), ("transient_query_error", "tool_execution_failed")])
def test_only_predeclared_call_error_is_allowed(monkeypatch, kind, code):
    from backend.evaluation.v2 import checks
    monkeypatch.setattr(checks, "evaluate", lambda case, record: [
        {"name": "tool_returned_valid_observation:first", "passed": False, "detail": ""},
        {"name": "tool_returned_valid_observation:second", "passed": False, "detail": ""}])
    spec = {"kind": kind, "tool": "accident_case_search", "faultCallId": "first", "control": False}
    record = {"runtimeTrace": {"toolCalls": [
        {"callId": "first", "tool": spec["tool"], "status": "error", "code": code},
        {"callId": "second", "tool": spec["tool"], "status": "error", "code": code}]}}
    result = assess({}, record, spec, [])
    assert result[0]["passed"] is True
    assert result[1]["passed"] is False
    assert record["checksBeforeInjectionAllowance"][0]["passed"] is False
    assert not next(c for c in result if c["name"] == "required_tool_recovered_with_evidence")["passed"]


def test_wrong_error_code_does_not_receive_injection_allowance(monkeypatch):
    from backend.evaluation.v2 import checks
    monkeypatch.setattr(checks, "evaluate", lambda case, record: [
        {"name": "tool_returned_valid_observation:first", "passed": False, "detail": ""}])
    spec = {"kind": "invalid_limit", "tool": "accident_case_search", "faultCallId": "first", "control": False}
    record = {"runtimeTrace": {"toolCalls": [
        {"callId": "first", "tool": spec["tool"], "status": "error", "code": "tool_denied"}]}}
    result = assess({}, record, spec, [])
    assert not result[0]["passed"] and not record["toolFaultObserved"]


@pytest.mark.parametrize("missing", ["passed", "toolFaultObserved", "toolRecoveryObtainedEvidence", "runtime"])
def test_recovery_requires_injection_and_evidence_and_task_success(missing):
    record = {"passed": True, "toolFaultObserved": True, "toolRecoveryObtainedEvidence": True,
        "runtimeTrace": {"status": "completed"}}
    assert is_recovered(record)
    if missing == "runtime": record["runtimeTrace"]["status"] = "degraded"
    else: record[missing] = False
    assert not is_recovered(record)
