"""Metric boundaries and deliberately corrupted outputs, without network I/O."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from backend.agent.llm_accident.contracts import EvidenceLedger, FinalAnalysis
from backend.evaluation.v2.recovery_benchmark import CONTROLS, FAULTS, feedback_for, mutate, recovered


@pytest.fixture
def sample():
    ledger = EvidenceLedger("benchmark-event", "2026-11-01T10:00:00+08:00")
    ref = ledger.add("event", "benchmark-event", {"vehicleCount": 2}, "fixture")
    body = {"findings": ["Collision reported; injury status unknown"], "confidence": .8,
        "suggestion": "Verify the scene", "urgency": "medium", "evidence_refs": [ref],
        "proposed_actions": [], "assumptions": []}
    return json.dumps(body), ledger.freeze([ref])


@pytest.mark.parametrize("kind", FAULTS)
def test_injected_fault_is_invalid_but_eligible_with_unchanged_protected_fields(sample, kind):
    content, snapshot = sample
    bad = mutate(content, kind)
    with pytest.raises(ValidationError):
        FinalAnalysis.model_validate_json(bad)
    assert feedback_for(bad, snapshot)["repairEligible"] is True
    if kind != "truncated_json":
        before, after = json.loads(content), json.loads(bad)
        assert before["evidence_refs"] == after["evidence_refs"]
        assert before["proposed_actions"] == after["proposed_actions"]


@pytest.mark.parametrize("kind", CONTROLS)
def test_mixed_protected_fault_never_eligible(sample, kind):
    content, snapshot = sample
    feedback = feedback_for(mutate(content, kind), snapshot)
    assert feedback["repairEligible"] is False
    assert feedback["guard"] == "protected_contract_invalid"


@pytest.mark.parametrize("mutation", ("task_failed", "repair_failed", "runtime_failed"))
def test_schema_only_success_is_not_task_recovery(mutation):
    record = {"passed": True, "runtimeTrace": {"outputRepairValidated": True, "status": "completed"}}
    assert recovered(record)
    bad = deepcopy(record)
    if mutation == "task_failed":
        bad["passed"] = False
    elif mutation == "repair_failed":
        bad["runtimeTrace"]["outputRepairValidated"] = False
    else:
        bad["runtimeTrace"]["status"] = "failed"
    assert not recovered(bad)
