"""Source-fidelity assertions fail when null becomes zero or evidence disappears."""
from copy import deepcopy
import json

import pytest

from backend.evaluation.v2.input_fidelity import capture_model_inputs, measurement_checks


def fixture():
    case = {"event": {"avgSpeed": None}, "expected": {"inputFidelityFields": ["avgSpeed", "duration"]}}
    event = {"avgSpeed": None, "duration": 0}
    record = {"inputFidelity": {"declaredEvent": deepcopy(event), "storedEvent": deepcopy(event)},
              "event": deepcopy(event), "initialGrounding": {"currentEvent": deepcopy(event)},
              "accidentAgentSelected": True, "modeRequested": "llm_tools", "modelRequests": 1}
    payload = {"event": deepcopy(event), "availableEvidence": [{"kind": "current_event", "data": deepcopy(event)}]}
    entries = [{"request": {"messages": [{"role": "user", "content": json.dumps(payload)}]}}]
    capture_model_inputs(record, entries)
    return case, record


def checks(case, record):
    result = []
    measurement_checks(case, record, lambda name, passed, detail="": result.append((name, passed)))
    return result


def test_preserves_both_null_and_real_zero():
    case, record = fixture()
    assert all(passed for _, passed in checks(case, record))


@pytest.mark.parametrize("stage", ["storage", "hydrated", "grounding", "model_event", "model_evidence"])
def test_unknown_to_zero_is_rejected_at_each_stage(stage):
    case, record = fixture()
    target = {"storage": record["inputFidelity"]["storedEvent"], "hydrated": record["event"],
              "grounding": record["initialGrounding"]["currentEvent"],
              "model_event": record["inputFidelity"]["modelInputs"][0]["event"],
              "model_evidence": record["inputFidelity"]["modelInputs"][0]["currentEvidence"][0]}[stage]
    target["avgSpeed"] = 0
    assert any(not passed for _, passed in checks(case, record))


@pytest.mark.parametrize("missing", ["audit", "grounding", "requests", "evidence", "field"])
def test_missing_evidence_cannot_vacuously_pass(missing):
    case, record = fixture()
    if missing == "audit": record.pop("inputFidelity")
    if missing == "grounding": record.pop("initialGrounding")
    if missing == "requests": record["inputFidelity"]["modelInputs"] = []
    if missing == "evidence": record["inputFidelity"]["modelInputs"][0]["currentEvidence"] = []
    if missing == "field": record["inputFidelity"]["storedEvent"].pop("avgSpeed")
    assert any(not passed for _, passed in checks(case, record))


def test_explicit_override_is_not_replaced_by_corrupted_recorded_source():
    case, record = fixture()
    record["inputFidelity"]["declaredEvent"]["avgSpeed"] = 0
    assert not dict(checks(case, record))["original_override_matches:avgSpeed"]


def test_old_cases_do_not_gain_retroactive_assertions():
    assert checks({"expected": {}}, {}) == []
