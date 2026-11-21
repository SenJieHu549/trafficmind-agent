"""Opt-in source-to-model measurement checks, separate from Judge quality."""
from copy import deepcopy
import json


def capture_model_inputs(record, entries):
    if "inputFidelity" not in record:
        return
    model_inputs = []
    for entry in entries:
        for message in (entry.get("request") or {}).get("messages", []):
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict) or "availableEvidence" not in payload:
                continue
            model_inputs.append({"event": deepcopy(payload.get("event") or {}),
                "currentEvidence": [deepcopy(e.get("data") or {}) for e in payload["availableEvidence"]
                                    if e.get("kind") == "current_event"]})
    record["inputFidelity"]["modelInputs"] = model_inputs


def measurement_checks(case, record, check):
    fields = case.get("expected", {}).get("inputFidelityFields", [])
    if not fields:
        return
    audit = record.get("inputFidelity") or {}
    declared = audit.get("declaredEvent") or {}
    check("original_input_recorded", all(field in declared for field in fields))
    # Explicit case overrides are checked independently of the recorded source.
    overrides = case.get("event", {})
    for field in fields:
        if field in overrides:
            check("original_override_matches:" + field,
                  field in declared and declared[field] == overrides[field])
    stages = {"storage": audit.get("storedEvent") or {}, "hydrated": record.get("event") or {},
              "grounding": (record.get("initialGrounding") or {}).get("currentEvent") or {}}
    model_inputs = audit.get("modelInputs") or []
    if record.get("accidentAgentSelected") and record.get("modeRequested") != "rule":
        check("model_input_fidelity_coverage", bool(model_inputs) and len(model_inputs) == record.get("modelRequests"))
        for index, model_input in enumerate(model_inputs):
            stages[f"model_event_{index}"] = model_input.get("event") or {}
            evidence = model_input.get("currentEvidence") or []
            check(f"model_current_evidence_present:{index}", bool(evidence))
            for ordinal, item in enumerate(evidence):
                stages[f"model_evidence_{index}_{ordinal}"] = item
    for stage, event in stages.items():
        for field in fields:
            value = event.get(field)
            expected = overrides.get(field) if field in overrides else declared.get(field)
            matches = (not isinstance(value, bool) and value == expected)
            check(f"input_fidelity:{stage}:{field}",
                  field in declared and field in event and matches,
                  "Original declared measurement, including null versus numeric zero, must survive unchanged.")
