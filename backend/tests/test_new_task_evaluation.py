"""Natural non-model review routing and plan-only scope stay explicit."""
from copy import deepcopy
import json
import sqlite3
from pathlib import Path

import pytest

from backend.evaluation.v2.checks import evaluate
from backend.evaluation.v2.runner import load_dataset, run_suite
from backend.evaluation.v2.quality.inputs import build_payload


DATASET = Path(__file__).resolve().parents[1] / "evaluation_data/trafficmind_new_tasks_20261111.json"


@pytest.fixture(scope="module")
def natural_review(tmp_path_factory):
    root, report = run_suite(DATASET, tmp_path_factory.mktemp("natural-review"),
                             case_ids=["H01", "H02", "H03", "H04"], deadline=180)
    return root, report


@pytest.mark.parametrize("case_id,stage", [("H01", "awaiting_review"), ("H02", "planned"),
                                          ("H03", "planned"), ("H04", "review_rejected")])
def test_natural_routes_do_not_fake_accident_model_execution(natural_review, case_id, stage):
    root, _ = natural_review
    record = json.loads((root / f"{case_id}-llm_tools/record.json").read_text(encoding="utf-8"))
    assert record["passed"], record["checks"]
    assert record["workflowStatus"] == stage
    assert record["actualMode"] == "not_selected" and record["modelRequests"] == 0
    assert not record["llmPathCompleted"]
    assert "scriptedDomainRoles" not in record["profile"]
    assert record["profile"].get("routing") != "explicit_offline_fixture"
    if stage == "planned":
        assert record["plan"]["metadata"]["analysisReuse"]["mode"] == "legacy_analysis"
        assert not record.get("workflow") and not record["isolatedEffects"]
    with sqlite3.connect(root / f"{case_id}-llm_tools/rag.db") as conn:
        assert {row[0] for row in conn.execute("SELECT event_type FROM rag_documents")} == {"congestion"}


def test_unexpected_accident_role_is_not_accepted_as_non_model_path(natural_review):
    root, _ = natural_review
    record = json.loads((root / "H02-llm_tools/record.json").read_text(encoding="utf-8"))
    record["tasks"].append({"agent_name": "AccidentAgent"})
    case = next(c for c in load_dataset(DATASET)["cases"] if c["caseId"] == "H02")
    assert not next(c for c in evaluate(case, record) if c["name"] == "accident_role_explicitly_not_selected")["passed"]


def test_judge_receives_only_memory_actually_present_in_model_requests():
    context = {"items": [{"memoryId": "delivered", "value": {"text": "Keep access available"}}],
               "notCurrentEventFacts": True, "notAuthorization": True}
    record = {"caseId": "case", "ids": {}, "event": {}, "agentResult": {"suggestion": "Verify conditions"},
              "memoryFixture": {"memoryIds": ["delivered", "not-delivered"]},
              "modelMemoryContexts": [context, deepcopy(context)]}
    payload = build_payload(record, {"question": "Continue analysis"})
    assert payload["referenceMemory"] == [context]
    assert "not-delivered" not in json.dumps(payload)
    record.pop("modelMemoryContexts")
    assert "referenceMemory" not in build_payload(record, {})
