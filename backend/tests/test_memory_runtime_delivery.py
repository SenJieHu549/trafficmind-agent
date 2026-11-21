"""Memory delivery reaches real role/model inputs, without changing event facts."""
import asyncio
from copy import deepcopy
import json

import pytest

from backend.agent.collaboration.context_projection import project_context_for_agent
from backend.memory.runtime_context import project_runtime_memory
from backend.tests.test_analysis_handoff import env
from backend.tests.test_llm_accident_agent import worker
from backend.tests.test_llm_accident_integration import setup


def memory_item(**changes):
    item = {
        "memoryId": "m-school", "memoryType": "constraint",
        "memoryKey": "constraint.pedestrian", "value": {"text": "Keep pedestrian access"},
        "status": "active", "sessionId": "session", "eventThreadId": "thread",
        "sourceType": "user_explicit", "sourceRunId": "previous",
    }
    return {**item, **changes}


def test_role_projection_is_separate_and_rejects_forbidden_memory():
    original = {"session_id": "session", "normalized_event": {
        "roadName": "Current road", "nearbySchool": False, "avgSpeed": 5,
    }, "agent_memory_contexts": {"AccidentAgent": {"items": [
        memory_item(),
        memory_item(memoryId="m-dynamic", value={"nested": {"avgSpeed": 99}}),
        memory_item(memoryId="m-other-role", memoryKey="school.nearby", memoryType="stable_fact",
                    value={"value": True}),
        memory_item(memoryId="m-foreign", sessionId="other"),
        memory_item(memoryId="m-expired", validUntil="2000-01-01T00:00:00Z"),
        memory_item(memoryId="m-future", validFrom="2999-01-01T00:00:00Z"),
        memory_item(memoryId="m-proposal", memoryType="proposal", memoryKey="proposal.action"),
    ]}}}
    before = deepcopy(original)
    projected = project_context_for_agent(original, "AccidentAgent")
    assert [i["memoryId"] for i in projected["memoryContext"]["items"]] == ["m-school"]
    assert projected["memoryContext"]["notAuthorization"]
    assert projected["roadName"] == "Current road"
    assert original == before
    assert "memoryContext" not in project_context_for_agent(original, "ConflictDetector")


def test_false_fact_and_thread_are_not_overridden():
    data = {"items": [
        memory_item(memoryType="stable_fact", memoryKey="school.nearby", value={"value": True}),
        memory_item(memoryId="wrong-thread", eventThreadId="other"),
    ]}
    assert not project_runtime_memory(
        data, "SignalAgent", session_id="session", event_thread_id="thread",
        current_event={"nearbySchool": False})


@pytest.mark.parametrize("status", ["candidate", "superseded", "expired", "rejected"])
def test_inactive_status_cannot_reach_runtime(status):
    assert not project_runtime_memory({"items": [memory_item(status=status)]}, "AccidentAgent")


def test_memory_payload_is_bounded_and_retains_provenance():
    data = {"items": [
        memory_item(memoryId=str(i), value={"text": "x" * 1000}) for i in range(100)
    ]}
    context = project_runtime_memory(data, "AccidentAgent")
    assert len(json.dumps(context, ensure_ascii=False)) <= 6000
    assert context["itemCount"] <= 8
    assert context["items"][0]["sourceRunId"] == "previous"


def test_real_orchestrator_delivers_memory_to_model_and_task_snapshot(setup, worker):
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator

    original = setup["collaboration"].get_run("source")
    grounding = json.loads(original["grounding_context"])
    event = {**setup["event"], "originalInput": "Analyze the current accident"}
    before = deepcopy(event)
    memory = {"AccidentAgent": {"items": [memory_item()]}}

    async def execute():
        return [frame async for frame in CollaborationOrchestrator().execute(
            "memory-delivery", "session", event, ["AccidentAgent"],
            grounding_context=grounding, agent_memory_contexts=memory)]

    asyncio.run(execute())
    payload = json.loads(setup["client"].requests[0]["messages"][1]["content"])
    assert payload["memoryContext"]["items"][0]["memoryId"] == "m-school"
    assert payload["memoryContext"]["notCurrentEventFacts"]
    assert not any(e["kind"] == "memory" for e in payload["availableEvidence"])
    assert event == before
    task = next(t for t in setup["collaboration"].list_tasks("memory-delivery")
                if t["agent_name"] == "AccidentAgent")
    snapshot = json.loads(task["input_snapshot"])
    assert snapshot["memoryContext"]["items"][0]["sourceRunId"] == "previous"
    output = json.loads(task["output_snapshot"])
    assert output["runtimeTrace"]["memoryInput"]["memoryIds"] == ["m-school"]


def test_expanded_parent_is_revalidated_in_accident_snapshot(setup, worker):
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator
    from backend.agent.llm_accident.integration import validate_evidence_snapshot
    from backend.agent.llm_accident.contracts import AgentRuntimeError
    from backend.agent.llm_accident.contracts import EvidenceLedger
    from backend.agent.llm_accident.tools import EventQueries
    from backend.rag.v2.chunker import TrafficKnowledgeChunker
    from backend.rag.v2.models import RagDocument, DocType

    doc = RagDocument(
        document_id="tool-doc", source_id="test", title="Case", doc_type=DocType.CASE,
        event_type="accident", grounding_scope="GLOBAL",
        content="事实：No injuries confirmed.\n措施：Keep access open.\n结果：Access remained available.")
    parents, children = TrafficKnowledgeChunker().chunk_document(doc)
    children[1].chunk_id = "tool-chunk"  # The existing scripted knowledge tool requests this ID.
    setup["docs"].upsert_document(doc)
    setup["docs"].upsert_chunks(parents + children)
    grounding = json.loads(setup["collaboration"].get_run("source")["grounding_context"])

    async def execute():
        return [frame async for frame in CollaborationOrchestrator().execute(
            "parent-snapshot", "session", setup["event"], ["AccidentAgent"],
            grounding_context=grounding)]

    asyncio.run(execute())
    task = next(t for t in setup["collaboration"].list_tasks("parent-snapshot")
                if t["agent_name"] == "AccidentAgent")
    snapshot = json.loads(task["output_snapshot"])["evidenceSnapshot"]
    validate_evidence_snapshot(snapshot, "handoff-event")
    # A historical child-only snapshot remains valid without inheriting new
    # context or changing its original evidence identity.
    queries = EventQueries("handoff-event")
    old_source = queries._knowledge_source(
        {"documentId": doc.document_id, "chunkId": "tool-chunk"},
        include_parent_context=False)
    legacy_ledger = EvidenceLedger(queries.event_id, str(queries.as_of))
    legacy_ledger.add("current_event", queries.event_id, queries.facts, "initial_event")
    old_id = legacy_ledger.add(
        old_source["kind"], old_source["sourceId"], old_source["data"], "grounding")
    old_snapshot = legacy_ledger.freeze([old_id])
    validate_evidence_snapshot(old_snapshot, "handoff-event")
    parents[0].raw_content += "\n结果：Updated outcome."
    # Mutate only the parent: upsert_chunks would also rewrite each child's
    # updated_at and correctly invalidate child-only snapshots for that reason.
    conn = setup["docs"]._get_conn()
    try:
        conn.execute("UPDATE rag_chunks SET raw_content=? WHERE chunk_id=?",
                     (parents[0].raw_content, parents[0].chunk_id))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(AgentRuntimeError, match="accident_evidence_source_changed"):
        validate_evidence_snapshot(snapshot, "handoff-event")
    validate_evidence_snapshot(old_snapshot, "handoff-event")
