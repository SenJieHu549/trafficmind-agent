"""Handoff contract and real SQLite Workflow/Driver integration, no models."""
import asyncio
from copy import deepcopy
import json
import sqlite3

import pytest

from backend.planning import analysis_handoff as handoff


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import backend.config as cfg
    import backend.tools.db_tools as db
    import backend.agent.collaboration.db_repository as collaboration
    import backend.chat.chat_db as chat
    import backend.rag.v2.document_repository as docs
    from backend.rag.v2.models import RagDocument, RagChunk
    from backend.tools.event_identity import hydrate_authoritative_event, compact_event_context
    from backend.workflow.repository import SQLiteWorkflowRepository
    from backend.planning import api

    path = str(tmp_path / "events.db")
    for module in (cfg, db, collaboration, chat):
        monkeypatch.setattr(module, "DB_PATH", path)
    monkeypatch.setattr(docs, "RAG_V2_DB_PATH", str(tmp_path / "rag.db"))
    db.init_db()
    collaboration.init_collaboration_tables()
    chat.reset_initialized()
    chat.init_chat_tables()
    docs.init_db()
    event = {"eventId": "handoff-event", "eventType": "accident", "eventTypeCn": "事故",
             "roadName": "handoff road", "avgSpeed": 45, "queueLength": 0, "duration": 60,
             "vehicleCount": 2, "weather": "clear", "timePeriod": "off_peak",
             "isMainRoad": False, "nearbySchool": False, "nearbyHospital": False}
    assert db.save_event_analysis({"eventId": event["eventId"], "standardEvent": event,
                                   "riskScore": 45, "riskLevel": "中风险", "status": "待派单", "report": ""})
    baseline = compact_event_context(hydrate_authoritative_event(event["eventId"]))
    baseline["createdAt"] = "2026-01-01T00:00:00+08:00"
    docs.upsert_document(RagDocument(document_id="doc", source_id="test", title="Accident guidance",
                                     content="Keep the emergency lane clear.", grounding_scope="GLOBAL",
                                     event_type="accident"))
    docs.upsert_chunks([RagChunk(chunk_id="chunk", document_id="doc", raw_content="Keep the emergency lane clear.",
                                contextual_content="Accident: Keep the emergency lane clear.")])
    grounding = {"currentEvent": baseline, "knowledgeContext": {"evidence": [
        {"documentId": "doc", "chunkId": "chunk", "excerpt": "Keep the emergency lane clear.", "evidenceId": "E1"}]}}
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO collaboration_runs (run_id,session_id,status,normalized_event,grounding_context,final_decision,completed_at) VALUES (?,?,?,?,?,?,?)",
                     ("source", "session", "completed", json.dumps(baseline), json.dumps(grounding),
                      json.dumps({"summary": "Keep emergency access open"}), "2026-01-01T00:01:00Z"))
        conn.execute("INSERT INTO collaboration_tasks (run_id,task_id,agent_name,status,output_snapshot) VALUES (?,?,?,?,?)",
                     ("source", "accident", "AccidentAgent", "succeeded", json.dumps({"suggestion": "Keep access open", "confidence": .9})))
    repo = SQLiteWorkflowRepository()
    monkeypatch.setattr(api, "_repo", repo)
    # Terminal projection remains enabled; no model/assessment is involved here.
    return {"db": path, "docs": docs, "repo": repo, "api": api, "event": baseline,
            "collaboration": collaboration.SQLiteCollaborationRepository()}


def snapshot(env):
    return handoff.build_analysis_snapshot("handoff-event", "session", "source")


def mutate(env, sql, args=()):
    with sqlite3.connect(env["db"]) as conn:
        conn.execute(sql, args)


def plan(env, reuse=True):
    api = env["api"]
    response = asyncio.run(api.create_plan_from_agent(api.AgentPlanRequest(
        eventId="handoff-event", sessionId="session", collaborationRunId="source", reuseAnalysis=reuse)))
    from backend.planning.models import Plan
    return Plan.from_dict(response["plan"])


def run_to_terminal(env, p):
    from backend.workflow.executor import WorkflowExecutor
    api = env["api"]
    async def scenario():
        # Real startup API revalidation and durable pending run creation.
        await api.run_plan(p.planId, api.PlanRunRequest(sessionId="session"))
        run = env["repo"].list_runs(definition_id=p.planId)[0]
        executor = WorkflowExecutor(repository=env["repo"])
        async for _ in executor.execute_created_run(run.run_id):
            pass
        return env["repo"].get_run(run.run_id)
    return asyncio.run(scenario())


def test_snapshot_roundtrip_and_metadata_changes_are_not_fact_changes(env):
    from backend.workflow.state import TrafficWorkflowState
    snap = snapshot(env)
    mutate(env, "UPDATE event_records SET status='处置中', report='display change' WHERE eventId='handoff-event'")
    assert handoff.validate_analysis_snapshot(snap, env["event"])["valid"]
    state = TrafficWorkflowState(analysis_snapshot=snap, analysis_validation={"valid": True})
    assert TrafficWorkflowState.from_dict(state.to_dict()).analysis_snapshot == snap
    assert handoff.event_facts({**env["event"], "avgSpeed": "45.0"}) == snap["facts"]


@pytest.mark.parametrize("change,code", [
    ("UPDATE collaboration_runs SET status='running'", "analysis_source_not_completed"),
    ("UPDATE collaboration_runs SET session_id='wrong'", "analysis_identity_mismatch"),
    ("UPDATE collaboration_runs SET final_decision='different conclusion'", "analysis_source_changed"),
    ("UPDATE event_records SET duration=3600", "analysis_facts_changed"),
    ("UPDATE event_records SET nearbySchool=1", "analysis_facts_changed"),
    ("UPDATE event_records SET status='已处置'", "event_terminal"),
])
def test_source_and_live_fact_failures(env, change, code):
    snap = snapshot(env)
    mutate(env, change)
    with pytest.raises(ValueError) as error:
        handoff.validate_analysis_snapshot(snap, env["event"])
    assert error.value.code == code


def test_missing_past_baseline_does_not_use_current_facts(env):
    mutate(env, "UPDATE collaboration_runs SET grounding_context='{}'")
    with pytest.raises(handoff.AnalysisHandoffError, match="analysis_baseline_missing"):
        snapshot(env)


def test_missing_evidence_is_rejected(env):
    with sqlite3.connect(env["db"]) as conn:
        row = conn.execute("SELECT grounding_context FROM collaboration_runs").fetchone()
        value = json.loads(row[0]); value["knowledgeContext"]["evidence"] = []
        conn.execute("UPDATE collaboration_runs SET grounding_context=?", (json.dumps(value),))
    with pytest.raises(handoff.AnalysisHandoffError, match="analysis_evidence_missing"):
        snapshot(env)


def test_source_review_requirement_cannot_be_dropped(env):
    mutate(env, "UPDATE collaboration_runs SET final_decision=?", (json.dumps({"summary": "review", "requiresHumanReview": ["conflict"]}),))
    with pytest.raises(handoff.AnalysisHandoffError, match="analysis_review_required"):
        snapshot(env)


def test_fusion_confidence_is_preserved(env):
    mutate(env, "UPDATE collaboration_runs SET final_decision=?", (json.dumps({"summary": "decision", "confidence": .8}),))
    assert snapshot(env)["agentOutputs"]["SourceFusion"]["confidence"] == .8


@pytest.mark.parametrize("kind", ["document", "chunk", "expired", "wrong_scope", "wrong_type", "deleted"])
def test_evidence_changes_are_rejected(env, kind):
    snap = snapshot(env)
    docs = env["docs"]
    doc, chunk = docs.get_document("doc"), docs.get_chunk("chunk")
    if kind == "chunk":
        chunk.raw_content = "Different guidance"; chunk.contextual_content = "Different guidance"
        docs.upsert_chunks([chunk])
    else:
        if kind == "document": doc.version += 1
        if kind == "expired":
            from datetime import datetime, timezone
            doc.effective_to = datetime(2025, 1, 1, tzinfo=timezone.utc)
        if kind == "wrong_scope": doc.grounding_scope = "REGIONAL"; doc.region_id = "unknown-region"
        if kind == "wrong_type": doc.event_type = "signal_fault"
        if kind == "deleted": doc.status = "deleted"
        docs.upsert_document(doc)
    with pytest.raises(handoff.AnalysisHandoffError):
        handoff.validate_analysis_snapshot(snap, env["event"])


def test_client_forgery_cannot_authorize_snapshot(env):
    snap = snapshot(env)
    snap["finalDecision"] = "forged"
    snap["snapshotDigest"] = handoff._digest({k: v for k,v in snap.items() if k != "snapshotDigest"})
    with pytest.raises(handoff.AnalysisHandoffError, match="analysis_source_changed"):
        handoff.validate_analysis_snapshot(snap, env["event"])


def test_changed_canonical_binding_invalidates_old_analysis(env, monkeypatch):
    from backend.regional.repository import SQLiteRegionalRepository
    snap = snapshot(env)
    monkeypatch.setattr(SQLiteRegionalRepository, "get_active_event_location_binding", lambda *args: {"regionId": "new-region"})
    with pytest.raises(handoff.AnalysisHandoffError, match="analysis_location_changed"):
        handoff.validate_analysis_snapshot(snap, env["event"])


def test_plan_reuse_completes_without_any_duplicate_analysis(env, monkeypatch):
    from backend.workflow.models import NodeType
    from backend.workflow.nodes.base import get_node_registry
    from backend.workflow.nodes import register_all_nodes
    p = plan(env)
    assert NodeType.ANALYSIS_REUSE in [s.stepType for s in p.steps]
    assert not any(s.stepType in (NodeType.AGENT_TASK, NodeType.RAG_RETRIEVE, NodeType.MEMORY_CONTEXT) for s in p.steps)
    import backend.workflow.nodes as nodes
    async def forbidden(*args, **kwargs):
        pytest.fail("duplicate analysis was called")
    for name in ("execute_rag_retrieve", "execute_agent_task", "execute_memory_context"):
        monkeypatch.setattr(nodes, name, forbidden)
    run = run_to_terminal(env, p)
    assert run.status.value == "completed"
    assert run.state["analysisValidation"]["mode"] == "reused"
    assert "SourceFusion" in run.state["agentOutputs"]
    assert len(env["repo"].list_action_records(run.run_id)) == 1
    assert all(n.status.value == "succeeded" for n in env["repo"].get_node_runs(run.run_id))


def test_old_api_mode_keeps_original_nodes(env):
    from backend.workflow.models import NodeType
    p = plan(env, reuse=False)
    assert any(s.stepType == NodeType.RAG_RETRIEVE for s in p.steps)
    assert not any(s.stepType == NodeType.ANALYSIS_REUSE for s in p.steps)


def test_other_event_types_keep_legacy_analysis_with_explicit_reason(env):
    from backend.workflow.models import NodeType
    mutate(env, "UPDATE event_records SET eventType='signal_fault',eventTypeCn='信号灯异常'")
    mutate(env, "UPDATE collaboration_runs SET normalized_event=?", (json.dumps({**env["event"], "eventType": "signal_fault"}),))
    p = plan(env, reuse=True)
    assert p.metadata["analysisReuse"]["reason"] == "event_type_not_supported_in_v1"
    assert any(s.stepType == NodeType.RAG_RETRIEVE for s in p.steps)
    assert not any(s.stepType == NodeType.ANALYSIS_REUSE for s in p.steps)


def test_startup_changed_facts_rejects_before_creating_run(env):
    from fastapi import HTTPException
    p = plan(env)
    mutate(env, "UPDATE event_records SET duration=120")
    with pytest.raises(HTTPException) as error:
        asyncio.run(env["api"].run_plan(p.planId, env["api"].PlanRunRequest()))
    assert error.value.status_code == 409
    assert not env["repo"].list_runs(definition_id=p.planId)


def test_startup_rejects_cross_session_and_derives_missing_session(env):
    from fastapi import HTTPException
    p = plan(env)
    with pytest.raises(HTTPException) as error:
        asyncio.run(env["api"].run_plan(p.planId, env["api"].PlanRunRequest(sessionId="wrong-session")))
    assert error.value.status_code == 409
    assert not env["repo"].list_runs(definition_id=p.planId)
    asyncio.run(env["api"].run_plan(p.planId, env["api"].PlanRunRequest()))
    run = env["repo"].list_runs(definition_id=p.planId)[0]
    assert run.session_id == "session"


def test_validator_disallows_handoff_bypass(env):
    from backend.planning.validator import validate_plan
    from backend.workflow.models import NodeType
    p = plan(env)
    next(s for s in p.steps if s.stepType == NodeType.ACTION).dependsOn = ["rule_router"]
    assert "analysis_handoff_bypassed" in [i.code for i in validate_plan(p)]


def test_unknown_confidence_is_not_zero_and_evidence_is_deduplicated(env):
    from backend.workflow.state import TrafficWorkflowState
    from backend.workflow.models import NodeConfig, NodeType
    from backend.workflow.nodes.analysis_reuse import execute_analysis_reuse
    from backend.workflow.nodes.evidence_evaluate import execute_evidence_evaluate
    state = TrafficWorkflowState(current_event=env["event"])
    asyncio.run(execute_analysis_reuse(state, NodeConfig(node_id="reuse", node_type=NodeType.ANALYSIS_REUSE,
                                                       config={"analysis_snapshot": snapshot(env)})))
    result = asyncio.run(execute_evidence_evaluate(state, NodeConfig(node_id="ev", node_type=NodeType.EVIDENCE_EVALUATE)))
    assert result["avg_confidence"] == .9
    assert result["unknown_confidence_count"] == 1
    assert result["total_evidence_count"] == 1
    state.agent_outputs["AccidentAgent"]["confidence"] = .1
    assert asyncio.run(execute_evidence_evaluate(state, NodeConfig(node_id="ev", node_type=NodeType.EVIDENCE_EVALUATE))).get("error")


@pytest.mark.parametrize("decision", ["approve", "reject", "changed", "cancel"])
def test_approval_resume_rechecks_and_preserves_terminal_state(env, decision):
    from backend.planning.adapter import plan_to_definition
    from backend.planning.models import PlanStep
    from backend.workflow.models import NodeType
    from backend.workflow.executor import WorkflowExecutor
    # Persisted source proposes a registered high-risk internal dispatch action.
    with sqlite3.connect(env["db"]) as conn:
        conn.execute("UPDATE collaboration_tasks SET output_snapshot=?", (json.dumps({
            "suggestion": "Notify operators", "confidence": .9,
            "proposed_actions": [{"actionType": "create_dispatch_task", "params": {"instruction": "test internal dispatch"}}]}),))
    p = plan(env)
    run = run_to_terminal(env, p)
    assert run.status.value == "awaiting_approval"
    assert not env["repo"].list_action_records(run.run_id)
    executor = WorkflowExecutor(repository=env["repo"])
    async def finish():
        if decision == "cancel":
            await executor.cancel(run.run_id, "test cancellation")
            async for _ in executor.execute_created_run(run.run_id): pass
            return
        if decision == "changed": mutate(env, "UPDATE event_records SET duration=1800")
        operation = executor.reject if decision == "reject" else executor.approve
        await operation(run.run_id, reviewer="test", comment="review")
        if decision in ("approve", "changed"):
            async for _ in executor.execute_created_run(run.run_id): pass
    asyncio.run(finish())
    final = env["repo"].get_run(run.run_id)
    expected = {"approve": "completed", "reject": "rejected", "changed": "failed", "cancel": "cancelled"}
    assert final.status.value == expected[decision]
    actions = env["repo"].list_action_records(run.run_id)
    if decision != "approve": assert not actions
    else:
        assert len(actions) == 2  # internal dispatch plus save_result
        assert all(a.status.value == "succeeded" for a in actions)


def test_real_driver_runs_handoff_and_releases_lease(env, monkeypatch):
    from backend.workflow.run_driver import RunDriver
    from backend.workflow.executor import WorkflowExecutor
    import backend.workflow.executor as execution
    from backend.workflow.recovery import RecoverySafetyClassifier, RecoverySafetyClass
    p = plan(env)
    async def scenario():
        await env["api"].run_plan(p.planId, env["api"].PlanRunRequest())
        run = env["repo"].list_runs(definition_id=p.planId)[0]
        monkeypatch.setattr(execution, "get_executor", lambda: WorkflowExecutor(repository=env["repo"]))
        driver = RunDriver(repository=env["repo"], lease_seconds=2, heartbeat_interval=.1)
        async def no_assessment(_): pass
        monkeypatch.setattr(driver, "_assess_if_terminal", no_assessment)
        await driver._poll_once()
        for _ in range(300):
            fresh = env["repo"].get_run(run.run_id)
            with sqlite3.connect(env["db"]) as conn:
                owner = conn.execute("SELECT driver_owner FROM workflow_runs WHERE run_id=?", (run.run_id,)).fetchone()[0]
            if fresh.status.value == "completed" and owner is None: break
            await asyncio.sleep(.02)
        assert fresh.status.value == "completed" and owner is None
        assert len(env["repo"].list_action_records(run.run_id)) == 1
    asyncio.run(scenario())
    assert RecoverySafetyClassifier().classify_node("analysis_reuse") == RecoverySafetyClass.READ_ONLY


def test_cancel_waiting_handoff_does_not_apply_late_result(env, monkeypatch):
    import threading
    import backend.workflow.nodes.analysis_reuse as node
    from backend.workflow.state import TrafficWorkflowState
    from backend.workflow.models import NodeConfig, NodeType
    entered, release = threading.Event(), threading.Event()
    def slow(*args):
        entered.set()
        release.wait(4)
        return {"valid": True}
    monkeypatch.setattr(node, "validate_analysis_snapshot", slow)
    state = TrafficWorkflowState(current_event=env["event"])
    async def scenario():
        task = asyncio.create_task(node.execute_analysis_reuse(state, NodeConfig(
            node_id="reuse", node_type=NodeType.ANALYSIS_REUSE, config={"analysis_snapshot": snapshot(env)})))
        try:
            for _ in range(100):
                if entered.is_set(): break
                await asyncio.sleep(.01)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        finally:
            release.set()
        await asyncio.sleep(.03)
        assert not state.analysis_snapshot and not state.agent_outputs
    asyncio.run(scenario())


def test_frozen_run_does_not_follow_later_definition(env):
    from backend.workflow.definition import DefinitionManager
    from backend.planning.models import Plan
    p = plan(env)
    api = env["api"]
    asyncio.run(api.run_plan(p.planId, api.PlanRunRequest()))
    run = env["repo"].list_runs(definition_id=p.planId)[0]
    current = env["repo"].get_definition(p.planId)
    current.metadata["plan"]["constraints"]["analysisSnapshot"]["finalDecision"] = "tampered latest"
    env["repo"].save_definition(current)
    frozen = DefinitionManager(env["repo"]).get_definition_at_version(p.planId, run.version)
    assert frozen.metadata["plan"]["constraints"]["analysisSnapshot"]["finalDecision"] != "tampered latest"
