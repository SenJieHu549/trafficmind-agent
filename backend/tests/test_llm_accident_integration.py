"""Real SQLite DAG -> Plan -> gated Workflow with scripted native LLM replies."""
import asyncio
from copy import deepcopy
import json
import sqlite3

import pytest

from backend.tests.test_analysis_handoff import env, run_to_terminal
from backend.tests.test_llm_accident_agent import Client, call, final, worker


@pytest.fixture
def setup(env, monkeypatch):
    from backend.agent.llm_accident import integration
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    from backend.agent.llm_accident.tools import EventQueries
    from backend.rag.v2.models import RagDocument, RagChunk
    import backend.agent.collaboration.orchestrator as orchestration
    from backend import config
    monkeypatch.setattr(config, "ACCIDENT_AGENT_MODE", "llm_tools")
    monkeypatch.setattr(orchestration, "LLM_ENABLED", False)
    with sqlite3.connect(env["db"]) as conn:
        conn.execute("UPDATE event_records SET createdAt=?", (env["event"]["createdAt"],))
    env["docs"].upsert_document(RagDocument(document_id="tool-doc", source_id="test", title="Extra evidence",
        content="Confirm injuries before requesting medical aid.", grounding_scope="GLOBAL", event_type="accident"))
    env["docs"].upsert_chunks([RagChunk(chunk_id="tool-chunk", document_id="tool-doc",
        raw_content="Confirm injuries before requesting medical aid.")])
    original = EventQueries.execute
    def query(self, name, params):
        if name == "accident_knowledge_search":
            return {"status": "ok", "sources": [self._knowledge_source({"documentId": "tool-doc", "chunkId": "tool-chunk"})]}
        return original(self, name, params)
    monkeypatch.setattr(EventQueries, "execute", query)
    def respond(messages):
        result = final(messages, actions=[{"actionType": "create_dispatch_task", "params": {"instruction": "Local test internal dispatch"}}])
        return result
    client = Client([[call()], [call("accident_knowledge_search", '{"query":"medical aid"}', "call_2")], respond])
    monkeypatch.setattr(integration, "AccidentAgentRuntime", lambda **kwargs: AccidentAgentRuntime(client=client, **kwargs))
    return {**env, "client": client}


def collaborate(env):
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator
    original = env["collaboration"].get_run("source")
    grounding = json.loads(original["grounding_context"])
    event = {**env["event"], "originalInput": "Analyze the accident; check road then relevant evidence."}
    async def execute():
        return [s async for s in CollaborationOrchestrator().execute("integrated", "session", event,
            ["AccidentAgent"], grounding_context=grounding)]
    frames = asyncio.run(execute())
    run = env["collaboration"].get_run("integrated")
    tasks = env["collaboration"].list_tasks("integrated")
    return run, tasks, frames


def make_plan(env):
    from backend.planning.models import Plan
    data = asyncio.run(env["api"].create_plan_from_agent(env["api"].AgentPlanRequest(eventId="handoff-event",
        sessionId="session", collaborationRunId="integrated", reuseAnalysis=True)))
    return Plan.from_dict(data["plan"])


def test_dag_multiround_evidence_plan_approval_and_completion(setup, worker):
    env = setup
    run, tasks, frames = collaborate(env)
    assert run["status"] == "completed"
    task = next(t for t in tasks if t["agent_name"] == "AccidentAgent")
    output = json.loads(task["output_snapshot"])
    assert task["timeout_seconds"] == 930 and task["max_retries"] == 0
    assert output["executionMode"] == "llm_tools" and not output["degraded"]
    assert output["runtimeTrace"]["citationPolicy"] == "canonical-evidence-id-1"
    assert all(ref.startswith("ev_") for ref in output["runtimeTrace"]["modelCitationRefs"])
    assert all(ref.startswith("ev_") for ref in output["evidence_refs"])
    assert output["runtimeTrace"]["toolRounds"] == 2
    assert len(output["runtimeTrace"]["llmCalls"]) == 3
    assert json.loads(run["grounding_context"])["agentEvidenceSnapshots"]["AccidentAgent"] == output["evidenceSnapshot"]
    dispatch = json.loads(next(t for t in tasks if t["agent_name"] == "DispatchAgent")["input_snapshot"])
    assert "AccidentAgent" in dispatch["domain_results"]
    assert any('"executionMode": "llm_tools"' in f for f in frames)
    p = make_plan(env)
    assert p.metadata["analysisReuse"]["mode"] == "reuse"
    run = run_to_terminal(env, p)
    assert run.status.value == "awaiting_approval"
    assert not env["repo"].list_action_records(run.run_id)
    assert run.state["analysisSnapshot"]["agentEvidenceSnapshots"]["AccidentAgent"]["snapshotDigest"] == output["evidenceSnapshot"]["snapshotDigest"]
    from backend.workflow.executor import WorkflowExecutor
    async def finish():
        executor = WorkflowExecutor(repository=env["repo"])
        await executor.approve(run.run_id, reviewer="local tester", comment="Internal test action")
        async for _ in executor.execute_created_run(run.run_id):
            pass
    asyncio.run(finish())
    completed = env["repo"].get_run(run.run_id)
    assert completed.status.value == "completed"
    actions = env["repo"].list_action_records(run.run_id)
    assert len(actions) == 2 and all(a.status.value == "succeeded" for a in actions)
    assert len(env["client"].requests) == 3  # Workflow never repeats the LLM analysis.


def test_tool_only_document_is_validated_again_at_plan_time(setup, worker):
    collaborate(setup)
    with sqlite3.connect(setup["docs"].RAG_V2_DB_PATH) as conn:
        conn.execute("UPDATE rag_documents SET status='archived' WHERE document_id='tool-doc'")
    with pytest.raises(Exception) as error:
        make_plan(setup)
    assert "analysis_agent_evidence_invalid" in str(error.value)


def test_forged_tool_snapshot_cannot_pass_even_with_recomputed_digest(setup, worker):
    from backend.agent.llm_accident.contracts import digest
    collaborate(setup)
    with sqlite3.connect(setup["db"]) as conn:
        raw = conn.execute("SELECT grounding_context FROM collaboration_runs WHERE run_id='integrated'").fetchone()[0]
        grounding = json.loads(raw)
        snapshot = grounding["agentEvidenceSnapshots"]["AccidentAgent"]
        snapshot["evidence"][-1]["data"]["excerpt"] = "fabricated evidence"
        snapshot["snapshotDigest"] = digest({k: v for k, v in snapshot.items() if k != "snapshotDigest"})
        conn.execute("UPDATE collaboration_runs SET grounding_context=? WHERE run_id='integrated'", (json.dumps(grounding),))
    with pytest.raises(Exception):
        make_plan(setup)


def test_degraded_rule_result_is_visible_and_not_approved_as_llm(setup, monkeypatch, worker):
    from backend.agent.llm_accident import integration
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    monkeypatch.setattr(integration, "AccidentAgentRuntime", lambda **kwargs: AccidentAgentRuntime(client=Client([RuntimeError("provider failed")]), **kwargs))
    run, tasks, frames = collaborate(setup)
    output = json.loads(next(t for t in tasks if t["agent_name"] == "AccidentAgent")["output_snapshot"])
    assert output["executionMode"] == "rule_fallback" and output["degraded"]
    assert json.loads(run["final_decision"])["requiresHumanReview"]
    with pytest.raises(Exception):
        make_plan(setup)


def test_failed_accident_does_not_become_completed_from_other_role(setup, monkeypatch):
    from backend.agent.collaboration import executor
    from backend.agent.llm_accident import integration
    async def fail(*args):
        return None, {"runtimeTrace": {"failureReason": "event_changed_during_analysis"}}
    monkeypatch.setattr(integration, "analyze_bound_accident", fail)
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator
    async def scenario():
        return [s async for s in CollaborationOrchestrator().execute("integrated", "session", setup["event"],
            ["AccidentAgent", "CongestionAgent"], grounding_context=json.loads(setup["collaboration"].get_run("source")["grounding_context"]))]
    asyncio.run(scenario())
    run = setup["collaboration"].get_run("integrated")
    assert run["status"] == "partial_success"
    assert "AccidentAgent" in json.loads(run["failed_agents"])
    with pytest.raises(Exception):
        make_plan(setup)


def test_disconnect_does_not_cancel_background_analysis():
    from backend.agent.collaboration.background_stream import detached_stream, _producers
    async def scenario():
        finished = asyncio.Event()
        async def source():
            yield 'event: start\ndata: {}\n\n'
            await asyncio.sleep(.03)
            finished.set()
            yield 'event: done\ndata: {}\n\n'
        transport = detached_stream(source(), timeout=1, heartbeat=.01)
        await anext(transport)
        await transport.aclose()
        await asyncio.wait_for(finished.wait(), .5)
        await asyncio.sleep(0)
        assert not _producers
    asyncio.run(scenario())


def test_background_has_keepalive_and_finite_deadline():
    from backend.agent.collaboration.background_stream import detached_stream
    async def scenario():
        async def source():
            await asyncio.sleep(1)
            yield "unreachable"
        frames = [f async for f in detached_stream(source(), timeout=.04, heartbeat=.01)]
        assert any(f.startswith(": keepalive") for f in frames)
        assert any("collaboration_timeout" in f for f in frames)
        assert any("event: done" in f for f in frames)
    asyncio.run(scenario())
