"""One isolated process / case / mode. Invoked only by the V2 runner."""
import asyncio
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

from backend.evaluation.v2.providers import EvaluationClient, content_hash, safe_value


def decoded(value, default=None):
    if isinstance(value, str):
        try: return json.loads(value)
        except ValueError: return default
    return value if value is not None else default


async def run_case(request, record, client, effects, frozen, stack):
    from backend.evaluation.v2.sandbox import seed
    from backend import config
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator
    from backend.agent.collaboration.db_repository import SQLiteCollaborationRepository
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    from backend.agent.router import route_agents
    from backend.grounding.assembler import GroundedEventContextAssembler
    from backend.planning import api
    from backend.workflow.executor import WorkflowExecutor
    from backend.workflow.run_driver import RunDriver
    from backend.workflow.repository import SQLiteWorkflowRepository

    case, dataset = request["case"], request["dataset"]
    if case.get("expected", {}).get("inputFidelityFields"):
        record["inputFidelity"] = {"declaredEvent": {
            **deepcopy(dataset["eventDefaults"]), **deepcopy(case.get("event", {}))}}
    event = seed(dataset, case, request["directory"])
    if "inputFidelity" in record:
        from backend.tools.db_tools import get_event_by_id
        stored = get_event_by_id(event["eventId"]) or {}
        record["inputFidelity"]["storedEvent"] = {
            field: deepcopy(stored[field]) for field in case["expected"]["inputFidelityFields"] if field in stored}
    # A small process-local override selects the same runtime as serving code.
    # No Agent, DAG, planner, validator or Workflow algorithm is reimplemented.
    stack.enter_context(patch("backend.agent.llm_accident.integration.AccidentAgentRuntime",
        lambda **kwargs: AccidentAgentRuntime(client=client, **kwargs)))
    stack.enter_context(patch("backend.knowledge.regional_context.EventKnowledgeContextService.get_context_for_event",
        lambda _self, event_id, **kwargs: frozen.get_context_for_event(event_id, **kwargs)))
    if case.get("toolFault"):
        from backend.agent.llm_accident.tools import EventQueries
        from backend.agent.llm_accident.contracts import AgentRuntimeError
        original_query = EventQueries.execute
        def injected_query(queries, name, params):
            if name == case["toolFault"]["tool"]:
                record["toolFaultObserved"] = True
                raise AgentRuntimeError(case["toolFault"]["code"])
            return original_query(queries, name, params)
        stack.enter_context(patch.object(EventQueries, "execute", injected_query))
    event["originalInput"] = case["question"]
    record.update(event=deepcopy(event))
    ids = record["ids"]
    ids["eventId"] = event["eventId"]
    if case.get("entryPoint", "orchestrator") == "routed_api":
        from backend.evaluation.v2.routed_entry import run_routed_entry
        await run_routed_entry(request, event, record, client, stack)
    else:
        grounding = await asyncio.to_thread(GroundedEventContextAssembler(knowledge_service=frozen).assemble, event["eventId"])
        grounding = grounding.to_dict()
        routing = route_agents(event)
        record.update(initialGrounding=deepcopy(grounding), routing=routing)
        record["collaborationFrames"] = [frame async for frame in CollaborationOrchestrator().execute(
            ids["collaborationRunId"], ids["sessionId"], event, routing["selectedAgents"],
            skipped_agents=routing.get("skippedAgents"), routing_reasons=routing.get("reasons"), grounding_context=grounding)]
    collaboration = SQLiteCollaborationRepository()
    source = collaboration.get_run(ids["collaborationRunId"])
    tasks = collaboration.list_tasks(ids["collaborationRunId"])
    record["collaboration"] = source
    record["tasks"] = tasks
    record.setdefault("initialGrounding", decoded(source.get("grounding_context"), {}))
    if record.get("humanReview"):
        record["humanReview"]["events"] = collaboration.list_events(ids["collaborationRunId"])
    accident = next((decoded(t.get("output_snapshot"), {}) for t in tasks if t["agent_name"] == "AccidentAgent"), {})
    selected_accident = any(t["agent_name"] == "AccidentAgent" for t in tasks)
    record.update(actualMode=accident.get("executionMode") if selected_accident else "not_selected",
                  accidentAgentSelected=selected_accident, degraded=accident.get("degraded", False),
                  runtimeTrace=accident.get("runtimeTrace", {}), evidenceSnapshot=accident.get("evidenceSnapshot"))
    record["agentResult"] = {k: deepcopy(v) for k, v in accident.items()
                             if k not in {"runtimeTrace", "evidenceSnapshot", "runtimeProgress"}}
    ids["accidentRunId"] = record["runtimeTrace"].get("runId")
    ids["traceId"] = source.get("trace_id")
    record["workflowStatus"] = "not_started"
    if source["status"] == "awaiting_review":
        record["workflowStatus"] = "awaiting_review"
        return
    if (record.get("humanReview", {}).get("after") or {}).get("status") == "rejected":
        record["workflowStatus"] = "review_rejected"
        return
    try:
        created = await api.create_plan_from_agent(api.AgentPlanRequest(eventId=event["eventId"],
            sessionId=ids["sessionId"], collaborationRunId=ids["collaborationRunId"], reuseAnalysis=True))
    except Exception as exc:
        record["workflowStatus"] = "planning_blocked"
        record["planningError"] = safe_value(getattr(exc, "detail", {"type": type(exc).__name__}))
        return
    record["plan"] = created["plan"]
    ids["planId"] = created["planId"]
    if case.get("stopAfterPlanning"):
        record["workflowStatus"] = "planned"
        return
    repo = SQLiteWorkflowRepository()
    await api.run_plan(ids["planId"], api.PlanRunRequest(sessionId=ids["sessionId"], triggeredBy="evaluation-v2"))
    workflow = repo.list_runs(definition_id=ids["planId"])[0]
    ids["workflowRunId"] = workflow.run_id

    async def no_assessment(_self, _run_id): pass
    stack.enter_context(patch("backend.workflow.executor.get_executor", lambda: WorkflowExecutor(repository=repo)))
    stack.enter_context(patch.object(RunDriver, "_assess_if_terminal", no_assessment))
    driver = RunDriver(repository=repo, lease_seconds=15, heartbeat_interval=1)

    async def drive():
        await driver._poll_once()  # Real claim / lease / heartbeat / executor / release.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            fresh = repo.get_run(workflow.run_id)
            with sqlite3.connect(config.DB_PATH) as conn:
                owner = conn.execute("SELECT driver_owner FROM workflow_runs WHERE run_id=?", (workflow.run_id,)).fetchone()[0]
            if (fresh.is_terminal() or fresh.status.value == "awaiting_approval") and not owner:
                return fresh
            await asyncio.sleep(.02)
        raise TimeoutError("evaluation_workflow_deadline")

    try:
        workflow = await drive()
        record["statusBeforeDecision"] = workflow.status.value
        record["actionsBeforeDecision"] = [a.to_dict() for a in repo.list_action_records(workflow.run_id)]
        record["effectsBeforeDecision"] = deepcopy(effects)
        seen_approvals = set()
        definition = repo.get_definition_version(workflow.definition_id, workflow.version)
        approval_budget = len(definition.definition_json.get("nodes", [])) if definition else 0
        record["approvalDecisions"] = []
        while workflow.status.value == "awaiting_approval":
            pending = workflow.state.get("pendingApproval") or {}
            approval_id = pending.get("approvalId")
            if not approval_id or approval_id in seen_approvals or len(seen_approvals) >= approval_budget:
                raise RuntimeError("evaluation_approval_progress_stalled")
            executor = WorkflowExecutor(repository=repo)
            operation = executor.reject if case["scenario"] == "reject" else executor.approve
            if case["scenario"] == "facts_changed" and not seen_approvals:
                with sqlite3.connect(config.DB_PATH) as conn:
                    conn.execute("UPDATE event_records SET duration=1800 WHERE eventId=?", (event["eventId"],))
            record["approvalResponse"] = await operation(workflow.run_id, reviewer="evaluation-harness",
                comment="Automated fixture decision; no real-world approval", approval_id=approval_id)
            record["approvalDecisions"].append({"approvalId": approval_id, "nodeId": pending.get("nodeId"),
                                                 "response": deepcopy(record["approvalResponse"])})
            if record["approvalResponse"].get("error"):
                raise RuntimeError("evaluation_approval_failed")
            seen_approvals.add(approval_id)
            workflow = repo.get_run(workflow.run_id) if case["scenario"] == "reject" else await drive()
        if case["scenario"] == "approve_repeat" and repo.get_run(workflow.run_id).status.value == "completed":
            before = {"actions": [a.to_dict() for a in repo.list_action_records(workflow.run_id)], "effects": deepcopy(effects)}
            async for _ in WorkflowExecutor(repository=repo).execute_created_run(workflow.run_id): pass
            await driver._poll_once()
            after = {"actions": [a.to_dict() for a in repo.list_action_records(workflow.run_id)], "effects": deepcopy(effects)}
            record["repeatExecution"] = {"before": before, "after": after, "unchanged": before == after}
    finally:
        workflow = repo.get_run(workflow.run_id)
        record.update(workflowStatus=workflow.status.value, workflow=workflow.to_dict(),
            nodes=[n.to_dict() for n in repo.get_node_runs(workflow.run_id)],
            approvals=[a.to_dict() for a in repo.list_approvals(workflow.run_id)],
            actions=[a.to_dict() for a in repo.list_action_records(workflow.run_id)],
            workflowEvents=[e.to_dict() for e in repo.list_events(workflow.run_id)])


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root = Path(request["directory"]).resolve()
    # Set BEFORE importing any serving module; .env uses override=False.
    for key, value in {"TRAFFICMIND_DB_PATH": root / "events.db", "RAG_V2_DB_PATH": root / "rag.db",
        "RAG_V2_FTS_PATH": root / "fts.db", "RAG_V2_VECTOR_DB_PATH": root / "vectors"}.items():
        os.environ[key] = str(value)
    os.environ["ACCIDENT_AGENT_MODE"] = request["mode"]
    os.environ["RAG_ALLOW_MODEL_DOWNLOAD"] = "false"
    os.environ["MEMORY_STORAGE_BACKEND"] = "sqlite"
    from backend.evaluation.v2.sandbox import assert_isolated_paths, FrozenKnowledge, action_registry
    from backend import config
    from backend.evaluation.v2.checks import evaluate
    assert_isolated_paths(root)
    if request["transport"] == "live" and request["mode"] != "rule" and not config.LLM_ENABLED:
        raise RuntimeError("live_evaluation_requires_configured_api_key")
    record = {"schemaVersion": "trafficmind-eval-record-v2", "caseId": request["case"]["caseId"],
        "name": request["case"]["name"], "scenario": request["case"]["scenario"], "modeRequested": request["mode"],
        "expected": deepcopy(request["case"]["expected"]),
        "datasetSplit": request["case"].get("split", request["dataset"].get("split", "acceptance")),
        "transport": request["transport"], "datasetHash": content_hash(request["dataset"]),
        "ids": {"evaluationRunId": request["evaluationRunId"],
                "caseRunId": request["evaluationRunId"] + "-" + request["case"]["caseId"] + "-" + request["mode"],
                "sessionId": "eval-session-" + request["case"]["caseId"],
                "collaborationRunId": "eval-collaboration-" + request["case"]["caseId"]},
        "profile": {"knowledge": "frozen_fixture_service", "fusion": "production_template_branch",
                    "actions": "isolated_executors", "postTerminalAssessment": "disabled",
                    "history": "isolated_database", "cases": "empty_fixture_database",
                    "entryPoint": request["case"].get("entryPoint", "orchestrator")}}
    effects, frozen = [], FrozenKnowledge(request["dataset"], request["case"])
    if request["case"].get("historicalCases"): record["profile"]["cases"] = "seeded_case_repository"
    client = EvaluationClient(request["transport"], request["case"], cassette=request.get("cassette"),
                              inject_failure=request["case"]["scenario"] == "provider_failure")
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            from datetime import datetime
            class FixtureCaptureClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    fixed = datetime.fromisoformat(request["dataset"]["asOf"])
                    return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
            # Pin only the hydration capture label. Runtime/lease deadlines use
            # their original clocks; historical asOf is separately seeded.
            stack.enter_context(patch("backend.tools.event_identity.datetime", FixtureCaptureClock))
            # Keep identical, real template fusion in every arm; avoid an
            # additional model confound / unrecorded network call.
            stack.enter_context(patch("backend.agent.collaboration.orchestrator.LLM_ENABLED", False))
            stack.enter_context(patch("backend.workflow.action_execution._executor_registry", action_registry(effects)))
            if request["transport"] != "live":
                import socket
                original_connect = socket.socket.connect
                def offline_connect(sock, address):
                    # Windows asyncio creates a loopback socket pair internally.
                    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                        return original_connect(sock, address)
                    raise RuntimeError("offline_evaluation_network_blocked")
                stack.enter_context(patch.object(socket.socket, "connect", offline_connect))
            asyncio.run(run_case(request, record, client, effects, frozen, stack))
            client.assert_consumed()
    except Exception as exc:
        import traceback
        record["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", type(exc).__name__),
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(exc.__traceback__)]}
    finally:
        record["durationMs"] = round((time.monotonic() - started) * 1000)
        record["isolatedEffects"] = effects
        record["knowledgeCalls"] = frozen.calls
        record["modelRequests"] = len(client.entries)
        record["llmPathCompleted"] = bool(record.get("actualMode") == request["mode"]
            and request["mode"] != "rule" and not record.get("degraded") and client.entries
            and record.get("runtimeTrace", {}).get("status") == "completed")
        record["injectedFault"] = request["case"]["scenario"] == "provider_failure" and request["mode"] != "rule"
        record["faultInjectionObserved"] = bool(record["injectedFault"] and len(client.entries) == 1
                                               and client.entries[0].get("error") == "RuntimeError")
        from backend.evaluation.v2.input_fidelity import capture_model_inputs
        capture_model_inputs(record, client.entries)
        record["checks"] = evaluate(request["case"], record)
        record["passed"] = all(c["passed"] for c in record["checks"])
        for name, value in (("record.json", record), ("cassette.json", client.entries)):
            (root / name).write_text(json.dumps(safe_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"caseId": record["caseId"], "mode": record["modeRequested"], "passed": record["passed"]}))


if __name__ == "__main__":
    main()
