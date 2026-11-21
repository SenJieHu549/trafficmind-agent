"""DAG adapter and source verification; never authorizes or executes actions."""
from copy import deepcopy

from backend.agent.llm_accident.contracts import AgentRuntimeError, EvidenceLedger, RuntimeLimits, digest
from backend.agent.llm_accident.runtime import AccidentAgentRuntime
from backend.agent.llm_accident.tools import EventQueries


def configured_mode():
    from backend import config
    if config.ACCIDENT_AGENT_MODE not in ("rule", "llm_context", "llm_tools"):
        raise AgentRuntimeError("invalid_accident_agent_mode")
    return config.ACCIDENT_AGENT_MODE


async def analyze_bound_accident(task, state, repo):
    from backend import config
    event_id = state.normalized_event.get("eventId")
    if not event_id:
        raise AgentRuntimeError("event_id_required")
    limits = RuntimeLimits(tool_timeout=config.ACCIDENT_TOOL_TIMEOUT_SECONDS,
                           llm_timeout=config.ACCIDENT_LLM_TIMEOUT_SECONDS,
                           overall_timeout=config.ACCIDENT_RUNTIME_TIMEOUT_SECONDS,
                           max_output_repairs=config.ACCIDENT_OUTPUT_REPAIR_ATTEMPTS)

    def progress(item):
        import uuid
        from datetime import datetime, timezone
        item = {**item, "updatedAt": datetime.now(timezone.utc).isoformat()}
        task.output_snapshot = {"runtimeProgress": item}
        repo.update_task(task.run_id, task.to_dict())
        repo.save_event(task.run_id, {"event_id": "accident_" + uuid.uuid4().hex,
            "event_type": "accident_runtime", "taskId": task.task_id, **item}, 0)

    from backend.agent.collaboration.roles import get_agent_capability
    from backend.agent.collaboration.context_projection import project_context_for_agent
    memory = project_context_for_agent(state.to_dict(), "AccidentAgent").get("memoryContext")
    memory_kwargs = {"memory_context": memory} if memory else {}
    report = await AccidentAgentRuntime(limits=limits,
        allowed_tools=get_agent_capability("AccidentAgent")["allowed_tools"]).run(event_id,
        question=state.normalized_event.get("originalInput") or "分析此事故，按需查证并提出有依据的处置建议。",
        grounding=deepcopy(state.grounding_context), mode=configured_mode(), on_event=progress,
        **memory_kwargs)
    trace = report["trace"]
    diagnostics = {"executionMode": trace["generationMode"], "degraded": trace["degraded"],
                   "fallbackReason": trace.get("fallbackReason"), "runtimeTrace": trace,
                   "evidenceSnapshot": report.get("evidenceSnapshot")}
    if not report.get("agentResult"):
        return None, diagnostics
    # Validate against durable sources before exposing this snapshot downstream.
    import asyncio
    try:
        await asyncio.to_thread(validate_evidence_snapshot, report["evidenceSnapshot"], event_id)
    except Exception as exc:
        trace.update(status="failed", failureReason=getattr(exc, "code", "evidence_validation_failed"))
        diagnostics["runtimeTrace"] = trace
        return None, diagnostics
    result = deepcopy(report["agentResult"])
    result.update(task_id=task.task_id, duration_ms=trace["durationMs"])
    return result, diagnostics


def validate_evidence_snapshot(snapshot, event_id):
    """Re-resolve sources without embedding, retrieval, or a model call.

    Conservative invalidation: a changed source requires a new analysis. A hash
    alone is never accepted as proof of an actual database source.
    """
    if not isinstance(snapshot, dict) or snapshot.get("schemaVersion") != "accident-evidence-v1":
        raise AgentRuntimeError("accident_snapshot_invalid")
    body = {k: v for k, v in snapshot.items() if k != "snapshotDigest"}
    if snapshot.get("snapshotDigest") != digest(body) or snapshot.get("eventId") != event_id:
        raise AgentRuntimeError("accident_snapshot_invalid")
    queries = EventQueries(event_id)
    from backend.planning.analysis_handoff import _time, _resolve_evidence
    if _time(snapshot.get("asOf")) != _time(queries.as_of):
        raise AgentRuntimeError("accident_snapshot_time_changed")
    ledger = EvidenceLedger(event_id, snapshot["asOf"])
    knowledge = []
    cached = {}
    for item in snapshot.get("evidence", []):
        kind, source_id, data = item["kind"], item["sourceId"], item["data"]
        if kind == "current_event":
            valid = source_id == event_id and data == queries.facts
        elif kind == "knowledge":
            resolved = _resolve_evidence([data], queries.facts, _time(queries.as_of))
            # Old snapshots only exposed the child. Do not retroactively add
            # parent data the model never saw, or change their evidence IDs.
            actual = queries._knowledge_source(
                resolved[0], include_parent_context="relatedContext" in data)
            valid = source_id == actual["sourceId"] and data == actual["data"]
            knowledge.append({**deepcopy(data), "evidenceId": item["evidenceId"], "origin": "accident_agent"})
        elif kind in ("regional_context", "case"):
            tool = "accident_road_context" if kind == "regional_context" else "accident_case_search"
            if tool not in cached:
                cached[tool] = queries.execute(tool, {} if kind == "regional_context" else {"limit": 5})["sources"]
            valid = any(s["sourceId"] == source_id and s["data"] == data for s in cached[tool])
        else:
            valid = False
        if not valid:
            raise AgentRuntimeError("accident_evidence_source_changed")
        origins = item.get("origins") or []
        if not origins:
            raise AgentRuntimeError("accident_snapshot_invalid")
        for origin in origins:
            actual_id = ledger.add(kind, source_id, data, origin)
        if actual_id != item.get("evidenceId") or digest({k: item[k] for k in
            ("kind", "sourceId", "eventId", "asOf", "data")}) != item.get("contentHash"):
            raise AgentRuntimeError("accident_snapshot_invalid")
    if ledger.freeze(snapshot.get("citedEvidenceIds", [])) != snapshot:
        raise AgentRuntimeError("accident_snapshot_invalid")
    if not any(e["kind"] == "current_event" for e in ledger.items()):
        raise AgentRuntimeError("accident_snapshot_invalid")
    return knowledge
