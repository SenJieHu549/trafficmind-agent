"""Versioned, source-verified analysis handoff. No retrieval or model calls.

The snapshot is a frozen input, not an authorization. Every action still uses
the existing policy/approval/idempotency machinery and rechecks live facts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Dict


class AnalysisHandoffError(ValueError):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(f"{code}: {message}")


# Deliberately exact in v1. No guessed tolerance or wall-clock extrapolation of
# duration; lifecycle status/capturedAt/updatedAt are checked separately.
FACT_FIELDS = (
    "eventId", "eventType", "roadName", "direction", "avgSpeed", "queueLength",
    "duration", "vehicleCount", "weather", "timePeriod", "isMainRoad",
    "nearbySchool", "nearbyHospital", "riskScore", "riskLevel",
)
NUMERIC_FIELDS = {"avgSpeed", "queueLength", "duration", "vehicleCount", "riskScore"}
BOOL_FIELDS = {"isMainRoad", "nearbySchool", "nearbyHospital"}


def _json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def event_facts(event: Dict[str, Any]) -> Dict[str, Any]:
    from backend.tools.event_tools import EVENT_TYPE_MAP
    reverse = {v: k for k, v in EVENT_TYPE_MAP.items()}
    facts = {}
    for key in FACT_FIELDS:
        if key not in event:
            raise AnalysisHandoffError("analysis_baseline_missing", f"缺少研判事实字段 {key}，请重新研判")
        value = event[key]
        if key in NUMERIC_FIELDS and value not in (None, ""):
            try:
                if isinstance(value, bool):
                    raise ValueError()
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError()
            except (TypeError, ValueError):
                raise AnalysisHandoffError("analysis_facts_invalid", f"非法数值字段 {key}")
        elif key in BOOL_FIELDS:
            if value not in (True, False, 0, 1):
                raise AnalysisHandoffError("analysis_facts_invalid", f"非法布尔字段 {key}")
            value = bool(value)
        elif isinstance(value, str):
            value = value.strip()
        if key == "eventType":
            value = reverse.get(value, value)
        facts[key] = value if value != "" or key not in NUMERIC_FIELDS else None
    if not all(facts.get(k) for k in ("eventId", "eventType", "roadName")):
        raise AnalysisHandoffError("analysis_baseline_missing", "缺少事件身份、类型或道路")
    if isinstance(event.get("reportedFacts"), dict) and event["reportedFacts"]:
        facts["reportedFacts"] = deepcopy(event["reportedFacts"])
    return facts


def _time(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # Existing event timestamps without offset are local application time.
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        raise AnalysisHandoffError("analysis_time_invalid", "研判时间依据无法解析")


def _document_stamp(doc, chunk):
    return _digest({"document": doc.model_dump(mode="json"),
                    "chunk": chunk.model_dump(mode="json")})


def _resolve_evidence(items, baseline, as_of):
    from backend.rag.v2.document_repository import get_document, get_chunk
    from backend.knowledge.regional_context import _effective_at, _location_match, event_type_applicable
    from backend.regional.repository import SQLiteRegionalRepository
    binding = SQLiteRegionalRepository().get_active_event_location_binding(baseline["eventId"])
    accepted = []
    seen = set()
    for item in items:
        doc_id, chunk_id = item.get("documentId"), item.get("chunkId")
        identity = (doc_id, chunk_id)
        if identity in seen:
            continue
        seen.add(identity)
        if not doc_id or not chunk_id:
            raise AnalysisHandoffError("analysis_evidence_unresolved", "证据缺少文档或分块身份")
        doc, chunk = get_document(doc_id), get_chunk(chunk_id)
        if doc is None or chunk is None or chunk.document_id != doc_id or doc.status != "active":
            raise AnalysisHandoffError("analysis_evidence_unresolved", "证据来源已缺失或不可用")
        if not event_type_applicable(doc.event_type, baseline["eventType"]):
            raise AnalysisHandoffError("analysis_evidence_inapplicable", "证据不适用于事件类型")
        if not _location_match(doc, binding)[0]:
            raise AnalysisHandoffError("analysis_evidence_inapplicable", "证据不适用于当前区域绑定")
        if not _effective_at(doc, as_of) or not _effective_at(doc, datetime.now(timezone.utc)):
            raise AnalysisHandoffError("analysis_evidence_expired", "证据不满足研判或执行时的有效期")
        excerpt = str(item.get("excerpt") or item.get("content") or "").strip()
        probe = excerpt[:-3].rstrip() if excerpt.endswith("...") else excerpt
        if not probe or probe not in (chunk.raw_content or "") and probe not in (chunk.contextual_content or ""):
            raise AnalysisHandoffError("analysis_evidence_changed", "证据摘录无法在原分块中核对")
        stamp = _document_stamp(doc, chunk)
        accepted.append({**deepcopy(item), "documentId": doc_id, "chunkId": chunk_id,
                         "excerpt": excerpt, "sourceStamp": stamp, "origin": "collaboration"})
    if not accepted:
        raise AnalysisHandoffError("analysis_evidence_missing", "没有可复用的正式知识证据，请补充依据后重新研判")
    return accepted


def build_analysis_snapshot(event_id: str, session_id: str, run_id: str, *, repo=None) -> Dict[str, Any]:
    from backend.agent.collaboration.db_repository import SQLiteCollaborationRepository
    repo = repo or SQLiteCollaborationRepository()
    run = repo.get_run(run_id)
    if not run or run.get("status") != "completed":
        raise AnalysisHandoffError("analysis_source_not_completed", "复用需要已完成的协同研判")
    event = _json(run.get("normalized_event"), {})
    if event.get("eventId") != event_id or session_id != run.get("session_id", ""):
        raise AnalysisHandoffError("analysis_identity_mismatch", "来源研判不属于当前事件或会话")
    if event.get("runKind") == "replay" or event.get("actionExecutionAllowed") is False:
        raise AnalysisHandoffError("analysis_execution_forbidden", "回放研判不可授权业务执行")
    grounding = _json(run.get("grounding_context"), {})
    baseline = grounding.get("currentEvent") or {}
    facts = event_facts(baseline)  # Never manufacture past facts from today's row.
    if facts["eventId"] != event_id:
        raise AnalysisHandoffError("analysis_identity_mismatch", "grounding 事件身份不一致")
    as_of = baseline.get("createdAt")
    _time(as_of)
    final = _json(run.get("final_decision"), run.get("final_decision") or "")
    if not final:
        raise AnalysisHandoffError("analysis_conclusion_missing", "来源研判缺少最终结论")
    if isinstance(final, dict) and final.get("requiresHumanReview"):
        raise AnalysisHandoffError("analysis_review_required", "来源研判仍要求人工复核，请复核并重新生成有效研判")
    tasks = sorted(repo.list_tasks(run_id), key=lambda t: t["task_id"])
    outputs = {}
    task_snapshots = {}
    for task in tasks:
        if task.get("status") != "succeeded":
            continue
        output = _json(task.get("output_snapshot"), {})
        summary = output.get("suggestion") or output.get("recommendation")
        if not summary:
            continue
        confidence = output.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            confidence = None
        outputs[task["agent_name"]] = {
            "summary": summary, "confidence": confidence,
            "evidenceRefs": deepcopy(output.get("evidence_refs") or output.get("evidenceRefs") or []),
            "sourceTaskId": task["task_id"],
        }
        if output.get("evidenceSnapshot"):
            task_snapshots[task["agent_name"]] = output["evidenceSnapshot"]
    knowledge = deepcopy((grounding.get("knowledgeContext") or {}).get("evidence") or [])
    agent_snapshots = deepcopy(grounding.get("agentEvidenceSnapshots") or {})
    for agent, agent_snapshot in agent_snapshots.items():
        if task_snapshots.get(agent) != agent_snapshot:
            raise AnalysisHandoffError("analysis_evidence_mismatch", "协作记录与 Agent 任务证据快照不一致")
        from backend.agent.llm_accident.integration import validate_evidence_snapshot
        try:
            knowledge.extend(validate_evidence_snapshot(agent_snapshot, event_id))
        except ValueError as exc:
            raise AnalysisHandoffError("analysis_agent_evidence_invalid", "事故 Agent 的工具证据已变化或无效，请重新研判") from exc
        output_refs = outputs.get(agent, {}).get("evidenceRefs", [])
        if output_refs != agent_snapshot.get("citedEvidenceIds"):
            raise AnalysisHandoffError("analysis_evidence_mismatch", "Agent 引用与冻结证据不一致")
    evidence = _resolve_evidence(knowledge, facts, _time(as_of))
    from backend.agent.collaboration.human_review import reviewed_analysis, ReviewError
    try:
        reviewed = reviewed_analysis(run_id, final)
    except ReviewError as err:
        raise AnalysisHandoffError(err.code, str(err)) from err
    if reviewed is not None:
        # Original proposals remain in task audit, but no superseded proposal
        # may silently re-enter Planning as a current recommendation.
        outputs = {}
    # Frozen final is surfaced in close/save_result, not only as an unused audit.
    confidence = final.get("confidence") if isinstance(final, dict) else None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = None
    outputs["SourceFusion"] = {"summary": final if isinstance(final, str) else json.dumps(final, ensure_ascii=False),
                               "confidence": confidence, "evidenceRefs": [e["chunkId"] for e in evidence]}
    source_stamp = _digest({"event": event, "grounding": grounding, "final": final,
                            "tasks": [{k: t.get(k) for k in ("task_id", "agent_name", "status", "output_snapshot")} for t in tasks]})
    location = (grounding.get("regionalContext") or {}).get("location") or {}
    snapshot = {"schemaVersion": 1, "factPolicy": "exact-v1", "eventId": event_id,
                "sessionId": session_id, "collaborationRunId": run_id,
                "completedAt": run.get("completed_at"), "asOf": as_of,
                "facts": facts, "sourceStamp": source_stamp,
                "location": {k: location.get(k) or None for k in ("regionId", "roadId", "intersectionId")},
                "finalDecision": final, "agentOutputs": outputs, "evidence": evidence}
    if agent_snapshots:
        snapshot["agentEvidenceSnapshots"] = agent_snapshots
    snapshot["snapshotDigest"] = _digest(snapshot)
    return snapshot


def validate_analysis_snapshot(snapshot: Dict[str, Any], current_event: Dict[str, Any], *, repo=None):
    from backend.tools.event_identity import hydrate_authoritative_event, compact_event_context, ensure_event_open_for_execution
    if not isinstance(snapshot, dict) or snapshot.get("schemaVersion") != 1 or snapshot.get("factPolicy") != "exact-v1":
        raise AnalysisHandoffError("analysis_snapshot_invalid", "不支持的研判快照")
    body = {k: v for k, v in snapshot.items() if k != "snapshotDigest"}
    if snapshot.get("snapshotDigest") != _digest(body):
        raise AnalysisHandoffError("analysis_snapshot_invalid", "研判快照摘要不匹配")
    event_id = snapshot["eventId"]
    if current_event.get("eventId") != event_id:
        raise AnalysisHandoffError("analysis_identity_mismatch", "执行事件与研判快照不一致")
    # Rebuild from durable source: an arbitrary client-generated hash is not authority.
    fresh = build_analysis_snapshot(event_id, snapshot["sessionId"], snapshot["collaborationRunId"], repo=repo)
    if fresh["snapshotDigest"] != snapshot["snapshotDigest"]:
        raise AnalysisHandoffError("analysis_source_changed", "来源研判或证据已变化，请重新生成方案")
    authoritative = hydrate_authoritative_event(event_id)
    ensure_event_open_for_execution(authoritative)
    from backend.regional.repository import SQLiteRegionalRepository
    binding = SQLiteRegionalRepository().get_active_event_location_binding(event_id) or {}
    if {k: binding.get(k) or None for k in ("regionId", "roadId", "intersectionId")} != snapshot["location"]:
        raise AnalysisHandoffError("analysis_location_changed", "事件区域绑定已变化，请重新研判")
    current = event_facts(compact_event_context(authoritative))
    changed = [key for key in set(current) | set(snapshot["facts"])
               if current.get(key) != snapshot["facts"].get(key)]
    if changed:
        raise AnalysisHandoffError("analysis_facts_changed", "关键事实已变化，请重新研判: " + ", ".join(changed))
    return {"mode": "reused", "valid": True, "collaborationRunId": snapshot["collaborationRunId"],
            "snapshotDigest": snapshot["snapshotDigest"], "evidenceCount": len(snapshot["evidence"]),
            "factPolicy": "exact-v1"}
