"""Durable conflict decisions and deterministic fusion, committed in one transaction.

No coroutine waits for a human. Decisions never approve external actions, and
resuming fusion never repeats domain/model/tool execution.
"""
from copy import deepcopy
import hashlib
import json
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from backend.agent.collaboration import db_repository as db
from backend.workflow.action_execution import contains_sensitive_key, sanitize_public_text


class ReviewError(ValueError):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status


class ConflictResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    conflictId: str = Field(min_length=1, max_length=200)
    choice: Literal["select", "edit"]
    selectedAgent: str = Field(default="", max_length=100)
    resolution: str = Field(default="", max_length=2000)


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expectedVersion: int = Field(ge=1)
    action: Literal["resolve", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str = Field(default="", max_length=1000)
    resolutions: list[ConflictResolution] = Field(default_factory=list, max_length=20)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def needs_review(state):
    return any(c.get("severity") in ("high", "critical") for c in state.conflicts) or any(
        not a.get("resolved") or a.get("requires_human_review") for a in state.arbitration_results)


def _stamp(conn, run_id):
    run = conn.execute("SELECT * FROM collaboration_runs WHERE run_id=?", (run_id,)).fetchone()
    tasks = conn.execute("SELECT * FROM collaboration_tasks WHERE run_id=? ORDER BY task_id", (run_id,)).fetchall()
    return digest({"event": json.loads(run["normalized_event"]),
                   "grounding": json.loads(run["grounding_context"]),
                   "tasks": [{key: task[key] for key in (
                       "task_id", "agent_name", "status", "input_snapshot", "output_snapshot")}
                       for task in tasks]})


def _event_stamp(conn, event_id):
    if not event_id:
        return None
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_records'").fetchone()
    row = conn.execute("SELECT * FROM event_records WHERE eventId=?", (event_id,)).fetchone() if exists else None
    return digest(dict(row)) if row else None


def _audit(conn, run_id, review_id, events, now):
    seq = conn.execute("SELECT COALESCE(MAX(sequence_number),0) FROM collaboration_events WHERE run_id=?",
                       (run_id,)).fetchone()[0]
    for offset, (kind, payload) in enumerate(events, 1):
        conn.execute("INSERT INTO collaboration_events VALUES (?,?,?,?,?,?)",
            (f"{review_id}:{kind}", run_id, kind, _json({"runId": run_id, **payload}), seq + offset, now))


def _public(row):
    checkpoint = json.loads(row["checkpoint_json"])
    return {"reviewId": row["review_id"], "runId": row["run_id"], "version": row["version"],
            "status": row["status"], "conflicts": checkpoint["conflicts"],
            "createdAt": row["created_at"], "decidedAt": row["decided_at"],
            "decision": json.loads(row["decision_json"]), "externalActionsApproved": False}


def get_review(run_id):
    db.init_collaboration_tables()
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT * FROM collaboration_reviews WHERE run_id=?", (run_id,)).fetchone()
        return _public(row) if row else None
    finally:
        conn.close()


def pause_for_review(state):
    """Freeze the already persisted role outputs; atomically pause before fusion."""
    db.init_collaboration_tables()
    conn = db.get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM collaboration_runs WHERE run_id=?", (state.run_id,)).fetchone()
        existing = conn.execute("SELECT * FROM collaboration_reviews WHERE run_id=?", (state.run_id,)).fetchone()
        if existing:
            return _public(existing)
        if not row or row["status"] != "arbitrating":
            raise ReviewError("review_run_not_ready", "当前研判不能进入人工裁决")
        now, review_id = db._utc_now_iso(), "review_" + uuid.uuid4().hex
        conn.execute("UPDATE collaboration_tasks SET status='blocked', completed_at='', error_message='awaiting_review' "
                     "WHERE run_id=? AND task_id='task_fusion'", (state.run_id,))
        event_id = state.normalized_event.get("eventId", "")
        checkpoint = {"conflicts": deepcopy(state.conflicts), "results": deepcopy(state.task_results),
                      "arbitration": deepcopy(state.arbitration_results),
                      "eventId": event_id, "eventStamp": _event_stamp(conn, event_id),
                      "sourceStamp": _stamp(conn, state.run_id),
                      "maxRunRowId": conn.execute("SELECT COALESCE(MAX(rowid),0) FROM collaboration_runs").fetchone()[0]}
        conn.execute("INSERT INTO collaboration_reviews (review_id,run_id,checkpoint_json,checkpoint_digest,created_at) "
                     "VALUES (?,?,?,?,?)", (review_id, state.run_id, _json(checkpoint), digest(checkpoint), now))
        final = {"fusionSummary": "存在待人工裁决的建议冲突，已保存研判结果，等待处理。",
                 "requiresHumanReview": True, "humanReviewId": review_id, "actionPlan": []}
        conn.execute("UPDATE collaboration_runs SET status='awaiting_review', final_decision=?, updated_at=?, "
                     "completed_at='' WHERE run_id=?", (_json(final), now, state.run_id))
        _audit(conn, state.run_id, review_id, [("run_awaiting_review", {"reviewId": review_id, "version": 1})], now)
        conn.commit()
        return get_review(state.run_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _resolutions(checkpoint, request):
    conflicts = {c["id"]: c for c in checkpoint["conflicts"]}
    if len(request.resolutions) != len(conflicts) or {r.conflictId for r in request.resolutions} != set(conflicts):
        raise ReviewError("review_incomplete", "必须逐项处理全部冲突，且不能重复提交冲突项", 422)
    resolved = []
    for choice in sorted(request.resolutions, key=lambda item: item.conflictId):
        conflict = conflicts[choice.conflictId]
        if choice.choice == "select":
            options = {p["agentName"]: p for p in conflict["proposals"]}
            if choice.selectedAgent not in options or choice.resolution:
                raise ReviewError("review_invalid_choice", "只能选择本冲突冻结的角色方案", 422)
            text = options[choice.selectedAgent]["suggestion"]
        else:
            if choice.selectedAgent or not choice.resolution.strip():
                raise ReviewError("review_invalid_edit", "人工修改方案不能为空，且不能同时选择原方案", 422)
            text = sanitize_public_text(choice.resolution).strip()
            if not text:
                raise ReviewError("review_invalid_edit", "人工修改方案不能为空", 422)
        resolved.append({"conflictId": choice.conflictId, "choice": choice.choice,
                         "selectedAgent": choice.selectedAgent, "resolution": text,
                         "participants": conflict["agents"]})
    return resolved


def _effective_analysis(checkpoint, resolutions, review_id):
    """Replace whole conflicted outputs, retaining independent domain results.

    Pre-review dispatch/fusion outputs depend on the superseded opinions and
    cannot safely be reused. No sentence-level semantic filtering is attempted.
    """
    from backend.agent.collaboration.roles import REGISTERED_AGENTS

    excluded = {a for c in checkpoint["conflicts"] for a in c["agents"]}
    excluded.update({"DispatchAgent", "ConflictDetector", "ConflictArbiter", "FusionAgent"})
    # Also invalidate outputs with declared dependencies on an excluded role.
    while True:
        dependent = {name for name, role in REGISTERED_AGENTS.items()
                     if set(role.get("dependencies", [])) & excluded}
        if dependent <= excluded:
            break
        excluded.update(dependent)
    outputs = []
    for name, result in sorted(checkpoint["results"].items()):
        if name in excluded or name not in REGISTERED_AGENTS or result.get("status") not in ("completed", "partial"):
            continue
        output = {key: deepcopy(result[key]) for key in (
            "findings", "suggestion", "recommendation", "confidence", "urgency",
            "evidence_refs", "proposed_actions", "assumptions") if key in result}
        output.update(agent_name=name, task_id=result.get("task_id", ""),
                      sourceVersion=digest(result), origin="retained_domain_result")
        outputs.append(output)
    conflicts = {c["id"]: c for c in checkpoint["conflicts"]}
    for resolution in resolutions:
        proposals = conflicts[resolution["conflictId"]]["proposals"]
        if resolution["choice"] == "select":
            proposals = [p for p in proposals if p["agentName"] == resolution["selectedAgent"]]
        refs = { _json(ref): deepcopy(ref) for p in proposals for ref in p.get("evidenceRefs", []) }
        outputs.append({"agent_name": "HumanReview", "task_id": f"{review_id}:{resolution['conflictId']}",
                        "origin": "human_resolution", "conflictId": resolution["conflictId"],
                        "sourceTaskIds": [p["taskId"] for p in proposals],
                        "sourceVersions": [p["resultVersion"] for p in proposals],
                        "findings": [], "suggestion": resolution["resolution"],
                        "proposed_actions": [], "evidence_refs": list(refs.values())})
    return {"schemaVersion": 1, "reviewId": review_id, "outputs": outputs,
            "supersededAgents": sorted(set(checkpoint["results"]) & excluded)}


def reviewed_analysis(run_id, final):
    """Return only a projection bound to an accepted durable review, or None.

    Legacy reviewed runs without a projection require fresh analysis; rebuilding
    from today's raw tasks would silently change a previously frozen decision.
    """
    review = get_review(run_id)
    marked = isinstance(final, dict) and (final.get("humanReview") or final.get("humanReviewId"))
    if not review and not marked:
        return None
    if (not review or review["status"] != "accepted" or not isinstance(final, dict)
            or review["decision"].get("finalDecision") != final):
        raise ReviewError("analysis_review_invalid", "人工裁决记录与融合结果不一致或尚未完成")
    projection = final.get("effectiveAnalysis")
    if (not isinstance(projection, dict) or projection.get("schemaVersion") != 1
            or projection.get("reviewId") != review["reviewId"]
            or not isinstance(projection.get("outputs"), list)):
        raise ReviewError("analysis_review_projection_missing", "历史裁决缺少有效输入快照，请重新研判")
    return deepcopy(projection)


def _fuse(checkpoint, resolutions, review_id, reviewer, comment, now):
    """Use explicit human resolutions; do not restate superseded role proposals."""
    human = {"reviewId": review_id, "version": 2, "reviewer": reviewer, "comment": comment,
             "decidedAt": now, "resolutions": resolutions, "externalActionsApproved": False}
    summary = "人工裁决已完成，本次融合以以下决定为准：\n" + "\n".join(
        f"[{r['conflictId']}] {r['resolution']}" for r in resolutions)
    projection = _effective_analysis(checkpoint, resolutions, review_id)
    retained = [o for o in projection["outputs"] if o["origin"] == "retained_domain_result"]
    if retained:
        summary += "\n保留未受冲突影响的领域研判（包含待核实项）：\n" + "\n".join(
            f"[{o['agent_name']}] " + "；".join(dict.fromkeys(
                [*o.get("findings", []), o.get("suggestion") or o.get("recommendation") or "",
                 *o.get("assumptions", [])])) for o in retained)
    degraded = any(r.get("degraded") for r in checkpoint["results"].values())
    return {"fusionSummary": summary, "generationMode": "human_review_template",
            "requiresHumanReview": degraded, "humanReview": human, "effectiveAnalysis": projection,
            "actionPlan": [], "limitations": ["人工裁决仅解决研判分歧，外部动作仍须单独审批"],
            "arbitration": {"humanResolutions": resolutions}}


def decide(run_id, review_id, request: ReviewDecision):
    """CAS + deterministic fusion + event/task/audit updates in one transaction."""
    body = request.model_dump()
    if contains_sensitive_key(body):
        raise ReviewError("review_sensitive_fields", "裁决内容包含禁止保存的敏感字段", 422)
    reviewer, comment = sanitize_public_text(request.reviewer).strip(), sanitize_public_text(request.comment).strip()
    if not reviewer or (request.action == "reject" and (not comment or request.resolutions)):
        raise ReviewError("review_invalid_decision", "请填写处理人；驳回时需填写原因且不提交方案", 422)
    request_digest = digest(body)
    db.init_collaboration_tables()
    conn = db.get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM collaboration_reviews WHERE run_id=? AND review_id=?",
                           (run_id, review_id)).fetchone()
        if not row:
            raise ReviewError("review_not_found", "未找到本次研判的裁决单", 404)
        if row["status"] != "pending":
            if row["request_digest"] == request_digest:
                return {"review": _public(row), "duplicate": True}
            raise ReviewError("review_already_decided", "裁决已处理，请刷新后查看结果")
        if request.expectedVersion != row["version"]:
            raise ReviewError("review_version_conflict", "裁决版本已变化，请刷新后重试")
        run = conn.execute("SELECT * FROM collaboration_runs WHERE run_id=?", (run_id,)).fetchone()
        checkpoint = json.loads(row["checkpoint_json"])
        if (not run or run["status"] != "awaiting_review" or digest(checkpoint) != row["checkpoint_digest"]
                or _stamp(conn, run_id) != checkpoint["sourceStamp"]
                or _event_stamp(conn, checkpoint["eventId"]) != checkpoint["eventStamp"]):
            raise ReviewError("review_source_changed", "研判或事件资料已变化，请重新研判")
        newer = conn.execute("SELECT 1 FROM collaboration_runs WHERE rowid>? AND run_id!=? "
            "AND json_valid(normalized_event) AND json_extract(normalized_event,'$.eventId')=? "
            "AND COALESCE(json_extract(normalized_event,'$.runKind'),'live')!='replay' LIMIT 1",
            (checkpoint["maxRunRowId"], run_id, checkpoint["eventId"])).fetchone() if checkpoint["eventId"] else None
        if newer:
            raise ReviewError("review_superseded", "该事件已有更新的研判，请处理最新结果")
        now = db._utc_now_iso()
        resolutions = _resolutions(checkpoint, request) if request.action == "resolve" else []
        if request.action == "resolve":
            final = _fuse(checkpoint, resolutions, review_id, reviewer, comment, now)
            grounding = json.loads(run["grounding_context"])
            final["agentEvidenceSnapshots"] = grounding.get("agentEvidenceSnapshots", {})
            final["agentExecutionModes"] = {name: r.get("executionMode", "rule") for name, r in checkpoint["results"].items()}
            if grounding:
                from backend.grounding.rendering import grounding_audit_summary
                final["groundingAudit"] = grounding_audit_summary(grounding)
            run_status, review_status, task_status = "completed", "accepted", "succeeded"
        else:
            final = {"fusionSummary": "人工驳回本次研判：" + comment, "requiresHumanReview": True,
                     "humanReviewId": review_id, "actionPlan": []}
            run_status, review_status, task_status = "failed", "rejected", "skipped"
        decision = {"action": request.action, "reviewer": reviewer, "comment": comment,
                    "resolutions": resolutions, "finalDecision": final}
        updated = conn.execute("UPDATE collaboration_reviews SET status=?,version=version+1,request_digest=?,decision_json=?,"
                     "decided_at=? WHERE review_id=? AND status='pending' AND version=?",
                     (review_status, request_digest, _json(decision), now, review_id, request.expectedVersion))
        if updated.rowcount != 1:
            raise ReviewError("review_version_conflict", "裁决版本已变化，请刷新后重试")
        conn.execute("UPDATE collaboration_runs SET status=?,final_decision=?,updated_at=?,completed_at=? WHERE run_id=?",
                     (run_status, _json(final), now, now, run_id))
        conn.execute("UPDATE collaboration_tasks SET status=?,input_snapshot=?,output_snapshot=?,error_message='',"
                     "attempt=attempt+1,completed_at=? WHERE run_id=? AND task_id='task_fusion'",
                     (task_status, _json({"humanReview": decision, "sourceStamp": checkpoint["sourceStamp"]}),
                      _json(final), now, run_id))
        for resolution in resolutions:
            conn.execute("UPDATE collaboration_conflicts SET status='resolved',resolution=?,resolved_by=?,"
                         "requires_human_review=0,resolved_at=? WHERE run_id=? AND conflict_id=?",
                         (resolution["resolution"], reviewer, now, run_id, resolution["conflictId"]))
        events = [("human_review_decided", {"reviewId": review_id, "status": review_status})]
        if request.action == "resolve":
            events += [("fusion_start", {"executionMode": "human_review_template"}),
                       ("fusion_done", {"fusionSummary": final["fusionSummary"]})]
        events += [("run_" + run_status, {"status": run_status})]
        _audit(conn, run_id, review_id, events, now)
        db.advance_completed_event(conn, json.loads(run["normalized_event"]), run_status, now)
        conn.commit()
        return {"review": get_review(run_id), "duplicate": False}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
