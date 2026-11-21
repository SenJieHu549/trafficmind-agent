"""Opt-in business acceptance baseline: pytest this file explicitly.

Failures are reported as failures, not xfail or production fixes. Scripted role
outputs isolate business invariants while Orchestrator/SQLite/API/Handoff and
the actual Planning adapter execute normally. No models or actions are run.
"""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.tests.test_analysis_handoff import env
from backend.agent.collaboration import human_review, orchestrator
from backend.agent.collaboration.executor import AgentExecutionResult
from backend.agent.collaboration.protocol import AgentResult
from backend.planning.analysis_handoff import build_analysis_snapshot, AnalysisHandoffError
from backend.planning.agent_planning_adapter import build_planning_input_from_agent
from backend.rag.v2.models import RagDocument, RagChunk


SIGNAL = "延长机动车绿灯20秒"
SAFETY = "保持行人过街相位，不延长机动车绿灯"
ACCESS = "保障急救通道，不得占用急救通道"
UNKNOWN = "伤亡情况未知，须先核实人员伤情"
EDIT = "保持行人过街相位，暂缓机动车延长方案，改为评估其他分流路线"
POLICY = ACCESS + "。信号调整应保留行人过街相位。"


def active_summaries(snapshot):
    """Exclude audit/historical proposals when inspecting current recommendations."""
    result = []
    for value in snapshot.get("agentOutputs", {}).values():
        summary = value.get("summary", "")
        if isinstance(summary, str) and summary.startswith("{"):
            result.append(json.loads(summary).get("fusionSummary", ""))
        else:
            result.append(summary)
    return result


@pytest.fixture
def quality_case(env, monkeypatch, request, tmp_path):
    from backend import config
    from backend.app import app
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    monkeypatch.setattr(orchestrator, "LLM_ENABLED", False)
    env["docs"].upsert_document(RagDocument(document_id="doc", source_id="test:frozen-policy",
        title="验证夹具：通行硬约束", content=POLICY, grounding_scope="GLOBAL", event_type="accident"))
    env["docs"].upsert_chunks([RagChunk(chunk_id="chunk", document_id="doc", raw_content=POLICY,
                                     contextual_content=POLICY)])
    source = env["collaboration"].get_run("source")
    grounding = json.loads(source["grounding_context"])
    grounding["knowledgeContext"]["evidence"][0]["excerpt"] = POLICY
    call_log = []
    outputs = {
        "SignalAgent": {"findings": ["信号配时拟增加机动车绿灯"], "suggestion": SIGNAL},
        "PublicSafetyAgent": {"findings": ["行人过街安全须保障"], "suggestion": SAFETY},
        "AccidentAgent": {"findings": [UNKNOWN, ACCESS], "suggestion": ACCESS + "；" + UNKNOWN},
        "DispatchAgent": {"findings": ["等待协调后的方案"], "suggestion": "依据最终裁决安排后续协调"},
    }
    for name, overrides in getattr(request, "param", {}).items():
        outputs[name].update(overrides)
    async def scripted_role(task, state, budget):
        call_log.append(task.agent_name)
        result = AgentResult(agent_name=task.agent_name, task_id=task.task_id, status="completed",
            confidence=.8, urgency="high", evidence_refs=["chunk"], **outputs[task.agent_name])
        return AgentExecutionResult(True, task.agent_name, task.task_id, result)
    monkeypatch.setattr(orchestrator, "execute_single_agent", scripted_role)
    event = deepcopy(env["event"])
    event["fieldSources"] = {"eventId": "event_records"}
    before = deepcopy(event)
    async def execute():
        return [frame async for frame in orchestrator.CollaborationOrchestrator().execute(
            "quality-run", "session", event, ["SignalAgent", "PublicSafetyAgent", "AccidentAgent"],
            grounding_context=grounding)]
    frames = asyncio.run(execute())
    assert event == before
    review = human_review.get_review("quality-run")
    assert review and review["status"] == "pending" and len(review["conflicts"]) == 3
    assert all("AccidentAgent" not in c["agents"] for c in review["conflicts"])
    case_id = request.node.name.split('_')[1].upper()
    record = {"caseId": case_id, "evidenceMode": "scripted_roles_real_runtime_and_adapter",
              "input": {"event": before, "policy": {"documentId": "doc", "chunkId": "chunk", "text": POLICY},
                        "roleOutputs": outputs, "review": review},
              "before": {"status": env["collaboration"].get_run("quality-run")["status"],
                         "frames": frames}, "checks": []}
    client = TestClient(app)
    case = {**env, "record": record, "review": review, "calls": call_log}

    def resolve(text=None):
        body = {"expectedVersion": 1, "action": "resolve", "reviewer": "validation-operator",
                "comment": "冻结业务验证输入，不执行外部动作",
                "resolutions": [{"conflictId": c["id"], **(
                    {"choice": "select", "selectedAgent": "PublicSafetyAgent"} if text is None else
                    {"choice": "edit", "resolution": text})} for c in review["conflicts"]]}
        response = client.post(f"/collaboration/runs/quality-run/reviews/{review['reviewId']}", json=body)
        record["request"] = body
        record["response"] = {"statusCode": response.status_code, "body": response.json()}
        run = env["collaboration"].get_run("quality-run")
        record["after"] = {"status": run["status"], "event": json.loads(run["normalized_event"]),
                           "finalDecision": json.loads(run["final_decision"]),
                           "roleCallLog": list(call_log), "review": human_review.get_review("quality-run")}
        try:
            case["snapshot"] = build_analysis_snapshot("handoff-event", "session", "quality-run")
            record["analysisSnapshot"] = case["snapshot"]
        except AnalysisHandoffError as exc:
            record["handoffError"] = {"code": exc.code, "message": str(exc)}
        return response

    def planning_input():
        value = build_planning_input_from_agent("handoff-event", "session", "quality-run", reuse_analysis=True)
        record["planningRequest"] = value.to_request_dict()
        return value

    case.update(resolve=resolve, planning_input=planning_input)
    yield case
    out = Path(os.environ.get("HUMAN_REVIEW_VALIDATION_DIR", str(tmp_path)))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case_id}_observed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def verify(case, checks):
    case["record"]["checks"] = [{"id": key, "passed": bool(value), "description": explanation}
                                for key, value, explanation in checks]
    failures = [c["id"] + ": " + c["description"] for c in case["record"]["checks"] if not c["passed"]]
    assert not failures, " | ".join(failures)


def test_v01_selected_proposal_is_effective_and_superseded_one_is_audit_only(quality_case):
    case = quality_case
    response = case["resolve"]()
    final = case["record"]["after"]["finalDecision"]
    chosen = final["humanReview"]["resolutions"]
    frozen = {c["id"]: next(p for p in c["proposals"] if p["agentName"] == "PublicSafetyAgent")
              for c in case["review"]["conflicts"]}
    verify(case, [
        ("selection_bound_to_source", all(r["resolution"] == frozen[r["conflictId"]]["suggestion"]
            and r["selectedAgent"] == "PublicSafetyAgent" for r in chosen), "裁决逐项对应冻结角色方案"),
        ("current_fusion", final["fusionSummary"].count(SAFETY) == len(chosen), "选定方案进入融合结果"),
        ("no_old_current_snapshot", all(SIGNAL not in s for s in active_summaries(case["snapshot"])), "旧方案不进入当前快照推荐"),
        ("audit_preserved", any(p["suggestion"] == SIGNAL for c in case["review"]["conflicts"] for p in c["proposals"]), "旧方案仍可审计"),
        ("api_success", response.status_code == 200, "选择原方案请求成功"),
    ])


def test_v02_edited_resolution_reaches_actual_planning_input(quality_case):
    case = quality_case
    case["resolve"](EDIT)
    planning = case["planning_input"]()
    final = planning.constraints["agentFinalDecision"]
    snapshot = planning.constraints["analysisSnapshot"]
    verify(case, [
        ("edit_exact", all(r["choice"] == "edit" and r["resolution"] == EDIT for r in final["humanReview"]["resolutions"]), "人工修改内容完整保留"),
        ("frozen_final_equal", snapshot["finalDecision"] == final, "Planning 两个裁决来源一致"),
        ("edit_in_snapshot", all(EDIT in s for s in active_summaries(snapshot)), "修改进入 Planning 复用快照"),
        ("no_role_rerun", len(case["calls"]) == 4, "只完成融合，原角色未重跑"),
    ])


def test_v03_unaffected_emergency_access_requirement_survives_fusion(quality_case):
    case = quality_case
    case["resolve"](EDIT)
    final = case["record"]["after"]["finalDecision"]
    snapshot = case["snapshot"]
    accident = next(t for t in case["collaboration"].list_tasks("quality-run") if t["agent_name"] == "AccidentAgent")
    verify(case, [
        ("source_exists", ACCESS in json.loads(accident["output_snapshot"])["suggestion"], "未冲突角色的原始要求真实存在"),
        ("fusion_retains_access", ACCESS in final["fusionSummary"], "融合结果保留未冲突的急救通道要求"),
        ("snapshot_retains_access", any(ACCESS in s for s in active_summaries(snapshot)), "下游快照保留此有效要求"),
    ])


def test_v04_unknown_injury_status_remains_explicit_in_current_result(quality_case):
    case = quality_case
    case["resolve"](EDIT)
    final = case["record"]["after"]["finalDecision"]
    verify(case, [
        ("unknown_in_source", UNKNOWN in case["record"]["input"]["roleOutputs"]["AccidentAgent"]["findings"], "原始未确认项存在"),
        ("unknown_retained", UNKNOWN in final["fusionSummary"], "裁决后仍明确保留伤情未知及核实要求"),
        ("unknown_in_handoff", any(UNKNOWN in s for s in active_summaries(case["snapshot"])), "未知项进入实际复用内容"),
    ])


def test_v05_policy_violating_edit_cannot_become_reusable_completed_analysis(quality_case):
    case = quality_case
    case["resolve"]("临时占用急救通道用于车辆排队")
    after = case["record"]["after"]
    verify(case, [
        ("explicit_policy", "不得占用急救通道" in case["record"]["input"]["policy"]["text"], "禁止事项来自冻结知识夹具"),
        ("not_completed", after["status"] != "completed", "违反明确约束时不能直接完成"),
        ("handoff_blocked", "snapshot" not in case, "违反约束的裁决不能形成可复用快照"),
    ])


def test_v06_unverified_human_fact_claim_requires_authoritative_confirmation(quality_case):
    case = quality_case
    case["resolve"]("现场已确认无人伤亡，无需再核实伤情，按无人伤亡方案处置")
    after = case["record"]["after"]
    verify(case, [
        ("event_not_overwritten", after["event"] == case["record"]["input"]["event"], "人工文字不覆盖当前事件"),
        ("unverified_claim_gated", after["status"] != "completed", "与既有未知项冲突的新事实须先核实，不能直接完成"),
        ("unverified_handoff_blocked", "snapshot" not in case, "未经核实的新事实不能随裁决直接成为可复用结论"),
    ])


def test_v07_evidence_identity_is_valid_and_changed_source_blocks_handoff(quality_case):
    case = quality_case
    case["resolve"]()
    snapshot = case["snapshot"]
    selected = case["record"]["after"]["finalDecision"]["humanReview"]["resolutions"]
    proposals = {c["id"]: {p["agentName"]: p for p in c["proposals"]} for c in case["review"]["conflicts"]}
    refs = set(snapshot["agentOutputs"]["SourceFusion"]["evidenceRefs"])
    identities = {(e["documentId"], e["chunkId"]) for e in snapshot["evidence"]}
    case["docs"].upsert_document(RagDocument(document_id="doc", source_id="test:frozen-policy",
        title="Changed source", content=POLICY, grounding_scope="GLOBAL", event_type="accident", status="expired"))
    blocked = False
    try:
        build_analysis_snapshot("handoff-event", "session", "quality-run")
    except AnalysisHandoffError as exc:
        blocked = True
        case["record"]["changedEvidenceError"] = {"code": exc.code, "message": str(exc)}
    verify(case, [
        ("selected_source_refs", all(set(proposals[r["conflictId"]][r["selectedAgent"]]["evidenceRefs"]) <= refs for r in selected), "选择方案来源引用在快照中可解析"),
        ("canonical_chunk", ("doc", "chunk") in identities, "证据对应真实文档和分块"),
        ("changed_source_blocked", blocked, "知识失效后拒绝继续复用旧证据"),
    ])


def test_v08_all_planning_channels_obey_decision_and_retain_valid_constraints(quality_case):
    case = quality_case
    case["resolve"](EDIT)
    planning = case["planning_input"]()
    constraints = planning.constraints
    snapshot = constraints["analysisSnapshot"]
    recommendations = constraints["agentRecommendations"]
    verify(case, [
        ("decision_present", EDIT in constraints["agentFinalDecision"]["fusionSummary"], "裁决进入实际 Planning 请求"),
        ("no_superseded_parallel_recommendation", not any(r.get("suggestion") == SIGNAL for r in recommendations), "并行 agentRecommendations 不能重新带入被替代方案"),
        ("all_constraints_in_reuse_snapshot", any(ACCESS in s for s in active_summaries(snapshot))
            and any(EDIT in s for s in active_summaries(snapshot)), "实际复用快照同时保留人工决定与未受影响要求"),
        ("no_action_approval", constraints["agentFinalDecision"]["humanReview"]["externalActionsApproved"] is False, "裁决不产生外部动作授权"),
    ])
