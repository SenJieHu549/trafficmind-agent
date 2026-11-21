"""Blind, bounded Judge inputs and deterministic audits of frozen records."""
from copy import deepcopy
import json

from backend.evaluation.v2.providers import content_hash, safe_value


def candidate(record):
    result = record.get("agentResult")
    if result is None:
        for task in record.get("tasks", []):
            if task.get("agent_name") == "AccidentAgent":
                raw = task.get("output_snapshot") or {}
                result = json.loads(raw) if isinstance(raw, str) else raw
                break
    result = result or {}
    data = {k: deepcopy(result[k]) for k in ("findings", "suggestion", "urgency", "assumptions",
                                           "proposed_actions", "evidence_refs") if k in result}
    # Remove runtime-added implementation labels, not substantive uncertainty.
    data["assumptions"] = [a for a in data.get("assumptions", [])
                           if not str(a).startswith("规则模式；置信度沿用适配器缺省值")]
    return data


def build_payload(record, case):
    output = candidate(record)
    claims = []
    for key in ("findings", "suggestion", "urgency", "assumptions", "proposed_actions"):
        values = output.get(key, [])
        if not isinstance(values, list): values = [values]
        for i, value in enumerate(values):
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            if text: claims.append({"claimId": f"{key}[{i}]", "text": text})
    snapshot = record.get("evidenceSnapshot") or {}
    event = record.get("event") or {}
    facts = {k: v for k, v in event.items() if k not in
             {"eventId", "originalInput", "fieldSources", "capturedAt", "snapshotSource", "runKind", "actionExecutionAllowed"}}
    evidence = [{"evidenceId": e["evidenceId"], "kind": e["kind"], "asOf": e.get("asOf"),
                 "data": deepcopy(e["data"])} for e in snapshot.get("evidence", [])]
    # Opaque source evidence IDs must remain intact for citation checking. The
    # serving/version identities and model/provider metadata never enter Judge.
    hidden = sorted({str(v) for v in (record.get("ids") or {}).values() if v} | {record["caseId"]},
                    key=lambda value: (-len(value), value))
    def blind(value):
        if isinstance(value, dict): return {k: blind(v) for k, v in value.items()}
        if isinstance(value, list): return [blind(v) for v in value]
        if isinstance(value, str):
            for index, identifier in enumerate(hidden): value = value.replace(identifier, f"ANON_{index}")
        return value
    memory = {}
    for context in record.get("modelMemoryContexts", []):
        if context.get("items"):
            memory[content_hash(context)] = deepcopy(context)
    extra = {"referenceMemory": list(memory.values())} if memory else {}
    payload = safe_value(blind({"task": case.get("question", event.get("originalInput", "")),
        "facts": facts, "claims": claims, "candidate": output, "evidence": evidence,
        "referenceFacts": case.get("annotations", {}).get("knownFacts", []),
        "unknownFacts": case.get("annotations", {}).get("unknownFacts", []),
        "availableCapabilities": ["只读道路/知识/案例查询", "create_dispatch_task 候选内部核实任务",
                                  "send_notification 候选通知", "update_event_status 候选状态变更"],
        "executionContract": "这里只评价候选研判与建议；审批和执行由后端处理，不能声称现实交通问题已解决。",
        **extra}))
    if len(json.dumps(payload, ensure_ascii=False)) > 65000: raise ValueError("judge_input_too_large")
    return payload


def hard_audit(record, case):
    from backend.agent.llm_accident.contracts import digest
    from backend.agent.llm_accident.tools import TOOL_MODELS
    result = deepcopy(record.get("checks", []))
    def check(name, passed): result.append({"name": name, "passed": bool(passed)})
    check("execution_assertions_present", bool(record.get("checks")))
    snapshot = record.get("evidenceSnapshot") or {}
    body = {k: v for k, v in snapshot.items() if k != "snapshotDigest"}
    check("frozen_evidence_digest", bool(snapshot) and snapshot.get("snapshotDigest") == digest(body))
    evidence = snapshot.get("evidence", [])
    known = {e.get("evidenceId") for e in evidence}
    refs = candidate(record).get("evidence_refs", [])
    check("candidate_references_exist", set(refs) <= known)
    for e in evidence:
        raw = {k: e.get(k) for k in ("kind", "sourceId", "eventId", "asOf", "data")}
        check("evidence_content_hash:" + str(e.get("evidenceId")),
              e.get("contentHash") == digest(raw) and e.get("evidenceId") == "ev_" + digest(raw)[:24])
    for i, call in enumerate(record.get("runtimeTrace", {}).get("toolCalls", [])):
        name = call.get("tool")
        valid = name in TOOL_MODELS and "params" in call
        if valid:
            try: TOOL_MODELS[name].model_validate(call.get("params", {}))
            except ValueError: valid = False
        check(f"tool_schema:{i}", valid)
    actual_cases = {e.get("sourceId") for e in evidence if e.get("kind") == "case"}
    for cid in case.get("expected", {}).get("requiredCaseIds", []): check("case_available:" + cid, cid in actual_cases)
    for cid in case.get("expected", {}).get("forbiddenCaseIds", []): check("future_case_excluded:" + cid, cid not in actual_cases)
    return result
