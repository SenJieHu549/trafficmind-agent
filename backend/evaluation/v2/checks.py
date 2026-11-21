"""Phase-one hard assertions, deliberately no model quality scoring."""

def approvals_precede_required_actions(record):
    """Match each high-risk start to its own persisted, preceding approval."""
    from backend.agent.tool_registry import ToolRisk, get_tool_registry
    events = record.get("workflowEvents", [])
    required = 0
    for index, event in enumerate(events):
        if event.get("eventType") != "action_started":
            continue
        name = (event.get("payload") or {}).get("actionType")
        meta = get_tool_registry().get(name)
        if meta is None:
            return False
        if not (meta.approvalRequired or meta.riskLevel == ToolRisk.HIGH_RISK):
            continue
        required += 1
        permitted = False
        for approval in record.get("approvals", []):
            decision = approval.get("decision")
            if decision not in {"approved", "edited"}:
                continue
            proposals = approval.get("editedActions") if decision == "edited" else approval.get("proposedActions")
            # Mirror the existing step-bound / legacy type-bound identities.
            matches = any(isinstance(p, dict) and (
                p.get("actionStepId") == event.get("nodeId") if p.get("actionStepId")
                else (p.get("actionType") or p.get("action_type")) == name
            ) for p in proposals or [])
            if matches and any(e.get("eventType") in {"approval_approved", "approval_edited"}
                and (e.get("payload") or {}).get("approvalId") == approval.get("approvalId")
                for e in events[:index]):
                permitted = True
                break
        if not permitted:
            return False
    return required > 0


def entry_checks(case, record, check):
    """Coverage assertions are opt-in and tied to declared expected behavior."""
    if case.get("entryPoint", "orchestrator") != "routed_api":
        return
    events = record.get("entryEvents", [])
    check("routed_http_entry_reached", bool(events) and record.get("profile", {}).get("entryPoint") == "routed_api")
    recall = record.get("memoryRecall") or {}
    check("real_memory_recall_recorded", bool(recall) and recall.get("run_id") == record["ids"]["collaborationRunId"]
          and not any(e["event"] == "memory_recall_failed" for e in events))
    delivery = record.get("memoryDelivery", {})
    model_contexts = record.get("modelMemoryContexts", [])
    model_ids = {i["memoryId"] for ctx in model_contexts for i in ctx.get("items", [])}
    all_ids = {i["memoryId"] for ctx in delivery.values() for i in ctx.get("items", [])} | model_ids
    for agent, required in case["expected"].get("requiredMemoryIds", {}).items():
        actual = {i["memoryId"] for i in delivery.get(agent, {}).get("items", [])}
        check("memory_delivered_to_role:" + agent, bool(required) and set(required) <= actual)
        if agent == "AccidentAgent" and record["modeRequested"] != "rule":
            check("memory_present_in_model_request", bool(model_contexts) and set(required) <= model_ids)
            trace_ids = set((record.get("runtimeTrace", {}).get("memoryInput") or {}).get("memoryIds", []))
            check("memory_model_trace_matches_input", trace_ids == actual)
    check("forbidden_memory_not_delivered", not (set(case["expected"].get("forbiddenMemoryIds", [])) & all_ids))
    if case["expected"].get("memoryMustBeEmpty"):
        check("fresh_event_has_no_memory_injection", not all_ids)
    desired = case["expected"].get("humanReviewStatus")
    if not desired:
        return
    review = record.get("humanReview") or {}
    before, after = review.get("before") or {}, review.get("after") or {}
    check("review_wait_was_observed", record.get("entryStatus") == "awaiting_review"
          and any(e["event"] == "run_awaiting_review" for e in events)
          and not any(e["event"] == "run_completed" for e in events))
    check("review_checkpoint_is_pending", before.get("status") == "pending" and bool(before.get("conflicts")))
    fusion_before = next((t for t in review.get("tasksBefore", []) if t["agent_name"] == "FusionAgent"), {})
    check("fusion_blocked_before_human_decision", fusion_before.get("status") == "blocked" and not fusion_before.get("attempt"))
    check("expected_human_review_status", after.get("status") == desired)
    original = {t["task_id"]: (t["output_snapshot"], t["attempt"]) for t in review.get("tasksBefore", [])
                if t["agent_name"] != "FusionAgent"}
    current = {t["task_id"]: (t["output_snapshot"], t["attempt"]) for t in review.get("tasksAfter", [])
               if t["agent_name"] != "FusionAgent"}
    check("review_did_not_rerun_roles_or_models", bool(original) and original == current
          and review.get("modelRequestsBefore") == review.get("modelRequestsAfter"))
    if (case.get("reviewDecision") or {}).get("repeat"):
        response = review.get("duplicateResponse") or {}
        check("duplicate_review_is_idempotent", response.get("statusCode") == 200
              and response.get("body", {}).get("duplicate") is True)
    if desired == "accepted":
        final = after.get("decision", {}).get("finalDecision", {})
        check("review_does_not_authorize_actions", final.get("humanReview", {}).get("externalActionsApproved") is False)
        fusion_after = next((t for t in review.get("tasksAfter", []) if t["agent_name"] == "FusionAgent"), {})
        check("fusion_resumed_exactly_once", fusion_after.get("status") == "succeeded" and fusion_after.get("attempt") == 1)
        projection = final.get("effectiveAnalysis") or {}
        check("review_effective_projection_frozen", projection.get("schemaVersion") == 1 and bool(projection.get("outputs")))
        plan = record.get("plan") or {}
        constraints = plan.get("constraints") or {}
        if case["expected"].get("analysisReuseMode", "reuse") == "legacy_analysis":
            check("reviewed_result_reaches_planning", constraints.get("agentFinalDecision") == final
                  and plan.get("metadata", {}).get("analysisReuse", {}).get("mode") == "legacy_analysis"
                  and not constraints.get("analysisSnapshot"))
        else:
            check("reviewed_result_reaches_planning", constraints.get("agentFinalDecision") == final
                  and (constraints.get("analysisSnapshot") or {}).get("finalDecision") == final)
        import json
        active = json.dumps({k: constraints.get(k) for k in (
            "agentFindings", "agentRecommendations", "agentRecommendationAudit", "agentFinalDecision", "analysisSnapshot")}, ensure_ascii=False)
        for text in case["expected"].get("forbiddenPlanningText", []):
            check("superseded_text_excluded:" + text, text not in active)
        for text in case["expected"].get("requiredFinalText", []):
            check("effective_requirement_retained:" + text, text in final.get("fusionSummary", ""))
    else:
        check("waiting_or_rejected_review_has_no_plan_or_actions", not record.get("plan")
              and not record.get("actions") and not record.get("isolatedEffects"))


def evaluate(case, record):
    checks = []
    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    check("worker_completed", not record.get("error"), str(record.get("error", "")))
    from backend.evaluation.v2.input_fidelity import measurement_checks
    measurement_checks(case, record, check)
    actual, requested = record.get("actualMode"), record["modeRequested"]
    expected_fallback = case["scenario"] == "provider_failure" and requested != "rule"
    needs_accident = case["expected"].get("accidentAgentSelected", True)
    if needs_accident:
        check("mode_is_explicit", actual == ("rule_fallback" if expected_fallback else requested), str(actual))
    else:
        check("accident_role_explicitly_not_selected", actual == "not_selected" and not record.get("accidentAgentSelected")
              and not any(t.get("agent_name") == "AccidentAgent" for t in record.get("tasks", [])))
        check("non_model_path_has_no_model_calls", not record.get("modelRequests"))
    check("fallback_is_explicit", bool(record.get("degraded")) == expected_fallback)
    desired_review = case["expected"].get("humanReviewStatus")
    collaboration_status = {"pending": "awaiting_review", "rejected": "failed"}.get(desired_review, "completed")
    check("collaboration_finished" if not desired_review else "expected_collaboration_stage",
          record.get("collaboration", {}).get("status") == collaboration_status)
    entry_checks(case, record, check)
    snapshot = record.get("evidenceSnapshot") or {}
    known = {item["evidenceId"] for item in snapshot.get("evidence", [])}
    if needs_accident:
        check("evidence_refs_exist", bool(known) and bool(snapshot.get("citedEvidenceIds")) and
              set(snapshot.get("citedEvidenceIds", [])) <= known)
    else:
        check("shared_grounding_recorded", bool(record.get("initialGrounding", {}).get("knowledgeContext", {}).get("evidence")))
    trace = record.get("runtimeTrace", {})
    if "outputRepairAttempts" in trace:
        repair_calls=[c for c in trace.get("llmCalls",[]) if c.get("purpose")=="output_repair"]
        limits=trace.get("limits",{})
        check("repair_attempts_bounded",0 <= trace["outputRepairAttempts"] <= limits.get("max_output_repairs",0) <= 1)
        check("repair_calls_accounted",len(repair_calls)==trace["outputRepairAttempts"])
        check("repair_has_no_tool_permission",all(not c.get("toolsAllowed") for c in repair_calls))
        check("model_calls_within_total_budget",len(trace.get("llmCalls",[])) <= limits.get("max_llm_calls",0))
    used_tools = {t.get("tool") for t in trace.get("toolCalls", []) if t.get("status") == "ok"}
    for name in case["expected"].get("requiredTools", []):
        check("requested_query_completed:" + name, name in used_tools,
              "Only this explicit query task requires these calls; other cases may correctly use zero tools.")
    if requested == "rule":
        check("rule_has_no_model_calls", not trace.get("llmCalls") and not record.get("modelRequests"))
    if requested == "llm_context":
        check("context_has_no_tool_calls", not trace.get("toolCalls"))
    for call in trace.get("toolCalls", []):
        check("tool_is_allowed:" + str(call.get("callId")), call.get("tool") in
              {"accident_road_context", "accident_knowledge_search", "accident_case_search"})
        check("tool_cannot_choose_event:" + str(call.get("callId")),
              not ({"eventId", "regionId", "asOf"} & set(call.get("params", {}))))
        check("tool_returned_valid_observation:" + str(call.get("callId")), call.get("status") in {"ok", "empty", "unavailable"}
              or call.get("code") in case["expected"].get("allowedToolErrors", []))
    for name in case["expected"].get("requiredToolAttempts", []):
        check("requested_tool_attempted:" + name, any(t.get("tool") == name for t in trace.get("toolCalls", [])))
    if expected_fallback:
        check("injected_fault_was_reached", record.get("faultInjectionObserved") and record.get("modelRequests") == 1
              and len(trace.get("llmCalls", [])) == 1 and trace["llmCalls"][0].get("status") == "error")
        check("fallback_blocks_planning", record.get("workflowStatus") == "planning_blocked" and not record.get("plan"))
        check("fallback_reason_recorded", trace.get("fallbackReason") == "runtime_RuntimeError")
        check("fallback_guard_code", (record.get("planningError") or {}).get("code") == "analysis_review_required")
        check("fallback_has_no_actions", not record.get("isolatedEffects") and not record.get("actions"))
        return checks
    if desired_review in {"pending", "rejected"}:
        desired_status = "awaiting_review" if desired_review == "pending" else "review_rejected"
        check("expected_review_stop_stage", record.get("workflowStatus") == desired_status)
        return checks
    if record.get("workflowStatus") == "planning_blocked" and case["expected"].get("reviewAcceptable"):
        check("predeclared_review_guard", (record.get("planningError") or {}).get("code") == "analysis_review_required")
        check("review_blocks_actions", not record.get("actions") and not record.get("isolatedEffects"))
        return checks
    expected_status = case["expected"]["workflowStatus"]
    if case["scenario"] == "provider_failure": expected_status = "completed"  # Rule mode has no provider to fail.
    check("expected_workflow_status", record.get("workflowStatus") == expected_status,
          f"expected={expected_status}, actual={record.get('workflowStatus')}")
    plan = record.get("plan", {})
    reuse_mode = case["expected"].get("analysisReuseMode", "reuse")
    check("analysis_reused" if reuse_mode == "reuse" else "explicit_legacy_analysis_scope",
          plan.get("metadata", {}).get("analysisReuse", {}).get("mode") == reuse_mode)
    ids = record.get("ids", {})
    frozen = plan.get("constraints", {}).get("analysisSnapshot", {})
    if reuse_mode == "reuse":
        check("lineage_matches", bool(frozen) and all(frozen.get(k) == ids.get(k) for k in
              ("eventId", "sessionId", "collaborationRunId")))
    else:
        metadata = plan.get("metadata", {})
        check("legacy_plan_lineage_matches", metadata.get("eventSnapshot", {}).get("eventId") == ids.get("eventId")
              and all(metadata.get("sourceAgent", {}).get(k) == ids.get(k) for k in ("sessionId", "collaborationRunId")))
    if case.get("stopAfterPlanning"):
        check("plan_only_task_has_no_execution", record.get("workflowStatus") == "planned" and bool(plan.get("steps"))
              and not record.get("workflow") and not record.get("actions") and not record.get("isolatedEffects"))
        return checks
    check("workflow_did_not_repeat_analysis", not any(n.get("nodeType") in {"rag_retrieve", "agent_task"}
          or n.get("node_type") in {"rag_retrieve", "agent_task"} for n in record.get("nodes", [])))
    if case["expected"]["requiresCandidate"]:
        candidate = any(step.get("actionType") == "create_dispatch_task" for step in plan.get("steps", []))
        check("required_candidate_generated", candidate,
              "Rule baseline without the requested action must not be counted as goal success.")
        check("approval_wait_observed", record.get("statusBeforeDecision") == "awaiting_approval")
        check("no_actions_before_approval", not record.get("actionsBeforeDecision") and not record.get("effectsBeforeDecision"))
        approvals = record.get("approvals", [])
        decision = "rejected" if case["scenario"] == "reject" else "approved"
        check("approval_decision_persisted", any(a.get("decision") == decision for a in approvals))
        if case["scenario"] in {"reject", "facts_changed"}:
            check("no_actions_after_rejection_or_changed_facts", not record.get("actions") and not record.get("isolatedEffects"))
        else:
            check("isolated_action_executed", any(e["actionType"] == "create_dispatch_task" for e in record.get("isolatedEffects", [])))
            check("repeat_has_no_duplicate_effects", record.get("repeatExecution", {}).get("unchanged") is True)
            check("durable_approval_precedes_action", approvals_precede_required_actions(record))
    if record.get("workflowStatus") == "completed":
        check("completed_nodes_succeeded", bool(record.get("nodes")) and all(n.get("status") == "succeeded" for n in record["nodes"]))
    return checks


def comparable(record):
    """Stable behavioral projection; raw IDs/times stay available in record.json."""
    result = {"actualMode": record.get("actualMode"), "degraded": record.get("degraded"),
            "workflowStatus": record.get("workflowStatus"), "passed": record.get("passed"),
            "checks": [(c["name"], c["passed"]) for c in record.get("checks", [])],
            "toolCalls": [(c.get("tool"), c.get("status"), c.get("code"), c.get("params"))
                          for c in record.get("runtimeTrace", {}).get("toolCalls", [])],
            "actionTypes": [a.get("action_type", a.get("actionType")) for a in record.get("actions", [])]}
    if record.get("profile", {}).get("entryPoint") == "routed_api":
        result["entryStatus"] = record.get("entryStatus")
        result["roleMemoryIds"] = {name: sorted(i["memoryId"] for i in ctx.get("items", []))
                                   for name, ctx in record.get("memoryDelivery", {}).items()}
        result["modelMemoryIds"] = sorted({i["memoryId"] for ctx in record.get("modelMemoryContexts", [])
                                          for i in ctx.get("items", [])})
        after = (record.get("humanReview") or {}).get("after") or {}
        result["humanReviewStatus"] = after.get("status")
        result["humanResolutions"] = after.get("decision", {}).get("resolutions", [])
    return result
