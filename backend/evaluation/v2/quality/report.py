"""Quality, coverage, task assertions and safety stay separate."""
from collections import Counter, defaultdict
from statistics import mean

from backend.evaluation.v2.quality.rubric import DIMENSIONS


def aggregate(results):
    groups = defaultdict(list)
    cohorts = defaultdict(dict)
    for item in results:
        groups[(item["split"], item["mode"], item["generationTransport"], item["judgeTransport"])].append(item)
        cohorts[(item["split"], item.get("requestedMode", item["mode"]), item["generationTransport"],
                 item["judgeTransport"])][item["recordHash"]] = item
    summaries = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        scored = [r for r in items if r["assessment"].get("status") == "scored" and
                  not r["assessment"]["verdict"]["unableToJudge"]]
        primary = [r for r in scored if r["primaryEligible"]]
        # One execution can be graded repeatedly. It contributes ONCE to task
        # rates; repeated ratings are retained only for stability diagnostics.
        executions = {r["recordHash"]: r for r in items}
        per_case = defaultdict(list)
        for r in primary: per_case[r["recordHash"]].append(r)
        dimensions = {}
        for d in DIMENSIONS:
            values = [mean(x["assessment"]["verdict"][d]["score"] for x in ratings) for ratings in per_case.values()]
            dimensions[d] = {"mean": round(mean(values), 4) if values else None,
                "distributionPerRating": dict(Counter(str(r["assessment"]["verdict"][d]["score"]) for r in primary)),
                "scoredExecutions": len(values)}
        case_averages = [mean(mean(r["assessment"]["verdict"][d]["score"] for d in DIMENSIONS)
                              for r in ratings) for ratings in per_case.values()]
        stability = []
        for ratings in per_case.values():
            if len(ratings) < 2: continue
            for d in DIMENSIONS:
                values = [r["assessment"]["verdict"][d]["score"] for r in ratings]
                stability.append({"inputHash": ratings[0]["assessment"]["inputHash"], "dimension": d,
                                  "range": max(values) - min(values), "ratings": len(values)})
        normal = [r for r in executions.values() if not r.get("isException", r["scenario"] in {"reject", "facts_changed", "provider_failure"})]
        exceptional = [r for r in executions.values() if r not in normal]
        rate = lambda rows: round(sum(r["hardPassed"] for r in rows) / len(rows), 4) if rows else None
        workflow_checks = [c for r in executions.values() for c in r["hardChecks"] if c["name"] in {
            "analysis_reused", "lineage_matches", "workflow_did_not_repeat_analysis", "approval_wait_observed",
            "no_actions_before_approval", "approval_decision_persisted", "no_actions_after_rejection_or_changed_facts",
            "durable_approval_precedes_action", "repeat_has_no_duplicate_effects", "review_blocks_actions",
            "completed_nodes_succeeded", "fallback_blocks_planning", "fallback_guard_code", "predeclared_review_guard"}]
        tool_attempts = sum(r.get("toolAttempts", 0) for r in executions.values())
        summaries.append({"split": key[0], "mode": key[1], "generationTransport": key[2], "judgeTransport": key[3],
            "executionCount": len(executions), "ratingCount": len(items), "primaryScoredExecutionCount": len(per_case),
            "primaryQualityCoverage": round(len(per_case) / len(executions), 4),
            "dimensions": dimensions, "equalWeightQualityMean": round(mean(case_averages), 4) if case_averages else None,
            "hardAssertionPassRate": rate(list(executions.values())), "normalTaskPassRate": rate(normal),
            "normalTaskCount": len(normal), "exceptionTaskCount": len(exceptional),
            "exceptionExpectedBehaviorPassRate": rate(exceptional),
            "workflowConstraintCheckPassRate": round(sum(c["passed"] for c in workflow_checks)/len(workflow_checks), 4) if workflow_checks else None,
            "legalToolRequestRate": round(sum(r.get("legalToolRequests", 0) for r in executions.values())/tool_attempts, 4) if tool_attempts else None,
            "toolAttempts": tool_attempts,
            "normalWorkflowCompletionRate": round(sum(r.get("workflowStatus") == "completed" for r in normal)/len(normal), 4) if normal else None,
            "fallbackExecutions": sum(r["degraded"] for r in executions.values()),
            "judgeErrorCount": sum(r["assessment"]["status"] == "error" for r in items),
            "unableCount": sum(r["assessment"]["status"] == "unavailable" or
                 (r["assessment"].get("verdict") or {}).get("unableToJudge", False) for r in items),
            "stability": stability, "humanCalibration": "pending_independent_human_labels"})
    cohort_summary = []
    for key, unique in cohorts.items():
        rows = list(unique.values())
        normal = [r for r in rows if not r.get("isException", r["scenario"] in {"reject", "facts_changed", "provider_failure"})]
        exceptional = [r for r in rows if r not in normal]
        passed_rate = lambda selected: round(sum(r["hardPassed"] for r in selected)/len(selected), 4) if selected else None
        cohort_summary.append({"split": key[0], "requestedMode": key[1], "generationTransport": key[2], "judgeTransport": key[3],
            "totalExecutions": len(rows), "hardPassedExecutions": sum(r["hardPassed"] for r in rows),
            "predeclaredBehaviorPassRate": passed_rate(rows), "fallbackCount": sum(r["degraded"] for r in rows),
            "normalTaskCount": len(normal), "normalTaskPassRate": passed_rate(normal),
            "exceptionTaskCount": len(exceptional), "exceptionExpectedBehaviorPassRate": passed_rate(exceptional)})
    return {"cohorts": cohort_summary, "groups": summaries, "failureCases": [{"assessmentId": r["assessmentId"], "caseId": r["caseId"],
        "fallbackReason": r.get("fallbackReason"), "workflowStatus": r.get("workflowStatus"),
        "hardFailures": [c["name"] for c in r["hardChecks"] if not c["passed"]],
        "judgeStatus": r["assessment"]["status"], "issues": {d: (r["assessment"].get("verdict") or {}).get(d, {}).get("issues", [])
                                                              for d in DIMENSIONS}}
        for r in results if not r["hardPassed"] or r["assessment"]["status"] != "scored" or
        any((r["assessment"].get("verdict") or {}).get(d, {}).get("score", 5) is not None and
            (r["assessment"].get("verdict") or {}).get(d, {}).get("score", 5) <= 2 for d in DIMENSIONS)]}
