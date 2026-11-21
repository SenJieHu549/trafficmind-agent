"""Frozen-cohort comparisons; no filtering failures out of task denominators."""
import argparse
import json
from pathlib import Path
from statistics import mean


def read_arm(root, arm, spec):
    generation = Path(spec["directory"])
    summary = json.loads((generation / "summary.json").read_text(encoding="utf-8"))
    quality = json.loads((Path(spec["qualityDirectory"]) / "quality_report.json").read_text(encoding="utf-8"))
    ratings = {(r["caseId"], r.get("requestedMode", r["mode"])): r for r in quality["results"]}
    records = {}
    for row in summary["results"]:
        record = json.loads((generation / row["recordPath"]).read_text(encoding="utf-8"))
        mode, cid = row["mode"], row["caseId"]
        rating = ratings.get((cid, mode))
        verdict = (rating or {}).get("assessment", {}).get("verdict")
        scored = bool(rating and rating.get("primaryEligible") and verdict and not verdict["unableToJudge"])
        records[(cid, mode)] = {"caseId": cid, "requestedMode": mode, "actualMode": record.get("actualMode"),
            "taskPassed": bool(rating and rating["hardPassed"]), "degraded": record.get("degraded", False),
            "failureReason": record.get("runtimeTrace", {}).get("fallbackReason") or record.get("runtimeTrace", {}).get("failureReason"),
            "workflowStatus": record.get("workflowStatus"), "judgeStatus": (rating or {}).get("assessment", {}).get("status", "missing"),
            "quality": mean(verdict[d]["score"] for d in ("reasoning", "evidence", "feasibility")) if scored else None,
            "dimensions": {d: verdict[d]["score"] for d in ("reasoning", "evidence", "feasibility")} if scored else None,
            "toolCalls": len(record.get("runtimeTrace", {}).get("toolCalls", [])),
            "llmCalls": len(record.get("runtimeTrace", {}).get("llmCalls", [])),
            "recordPath": str(generation / row["recordPath"])}
    return records


def metrics(rows):
    scored = [r["quality"] for r in rows if r["quality"] is not None]
    return {"n": len(rows), "taskPassed": sum(r["taskPassed"] for r in rows),
        "taskPassRate": round(sum(r["taskPassed"] for r in rows)/len(rows), 4) if rows else None,
        "fallbackCount": sum(r["degraded"] for r in rows),
        "invalidReferenceCount": sum(r["failureReason"] == "unknown_evidence_reference" for r in rows),
        "qualityMeanConditional": round(mean(scored), 4) if scored else None, "qualityScoredN": len(scored),
        "qualityCoverage": round(len(scored)/len(rows), 4) if rows else None,
        "judgeErrorCount": sum(r["judgeStatus"] in {"missing", "error", "unavailable"} for r in rows),
        "toolsTotal": sum(r["toolCalls"] for r in rows), "llmRequestsTotal": sum(r["llmCalls"] for r in rows)}


def compare(left, right, ids):
    if set(left) != set(ids) or set(right) != set(ids): raise ValueError("unpaired_task_denominators")
    paired = [cid for cid in ids if left[cid]["quality"] is not None and right[cid]["quality"] is not None]
    deltas = [right[cid]["quality"] - left[cid]["quality"] for cid in paired]
    return {"n": len(ids), "before": metrics([left[cid] for cid in ids]), "after": metrics([right[cid] for cid in ids]),
        "taskPassDeltaPercentagePoints": round(100 * sum(int(right[cid]["taskPassed"])-int(left[cid]["taskPassed"]) for cid in ids)/len(ids), 4),
        "pairedQualityN": len(paired), "pairedQualityCaseIds": paired,
        "pairedQualityMeanDelta": round(mean(deltas), 4) if deltas else None,
        "qualityImprovedCases": sum(d > 1e-9 for d in deltas), "qualityTiedCases": sum(abs(d) <= 1e-9 for d in deltas),
        "qualityWorseCases": sum(d < -1e-9 for d in deltas),
        "taskRegressions": [cid for cid in ids if left[cid]["taskPassed"] and not right[cid]["taskPassed"]],
        "taskRecoveries": [cid for cid in ids if not left[cid]["taskPassed"] and right[cid]["taskPassed"]]}


def build_report(root):
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    runs = {arm: read_arm(root, arm, spec) for arm, spec in manifest["runs"].items() if spec.get("qualityDirectory")}
    report = {"protocolHash": manifest["protocolHash"], "humanCalibration": "deferred_not_claimed", "arms": {}, "comparisons": {}}
    parity = {}
    parity_conflicts = []
    for arm, records in runs.items():
        split = arm.rsplit("_",1)[1]
        for (cid, mode), result in records.items():
            record = json.loads(Path(result["recordPath"]).read_text(encoding="utf-8"))
            event = {k:v for k,v in record.get("event",{}).items() if k != "capturedAt"}
            initial = [{k:e.get(k) for k in ("kind","sourceId","eventId","asOf","data")}
                for e in (record.get("evidenceSnapshot") or {}).get("evidence",[])
                if set(e.get("origins",[])) & {"initial_event","grounding"}]
            facts = {"event":event,"initialEvidence":initial}
            key=(split,cid)
            if key not in parity: parity[key]=facts
            elif parity[key] != facts: parity_conflicts.append({"arm":arm,"caseId":cid,"mode":mode})
    report["inputParity"]={"matched":not parity_conflicts,"conflicts":parity_conflicts,
                          "scope":"event facts and initial durable evidence; excludes model output and later tool evidence"}
    for arm, records in runs.items():
        split = arm.rsplit("_", 1)[1]
        cohorts = manifest["datasets"][split]
        report["arms"][arm] = {}
        for mode in sorted({m for _, m in records}):
            report["arms"][arm][mode] = {name: metrics([row for (cid, m), row in records.items() if m == mode and cid in ids])
                for name, ids in (("all", cohorts["caseIds"]), ("common", cohorts["commonCases"]), ("toolSpecific", cohorts["toolSpecificCases"]))}
    def comparison(name, arm1, mode1, arm2, mode2, split):
        if arm1 not in runs or arm2 not in runs: return
        left = {cid: r for (cid,m),r in runs[arm1].items() if m == mode1}
        right = {cid: r for (cid,m),r in runs[arm2].items() if m == mode2}
        cohorts = manifest["datasets"][split]
        report["comparisons"][name] = {cohort: compare({c:left[c] for c in ids}, {c:right[c] for c in ids}, ids)
            for cohort,ids in (("all",cohorts["caseIds"]),("common",cohorts["commonCases"]),("toolSpecific",cohorts["toolSpecificCases"]))}
    for split in ("dev", "holdout"):
        baseline, candidate = "baseline_" + split, "candidate_" + split
        comparison(split+"_rule_to_context",baseline,"rule",baseline,"llm_context",split)
        comparison(split+"_context_to_tools",baseline,"llm_context",baseline,"llm_tools",split)
        for mode in ("llm_context", "llm_tools"):
            comparison(split+"_citation_optimization_"+mode,baseline,mode,candidate,mode,split)
        comparison(split+"_candidate_context_to_tools",candidate,"llm_context",candidate,"llm_tools",split)
    report["failureCases"] = [{"arm": arm, **r} for arm,records in runs.items() for r in records.values()
                              if not r["taskPassed"] or r["quality"] is None]
    (root / "ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    args = p.parse_args()
    report = build_report(args.experiment)
    print(json.dumps({"arms": list(report["arms"]), "comparisons": list(report["comparisons"])}))
