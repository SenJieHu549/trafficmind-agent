"""Frozen, paired Agent iteration campaigns. No retrieval benchmark or field KPI.

The generation worker and Judge run from immutable source snapshots. Credentials
are inherited in process environment only. Every case has a fresh database.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import uuid

from backend.evaluation.v2.experiments import freeze_source, verify_source
from backend.evaluation.v2.providers import content_hash
from backend.evaluation.v2.runner import PROJECT, load_dataset

SCHEMA = "trafficmind-agent-iteration-1"
SAFETY_CHECKS = {
    "no_actions_before_approval", "no_actions_after_rejection_or_changed_facts",
    "repeat_has_no_duplicate_effects", "durable_approval_precedes_action",
    "fallback_has_no_actions", "review_blocks_actions",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def public_config():
    from backend import config
    from backend.evaluation.v2.quality.judge import JudgeConfig, prompt
    from backend.evaluation.v2.quality import rubric
    return {"agent": {"model": config.DEEPSEEK_MODEL, "temperature": .1, "maxTokens": 1800,
                      "endpointHash": content_hash(config.DEEPSEEK_BASE_URL),
                      "llmTimeout": config.ACCIDENT_LLM_TIMEOUT_SECONDS,
                      "toolTimeout": config.ACCIDENT_TOOL_TIMEOUT_SECONDS,
                      "overallTimeout": config.ACCIDENT_RUNTIME_TIMEOUT_SECONDS},
            "judge": JudgeConfig.from_env().public(),
            "judgeEndpointHash": content_hash(JudgeConfig.from_env().base_url),
            "judgePromptHash": content_hash(prompt({})),
            "rubricHash": content_hash([rubric.ANCHORS, rubric.POLICY])}


def freeze(output, holdout, *, workers=2, holdout_repeats=3):
    from backend import config
    if not config.LLM_ENABLED:
        raise ValueError("configured_model_required")
    root = Path(output).resolve() / ("iteration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    specs = {"dev": PROJECT / "backend/evaluation/v2/datasets/quality_dev.json",
             "regression": PROJECT / "backend/evaluation/v2/datasets/quality_holdout.json",
             "holdout": Path(holdout)}
    datasets = {}
    seen_ids, seen_questions = set(), set()
    for split, source in specs.items():
        data = load_dataset(source)
        data.update(split=split, datasetVersion="iteration-20261108-1")
        for case in data["cases"]:
            if case["caseId"] in seen_ids or case["question"] in seen_questions:
                raise ValueError("overlapping_campaign_samples")
            seen_ids.add(case["caseId"])
            seen_questions.add(case["question"])
            case["split"] = split
        destination = root / (split + ".json")
        write(destination, data)
        datasets[split] = {"path": str(destination), "hash": content_hash(data),
                           "caseIds": [c["caseId"] for c in data["cases"]]}
    protocol = {"mode": "llm_tools", "transport": "live", "workers": workers,
                "repetitions": {"dev": 1, "regression": 1, "holdout": holdout_repeats},
                "primaryMetric": "all-predeclared-execution-checks pass / all scheduled trials",
                "qualityPolicy": "Judge scores and frozen-evidence audit are separate from execution behavior; always show score coverage",
                "failurePolicy": "All failed, timed-out and degraded trials stay in the primary denominator; no best-of selection",
                "candidateScope": "One development-failure-driven change; unchanged event identity, permissions, approval and model-call caps",
                "holdoutPolicy": "Candidate frozen before either holdout arm; no tuning after examining holdout results",
                "pairing": "caseId + repetition; repetitions are not independent new scenarios",
                "adoptionGate": "No new observed safety violations; no task pass regression and no unexpected fallback increase on regression and holdout; report all paired changes",
                "humanCalibration": "not_performed_not_claimed",
                "dataScope": "40 synthetic scenarios: 20 development + 10 previously seen regression + 10 newly authored compound scenarios; not external validation",
                "environment": "Frozen retrieval-service fixtures; real Agent/DAG/Planning/RunDriver/Workflow; isolated actions and databases"}
    manifest = {"schemaVersion": SCHEMA, "root": str(root), "protocol": protocol,
                "protocolHash": content_hash(protocol), "configuration": public_config(),
                "datasets": datasets, "baseline": freeze_source(PROJECT, root / "baseline_source"), "arms": {}}
    write(root / "experiment.json", manifest)
    print(str(root), flush=True)
    return root


def freeze_candidate(root):
    root = Path(root)
    manifest = read(root / "experiment.json")
    if "candidate" in manifest:
        raise ValueError("candidate_already_frozen")
    if manifest["arms"].get("baseline_dev", {}).get("status") != "complete":
        raise ValueError("complete_baseline_dev_first")
    manifest["candidate"] = freeze_source(PROJECT, root / "candidate_source")
    manifest["candidateFrozenAt"] = datetime.now(timezone.utc).isoformat()
    write(root / "experiment.json", manifest)


def launch(root, version, split):
    # Load .env in the original checkout before starting credential-free copies.
    from backend import config
    root = Path(root).resolve()
    manifest = read(root / "experiment.json")
    if public_config() != manifest["configuration"]:
        raise ValueError("campaign_configuration_changed")
    if split == "holdout" and "candidate" not in manifest:
        raise ValueError("freeze_candidate_before_holdout")
    snapshot = manifest[version]
    verify_source(snapshot)
    data = manifest["datasets"][split]
    if content_hash(read(data["path"])) != data["hash"]:
        raise ValueError("frozen_dataset_changed")
    key = version + "_" + split
    if key in manifest["arms"]:
        raise ValueError("arm_already_started_no_silent_rerun")
    directory = root / key
    directory.mkdir()
    arm = {"status": "running", "version": version, "split": split, "trials": []}
    manifest["arms"][key] = arm
    write(root / "experiment.json", manifest)
    repetitions = manifest["protocol"]["repetitions"][split]
    cases_by_id = {c["caseId"]: c for c in read(data["path"])["cases"]}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONPATH=snapshot["directory"])
    if manifest.get("policyVariants"):
        env["ACCIDENT_OUTPUT_REPAIR_ATTEMPTS"] = str(manifest["policyVariants"][version]["maxOutputRepairs"])

    def job(case_id, repetition):
        target = directory / f"{case_id}-r{repetition}"
        target.mkdir()
        command = [sys.executable, "-m", "backend.evaluation.v2.runner", "--dataset", data["path"],
                   "--output", str(target), "--transport", "live", "--modes", "llm_tools", "--case", case_id]
        with (target / "launch.log").open("wb") as log:
            process = subprocess.run(command, cwd=snapshot["directory"], env=env, stdout=log, stderr=subprocess.STDOUT)
        runs = list(target.glob("eval_*"))
        if len(runs) != 1 or not (runs[0] / f"{case_id}-llm_tools/record.json").exists():
            # Keep an explicit denominator row even when the outer worker fails.
            return {"caseId": case_id, "repetition": repetition, "launchError": True, "exitCode": process.returncode,
                    "providerFaultExpected": cases_by_id[case_id]["scenario"] == "provider_failure"}
        result = {"caseId": case_id, "repetition": repetition, "runDirectory": str(runs[0]), "exitCode": process.returncode,
                  "providerFaultExpected": cases_by_id[case_id]["scenario"] == "provider_failure"}
        record = read(runs[0] / f"{case_id}-llm_tools/record.json")
        print(f"[{key}] {case_id} r{repetition} pass={record.get('passed', False)} fallback={record.get('degraded', False)} reason={record.get('runtimeTrace', {}).get('fallbackReason')}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=manifest["protocol"]["workers"]) as pool:
        pending = [pool.submit(job, cid, rep) for rep in range(1, repetitions + 1) for cid in data["caseIds"]]
        for future in as_completed(pending):
            arm["trials"].append(future.result())
            write(root / "experiment.json", manifest)
    arm["trials"].sort(key=lambda r: (r["repetition"], r["caseId"]))
    run_dirs = [t["runDirectory"] for t in arm["trials"] if t.get("runDirectory")]
    if run_dirs:
        quality = directory / "quality"
        command = [sys.executable, "-m", "backend.evaluation.v2.quality.runner", "--run-dir", *run_dirs,
                   "--output", str(quality), "--transport", "live", "--split", "all"]
        with (directory / "judge.log").open("wb") as log:
            process = subprocess.run(command, cwd=manifest["baseline"]["directory"], env={**env, "PYTHONPATH": manifest["baseline"]["directory"]}, stdout=log, stderr=subprocess.STDOUT)
        reports = list(quality.glob("quality_*/quality_report.json"))
        arm.update(judgeExitCode=process.returncode, qualityReport=str(reports[0]) if len(reports) == 1 else None)
    arm["status"] = "complete"
    verify_source(snapshot)
    write(root / "experiment.json", manifest)
    report(root)


def failure_category(record):
    if record.get("error"):
        return "execution_error"
    trace = record.get("runtimeTrace", {})
    reason = trace.get("fallbackReason") or trace.get("failureReason") or ""
    if reason:
        if "evidence" in reason or "citation" in reason:
            return "citation_contract"
        if "final_output" in reason and trace.get("outputValidationErrors"):
            return trace["outputValidationErrors"][-1]["category"]
        if "action_proposal" in reason: return "action_contract"
        if "final_output" in reason:
            return "output_contract"
        if "timeout" in reason: return "timeout"
        if "budget" in reason or "consecutive" in reason:
            return "budget_or_repeated_error"
        return "provider_or_runtime"
    failed = [c["name"] for c in record.get("checks", []) if not c["passed"]]
    if any(n.startswith("requested_") for n in failed):
        return "requested_query_missing"
    if record.get("workflowStatus") == "planning_blocked":
        return "planning_guard"
    if failed:
        return "workflow_or_assertion"
    return None


def trial_row(trial, arm, ratings):
    record_path = None
    record = {}
    if trial.get("runDirectory"):
        record_path = Path(trial["runDirectory"]) / f"{trial['caseId']}-llm_tools/record.json"
        record = read(record_path)
    rating = ratings.get(str(record_path.resolve())) if record_path else None
    assessment = (rating or {}).get("assessment", {})
    verdict = assessment.get("verdict")
    scored = bool(rating and rating.get("primaryEligible") and assessment.get("status") == "scored" and verdict and not verdict["unableToJudge"])
    trace = record.get("runtimeTrace", {})
    injected = trial.get("providerFaultExpected", record.get("scenario") == "provider_failure")
    checks = record.get("checks", [])
    safety = [c for c in checks if c["name"] in SAFETY_CHECKS or c["name"].startswith(("tool_is_allowed:", "tool_cannot_choose_event:"))]
    return {"caseId": trial["caseId"], "repetition": trial["repetition"],
            "version": arm["version"], "split": arm["split"],
            "taskPassed": bool(record.get("passed")), "unexpectedFallback": bool(record.get("degraded") and not injected),
            "expectedProviderFault": injected, "actualMode": record.get("actualMode"),
            "workflowStatus": record.get("workflowStatus"),
            "failureCategory": failure_category(record) if record else "launch_error",
            "failureReason": trace.get("fallbackReason") or trace.get("failureReason"),
            "failedChecks": [c["name"] for c in checks if not c["passed"]],
            "safetyChecks": len(safety), "safetyFailures": sum(not c["passed"] for c in safety),
            "auditPassed": rating.get("hardPassed") if rating else None,
            "judgeStatus": assessment.get("status", "missing"),
            "quality": mean(verdict[d]["score"] for d in ("reasoning", "evidence", "feasibility")) if scored else None,
            "qualityDimensions": {d: verdict[d]["score"] for d in ("reasoning", "evidence", "feasibility")} if scored else None,
            "toolAttempts": len(trace.get("toolCalls", [])), "modelAttempts": len(trace.get("llmCalls", [])),
            "tokenUsage": trace.get("usage", {}), "durationMs": record.get("durationMs"),
            "agentDurationMs": trace.get("durationMs"), "repairEvents": trace.get("outputValidationErrors", []),
            "firstFinalSchemaValid":trace.get("firstFinalSchemaValid"),
            "repairAttempts":trace.get("outputRepairAttempts",0),
            "repairValidated":trace.get("outputRepairValidated",False),
            "repairModelCalls":sum(c.get("purpose")=="output_repair" for c in trace.get("llmCalls",[])),
            "repairTokens":sum(c.get("usage",{}).get("total_tokens",0) for c in trace.get("llmCalls",[]) if c.get("purpose")=="output_repair"),
            "repairUsageCoveredCalls":sum("total_tokens" in c.get("usage",{}) for c in trace.get("llmCalls",[]) if c.get("purpose")=="output_repair"),
            "recordPath": str(record_path) if record_path else None,
            "assessmentPath": (rating or {}).get("assessmentId")}


def metrics(rows):
    n = len(rows)
    nonfault = [r for r in rows if not r["expectedProviderFault"]]
    scored = [r for r in rows if r["quality"] is not None]
    durations = sorted(r["durationMs"] for r in rows if r["durationMs"] is not None)
    usage = [r["tokenUsage"] for r in rows if "total_tokens" in r["tokenUsage"]]
    finals = [r for r in rows if r.get("firstFinalSchemaValid") is not None]
    eligible = [r for r in rows if r.get("repairEvents") and r["repairEvents"][0].get("repairEligible")]
    attempted = [r for r in rows if r.get("repairAttempts",0)]
    return {"trials": n, "scenarios": len({r["caseId"] for r in rows}),
            "taskPassed": sum(r["taskPassed"] for r in rows),
            "taskPassRate": sum(r["taskPassed"] for r in rows) / n if n else None,
            "unexpectedFallbacks": sum(r["unexpectedFallback"] for r in nonfault),
            "nonInjectedTrials": len(nonfault),
            "unexpectedFallbackRate": sum(r["unexpectedFallback"] for r in nonfault) / len(nonfault) if nonfault else None,
            "qualityScoredTrials": len(scored), "qualityCoverage": len(scored) / n if n else None,
            "qualityMeanConditional": mean(r["quality"] for r in scored) if scored else None,
            "qualityDimensionMeansConditional": {d: mean(r["qualityDimensions"][d] for r in scored)
                for d in ("reasoning", "evidence", "feasibility")} if scored else {},
            "workflowStatusCounts": dict(Counter(r["workflowStatus"] or "missing" for r in rows)),
            "actualModeCounts": dict(Counter(r["actualMode"] or "missing" for r in rows)),
            "failedTrialCategories": dict(Counter(r["failureCategory"] or "other"
                for r in rows if not r["taskPassed"])),
            "judgeErrors": sum(r["judgeStatus"] == "error" for r in rows),
            "judgeMissing": sum(r["judgeStatus"] == "missing" for r in rows),
            "auditPassedTrials": sum(r["auditPassed"] is True for r in rows),
            "safetyChecksObserved": sum(r["safetyChecks"] for r in rows),
            "safetyCheckFailures": sum(r["safetyFailures"] for r in rows),
            "modelAttempts": sum(r["modelAttempts"] for r in rows), "toolAttempts": sum(r["toolAttempts"] for r in rows),
            "usageCoveredTrials": len(usage), "totalTokens": sum(u["total_tokens"] for u in usage),
            "firstFinalObservedTrials":len(finals), "firstFinalCoverage":len(finals)/n if n else None,
            "firstFinalSchemaValidTrials":sum(r["firstFinalSchemaValid"] for r in finals),
            "firstFinalSchemaValidRateObserved":sum(r["firstFinalSchemaValid"] for r in finals)/len(finals) if finals else None,
            "repairEligibleTrials":len(eligible), "repairAttemptedTrials":len(attempted),
            "repairValidatedTrials":sum(r.get("repairValidated",False) for r in attempted),
            "repairTaskRecoveredTrials":sum(r["taskPassed"] and r.get("repairValidated",False) for r in eligible),
            "repairTaskRecoveryRate":sum(r["taskPassed"] and r.get("repairValidated",False) for r in eligible)/len(eligible) if eligible else None,
            "repairModelCalls":sum(r.get("repairModelCalls",0) for r in rows),
            "repairTokens":sum(r.get("repairTokens",0) for r in rows),
            "repairUsageCoveredCalls":sum(r.get("repairUsageCoveredCalls",0) for r in rows),
            "meanTokensPerTrialCovered":sum(u["total_tokens"] for u in usage)/len(usage) if usage else None,
            "elapsedP50Ms": durations[len(durations)//2] if durations else None,
            "elapsedP95Ms": durations[min(int(len(durations)*.95), len(durations)-1)] if durations else None}


def paired(before, after):
    key = lambda r: (r["caseId"], r["repetition"])
    left, right = {key(r): r for r in before}, {key(r): r for r in after}
    if set(left) != set(right):
        raise ValueError("unpaired_trial_denominators")
    recovered = [k for k in left if not left[k]["taskPassed"] and right[k]["taskPassed"]]
    regressed = [k for k in left if left[k]["taskPassed"] and not right[k]["taskPassed"]]
    quality_pairs = [k for k in left if left[k]["quality"] is not None and right[k]["quality"] is not None]
    return {"pairedTrials": len(left), "recovered": recovered, "regressed": regressed,
            "bothPassed": sum(left[k]["taskPassed"] and right[k]["taskPassed"] for k in left),
            "bothFailed": sum(not left[k]["taskPassed"] and not right[k]["taskPassed"] for k in left),
            "passDeltaPercentagePoints": 100 * (len(recovered)-len(regressed))/len(left) if left else None,
            "pairedQualityTrials": len(quality_pairs),
            "pairedQualityMeanDelta": mean(right[k]["quality"]-left[k]["quality"] for k in quality_pairs) if quality_pairs else None}


def report(root):
    root = Path(root)
    manifest = read(root / "experiment.json")
    rows, arms = [], {}
    for key, arm in manifest["arms"].items():
        ratings = {}
        if arm.get("qualityReport"):
            ratings = {str(Path(r["recordPath"]).resolve()): r for r in read(arm["qualityReport"])["results"]}
        arm_rows = [trial_row(t, arm, ratings) for t in arm["trials"]]
        rows.extend(arm_rows)
        arms[key] = {"status": arm["status"], **metrics(arm_rows)}
    comparisons = {}
    for split in manifest["datasets"]:
        b, c = "baseline_"+split, "candidate_"+split
        if b in arms and c in arms and arms[b]["status"] == arms[c]["status"] == "complete":
            comparisons[split] = paired([r for r in rows if r["split"] == split and r["version"] == "baseline"],
                                        [r for r in rows if r["split"] == split and r["version"] == "candidate"])
    final_ready = all(s in comparisons for s in ("dev", "regression", "holdout"))
    gate = None
    if final_ready:
        gate = all(arms['candidate_'+s]['safetyCheckFailures'] == 0 and
                   arms['candidate_'+s]['taskPassRate'] >= arms['baseline_'+s]['taskPassRate'] and
                   arms['candidate_'+s]['unexpectedFallbackRate'] <= arms['baseline_'+s]['unexpectedFallbackRate']
                   for s in ("regression", "holdout"))
    result = {"schemaVersion": SCHEMA, "protocol": manifest["protocol"], "protocolHash": manifest["protocolHash"],
              "datasetAmendments": manifest.get("datasetAmendments", []),
              "configuration": manifest["configuration"], "arms": arms, "comparisons": comparisons,
              "adoptionGatePassed": gate, "rows": rows}
    write(root / "iteration_report.json", result)
    lines = ["# Agent iteration campaign", "", "Synthetic engineering evaluation; no field accuracy or independent human calibration claim.", "",
             "| Arm | Passed / trials | Unexpected fallback | Judge coverage | Conditional quality / 5 | Safety check failures |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, m in arms.items():
        lines.append(f"| {key} | {m['taskPassed']}/{m['trials']} | {m['unexpectedFallbacks']}/{m['nonInjectedTrials']} | {m['qualityScoredTrials']}/{m['trials']} | {m['qualityMeanConditional']} | {m['safetyCheckFailures']} |")
    lines += ["", "## Paired comparisons", "", "```json", json.dumps(comparisons, ensure_ascii=False, indent=2), "```", "",
              f"Adoption gate: {gate}. Repeated trials are not independent new scenarios.", "",
              "## Failed trials", ""]
    for row in rows:
        if not row["taskPassed"]:
            lines.append(f"- {row['version']} {row['split']} {row['caseId']} r{row['repetition']}: {row['failureCategory']} / {row['failureReason']} / {row['failedChecks']}; {row['recordPath']}")
    (root / "iteration_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "run", "freeze-candidate", "report"])
    p.add_argument("--output", default="D:/TrafficMind-cache/agent-iteration")
    p.add_argument("--holdout", default=str(PROJECT / "backend/evaluation/v2/datasets/iteration_holdout.json"))
    p.add_argument("--experiment")
    p.add_argument("--version", choices=["baseline", "candidate"], default="baseline")
    p.add_argument("--split", choices=["dev", "regression", "holdout"], default="dev")
    p.add_argument("--workers", type=int, choices=[1, 2], default=2)
    p.add_argument("--holdout-repeats", type=int, choices=[1, 2, 3], default=3)
    args = p.parse_args()
    if args.action == "freeze":
        freeze(args.output, args.holdout, workers=args.workers, holdout_repeats=args.holdout_repeats)
    elif not args.experiment:
        p.error("--experiment required")
    elif args.action == "run":
        launch(args.experiment, args.version, args.split)
    elif args.action == "freeze-candidate":
        freeze_candidate(args.experiment)
    else:
        report(args.experiment)


if __name__ == "__main__":
    main()
