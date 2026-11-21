"""Conditional output-recovery benchmark: injected faults, real correction calls.

Historical valid model/tool prefixes are replayed with strict request hashes.
Only their last output is corrupted. This is deliberately NOT an estimate of
natural error frequency, fresh end-to-end task success, or semantic quality.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

# All serving configuration is import-time. Isolate before backend imports.
if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "worker":
    _request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    _directory = Path(_request["directory"]).resolve()
    for _key, _file in {"TRAFFICMIND_DB_PATH": "events.db", "RAG_V2_DB_PATH": "rag.db",
                        "RAG_V2_FTS_PATH": "fts.db", "RAG_V2_VECTOR_DB_PATH": "vectors"}.items():
        os.environ[_key] = str(_directory / _file)

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.experiments import freeze_source, verify_source
from backend.evaluation.v2.providers import content_hash

FAULTS = ("truncated_json", "missing_suggestion", "confidence_type", "findings_overflow", "urgency_enum")
CONTROLS = ("unknown_reference", "invalid_action")
CONTEXTS = ("D01", "D05", "D09", "D10", "D11", "D13", "D19", "D20")
DEFAULT_PARENT = "D:/TrafficMind-cache/output-repair/iteration_20261109T014049_8ef79530"


def mutate(content, kind):
    body = json.loads(content)
    if kind == "truncated_json":
        return json.dumps(body, ensure_ascii=False).rstrip()[:-1]
    if kind == "missing_suggestion":
        del body["suggestion"]
    elif kind == "confidence_type":
        body["confidence"] = "uncertain"
    elif kind == "findings_overflow":
        body["findings"] = (body["findings"] * 11)[:11]
    elif kind == "urgency_enum":
        body["urgency"] = "urgent"
    elif kind in CONTROLS:
        # Ordinary invalid field PLUS a protected-contract violation: never
        # enter the repair branch simply because a repairable field also fails.
        body["confidence"] = "uncertain"
        if kind == "unknown_reference":
            body["evidence_refs"] = ["ev_not_in_this_event_ledger"]
        else:
            body["proposed_actions"] = [{"actionType": "delete_event", "params": {}}]
    else:
        raise ValueError("unknown_fault_kind")
    return json.dumps(body, ensure_ascii=False)


def feedback_for(content, snapshot):
    from pydantic import ValidationError
    from backend.agent.llm_accident.contracts import EvidenceLedger, FinalAnalysis
    from backend.agent.llm_accident.output_validation import validation_feedback
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    from backend.agent.llm_accident.citations import resolve_citations
    ledger = EvidenceLedger(snapshot["eventId"], snapshot["asOf"])
    for e in snapshot["evidence"]:
        assert ledger.add(e["kind"], e["sourceId"], e["data"], e["origins"][0]) == e["evidenceId"]
    try:
        FinalAnalysis.model_validate_json(content)
    except ValidationError as exc:
        return validation_feedback(exc, content, ledger, compact=False,
            validate_actions=AccidentAgentRuntime._validate_actions, resolve_refs=resolve_citations)
    raise ValueError("fault_did_not_break_schema")


def freeze(output, parent=DEFAULT_PARENT):
    from datetime import datetime, timezone
    import uuid
    from backend import config
    from backend.evaluation.rag_ablation import source_fingerprint
    if not config.LLM_ENABLED:
        raise ValueError("configured_live_model_required")
    parent = Path(parent).resolve()
    root = Path(output).resolve() / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    source_rows = it.read(parent / "iteration_report.json")["rows"]
    items = []
    for case_id in CONTEXTS:
        row = next(r for r in source_rows if r["version"] == "baseline" and r["split"] == "dev" and r["caseId"] == case_id)
        assert row["taskPassed"]
        path = Path(row["recordPath"])
        record, cassette = it.read(path), it.read(path.parent / "cassette.json")
        assert len(cassette) < 4 and not cassette[-1]["reply"]["message"].get("tool_calls")
        assert record["runtimeTrace"]["citationPolicy"] == "canonical-evidence-id-1"
        for kind in (*FAULTS, *CONTROLS):
            prefix = deepcopy(cassette)
            content = mutate(prefix[-1]["reply"]["message"]["content"], kind)
            prefix[-1]["reply"]["message"]["content"] = content
            # The injected output is not a newly billed model response.
            prefix[-1]["reply"]["usage"] = {}
            prefix[-1]["reply"]["model"] = "synthetic-output-fault"
            feedback = feedback_for(content, record["evidenceSnapshot"])
            assert feedback["repairEligible"] == (kind in FAULTS)
            items.append({"key": case_id + "_" + kind, "caseId": case_id, "kind": kind,
                "control": kind in CONTROLS, "originalRecord": str(path),
                "originalRecordHash": content_hash(record), "originalCassetteHash": content_hash(cassette),
                "request": it.read(path.parent / "request.json"), "cassette": prefix,
                "prefixHash": content_hash(prefix), "injectedContentHash": content_hash(content),
                "citationPolicy": "canonical", "eligibility": feedback})
    paths = it.read(parent / "source_store_before.json")["paths"]
    it.write(root / "source_store_before.json", {"paths": paths, "fingerprint": source_fingerprint(paths)})
    manifest = {"schemaVersion": "trafficmind-conditional-recovery-1", "parent": str(parent),
        "scope": "synthetic fault-injection conditional recovery, strict historical prefix plus live correction",
        "configuration": it.public_config()["agent"], "contexts": list(CONTEXTS), "faultTypes": list(FAULTS),
        "batches": 3, "workers": 3, "faultSamples": 40, "candidateTrials": 120,
        "baselineTrials": 40, "protectedControlTrials": 16,
        "primaryMetric": "record.passed AND outputRepairValidated AND runtime completed / all 120 scheduled candidate trials",
        "denominatorPolicy": "Keep provider errors, timeouts, invalid corrections and downstream failures; no retry or best-of selection.",
        "limits": "One correction maximum, within original 4 model calls and unchanged deadlines; tools disabled during correction.",
        "labelPolicy": "Original case assertions unmodified. Expected rejection and changed-facts blocking count as correct behavior.",
        "batchPolicy": "Same 40 injected outputs in each of 3 sequential batches; 8 contexts, not 120 independent scenarios.",
        "baselinePolicy": "One deterministic fail-fast replay per unique injected output. No network calls and no per-batch duplication.",
        "controlPolicy": "16 mixed ordinary+protected violations; must block with zero correction calls and zero action effects.",
        "exclusions": "No natural-error-rate, held-out overall task accuracy, human semantic scoring, or production enablement claim.",
        "cases": items}
    manifest["source"] = freeze_source(it.PROJECT, root / "source")
    manifest["manifestHash"] = content_hash(manifest)
    it.write(root / "manifest.json", manifest)
    print("FROZEN " + str(root), flush=True)
    return root


def worker(request_path, spec_path):
    # Reuse the existing real Agent/DAG/Planning/Workflow isolation and strict
    # prefix matching. This wrapper only labels the deliberate corruption.
    from backend.evaluation.v2.historical_repair import worker as historical_worker
    historical_worker(request_path, spec_path)
    directory = Path(it.read(request_path)["directory"])
    spec, record = it.read(spec_path), it.read(directory / "record.json")
    record.update(transport="fault_injected_prefix_then_live", outputFaultInjected=True,
        outputFaultKind=spec["kind"], outputFaultControl=spec["control"],
        naturalFailureReplay=False, benchmarkScope="conditional_output_recovery")
    it.write(directory / "record.json", record)


def trial(root, manifest, item, arm, batch):
    import subprocess
    directory = root / (arm + (f"_batch{batch}" if batch else "")) / item["key"]
    directory.mkdir(parents=True, exist_ok=False)
    request = deepcopy(item["request"])
    request.update(directory=str(directory), transport="live")
    request.pop("cassette", None)
    it.write(directory / "request.json", request)
    it.write(directory / "fault_spec.json", item)
    env = {**os.environ, "ACCIDENT_OUTPUT_REPAIR_ATTEMPTS": "0" if arm == "baseline" else "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPATH": manifest["source"]["directory"]}
    with (directory / "worker.log").open("wb") as log:
        try:
            result = subprocess.run([sys.executable, "-m", "backend.evaluation.v2.recovery_benchmark", "worker",
                str(directory / "request.json"), str(directory / "fault_spec.json")],
                cwd=manifest["source"]["directory"], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess([], -1)
    if result.returncode:
        # An infrastructure failure stays a failed scheduled trial; no rerun.
        it.write(directory / "infrastructure_failure.json", {"exitCode": result.returncode})
    return {"key": item["key"], "arm": arm, "batch": batch, "caseId": item["caseId"],
            "kind": item["kind"], "control": item["control"], "directory": str(directory)}


def recovered(record):
    trace = record.get("runtimeTrace", {})
    return bool(record.get("passed") and trace.get("outputRepairValidated") and trace.get("status") == "completed")


def audit(root):
    from collections import Counter
    from statistics import mean
    from backend.agent.llm_accident.contracts import FinalAnalysis
    from backend.evaluation.rag_ablation import source_fingerprint
    root = Path(root)
    manifest = it.read(root / "manifest.json")
    saved = deepcopy(manifest); expected = saved.pop("manifestHash")
    assert content_hash(saved) == expected
    verify_source(manifest["source"])
    paths = it.read(root / "source_store_before.json")
    assert source_fingerprint(paths["paths"]) == paths["fingerprint"]
    rows = []
    for spec in it.read(root / "trials.json"):
        item = next(c for c in manifest["cases"] if c["key"] == spec["key"])
        path = Path(spec["directory"]) / "record.json"
        if not path.exists():
            rows.append({**spec, "recovered": False, "infrastructureFailure": True})
            continue
        record, cassette = it.read(path), it.read(path.parent / "cassette.json")
        trace = record["runtimeTrace"]
        n = len(item["cassette"])
        assert record["historicalPrefixFullyMatched"] and cassette[:n] == item["cassette"]
        assert content_hash(cassette[n-1]["reply"]["message"]["content"]) == item["injectedContentHash"]
        assert len(trace["llmCalls"]) == len(cassette) <= 4
        assert trace["limits"]["max_output_repairs"] == (0 if spec["arm"] == "baseline" else 1)
        assert trace["firstFinalSchemaValid"] is False
        assert record["passed"] == all(c["passed"] for c in record["checks"])
        fresh = cassette[n:]
        assert len(fresh) == record["freshRecoveryRequestAttempts"] == trace["outputRepairAttempts"] <= 1
        for entry in cassette:
            assert content_hash(entry["request"]) == entry["requestHash"]
        assert all(e["request"]["allow_tools"] is False for e in fresh)
        assert all(not c["toolsAllowed"] for c in trace["llmCalls"] if c["purpose"] == "output_repair")
        if spec["arm"] == "baseline" or spec["control"]:
            assert not fresh and not record.get("actions") and not record.get("isolatedEffects")
        if spec["control"]:
            assert not trace["outputValidationErrors"][0]["repairEligible"]
        else:
            assert trace["outputValidationErrors"][0]["repairEligible"]
        schema_valid = False
        if fresh and fresh[-1].get("reply"):
            try:
                FinalAnalysis.model_validate_json(fresh[-1]["reply"]["message"].get("content") or "")
            except ValueError:
                pass
            else:
                schema_valid = True
        tokens = sum(e.get("reply", {}).get("usage", {}).get("total_tokens", 0) for e in fresh)
        rows.append({**spec, "recovered": recovered(record), "taskPassed": record["passed"],
            "repairValidated": trace["outputRepairValidated"], "schemaValidAfterCorrection": schema_valid,
            "failureReason": trace["failureReason"], "failedChecks": [c["name"] for c in record["checks"] if not c["passed"]],
            "repairCalls": len(fresh), "repairTokens": tokens,
            "repairUsageCovered": all("total_tokens" in e.get("reply", {}).get("usage", {}) for e in fresh),
            "repairDurationMs": sum(c["durationMs"] for c in trace["llmCalls"] if c["purpose"] == "output_repair"),
            "workflowStatus": record["workflowStatus"], "recordPath": str(path)})
    def stats(values):
        return {"trials": len(values), "recovered": sum(r["recovered"] for r in values),
            "recoveryRate": sum(r["recovered"] for r in values) / len(values) if values else None}
    candidates = [r for r in rows if r["arm"] == "candidate"]
    baseline = [r for r in rows if r["arm"] == "baseline"]
    controls = [r for r in rows if r["control"]]
    assert len(rows) == 176 and len(candidates) == 120 and len(baseline) == 40 and len(controls) == 16
    assert len({(r["arm"], r["batch"], r["key"]) for r in rows}) == len(rows)
    complete = [r for r in candidates if not r.get("infrastructureFailure")]
    durations = sorted(r["repairDurationMs"] for r in complete)
    import math
    percentile = lambda p: durations[max(0, math.ceil(len(durations)*p)-1)] if durations else None
    result = {"scope": manifest["scope"], "contexts": len(CONTEXTS), "faultTypes": len(FAULTS),
        "uniqueFaultSamples": 40, "batches": 3, "candidate": stats(candidates), "baseline": stats(baseline),
        "byBatch": {str(b): stats([r for r in candidates if r["batch"] == b]) for b in (1, 2, 3)},
        "byFault": {k: stats([r for r in candidates if r["kind"] == k]) for k in FAULTS},
        "byContext": {k: stats([r for r in candidates if r["caseId"] == k]) for k in CONTEXTS},
        "stableSamplesAllThreeBatches": sum(all(r["recovered"] for r in candidates if r["key"] == k)
            for k in {r["key"] for r in candidates}),
        "protectedControls": {"trials": len(controls), "blockedWithoutRepairOrEffects": sum(not r.get("infrastructureFailure") for r in controls)},
        "correctionSchemaValidTrials": sum(r.get("schemaValidAfterCorrection", False) for r in candidates),
        "postCorrectionValidatedTrials": sum(r.get("repairValidated", False) for r in candidates),
        "freshCorrectionRequests": sum(r.get("repairCalls", 0) for r in candidates),
        "freshCorrectionTokens": sum(r.get("repairTokens", 0) for r in candidates),
        "usageCoveredResponses": sum(r.get("repairUsageCovered", False) and r.get("repairCalls", 0) > 0 for r in candidates),
        "correctionLatencyMs": {"mean": mean(durations) if durations else None, "p50": percentile(.5), "p95": percentile(.95)},
        "failureReasons": dict(Counter(r.get("failureReason") or ("downstream_assertion" if not r["recovered"] else "none") for r in candidates)),
        "infrastructureFailures": sum(bool(r.get("infrastructureFailure")) for r in rows),
        "audit": {"sourceManifestAndProtocolVerified": True, "allPrefixRequestHashesVerified": True,
            "sourceRagStoresUnchanged": True, "correctionToolsDisabled": True, "globalFourCallCapPreserved": True,
            "defaultRepairFlagChanged": False}, "rows": rows}
    it.write(root / "report.json", result)
    return result


def run(root):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from backend import config
    root = Path(root); manifest = it.read(root / "manifest.json")
    if (root / "state.json").exists():
        raise ValueError("campaign_already_started_no_resampling")
    saved = deepcopy(manifest); expected = saved.pop("manifestHash")
    assert content_hash(saved) == expected
    assert it.public_config()["agent"] == manifest["configuration"]
    verify_source(manifest["source"])
    it.write(root / "state.json", {"status": "running"})
    trials = []
    def group(arm, batch, items):
        with ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
            futures = [pool.submit(trial, root, manifest, item, arm, batch) for item in items]
            for future in as_completed(futures):
                row = future.result(); trials.append(row)
                it.write(root / "trials.json", trials)
                record_path = Path(row["directory"]) / "record.json"
                record = it.read(record_path) if record_path.exists() else {}
                print(f"{arm} batch={batch} {row['key']} recovered={recovered(record)} {len(trials)}/176", flush=True)
    try:
        samples = [c for c in manifest["cases"] if not c["control"]]
        controls = [c for c in manifest["cases"] if c["control"]]
        group("baseline", 0, samples)
        group("control", 0, controls)
        # Validate deterministic transport/isolation before spending on models.
        for row in trials:
            record = it.read(Path(row["directory"]) / "record.json")
            assert record["historicalPrefixFullyMatched"] and not record["freshRecoveryRequestAttempts"]
            assert not record.get("actions") and not record.get("isolatedEffects")
            assert record["runtimeTrace"]["failureReason"] == "invalid_final_output"
        for batch in (1, 2, 3):
            group("candidate", batch, samples)
        report = audit(root)
        it.write(root / "state.json", {"status": "complete", "result": report["candidate"]})
        print("COMPLETE " + json.dumps(report["candidate"]), flush=True)
    except Exception:
        it.write(root / "state.json", {"status": "failed", "completedTrials": len(trials)})
        raise


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PARENT)
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "audit":
        print(json.dumps(audit(sys.argv[2])["candidate"]))
    else:
        worker(sys.argv[2], sys.argv[3])
