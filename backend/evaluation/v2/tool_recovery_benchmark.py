"""Small conditional tool-recovery suite, compatible with the format suite.

Replay exactly the first historical tool-call turn; continue with real model
calls after an injected parameter error or a one-shot query exception. Never
replace generation, guards, planning, workflow or tools with a scripted success.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] in {"worker", "format_worker"}:
    _request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    _directory = Path(_request["directory"]).resolve()
    for _key, _name in {"TRAFFICMIND_DB_PATH": "events.db", "RAG_V2_DB_PATH": "rag.db",
                        "RAG_V2_FTS_PATH": "fts.db", "RAG_V2_VECTOR_DB_PATH": "vectors"}.items():
        os.environ[_key] = str(_directory / _name)

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.experiments import freeze_source, verify_source
from backend.evaluation.v2.providers import content_hash, safe_value

PARENT = "D:/TrafficMind-cache/output-repair/iteration_20261109T014049_8ef79530"
FORMAT_SUITE = "D:/TrafficMind-cache/output-repair/recovery_20261109T025515_2c6a14c6"
CONTEXTS = {"D10": "accident_knowledge_search", "D13": "accident_case_search", "D15": "accident_case_search"}
FAULTS = ("invalid_limit", "transient_query_error")


def inject_limit(reply, tool):
    reply = deepcopy(reply)
    call = next(c for c in reply["message"]["tool_calls"] if c["function"]["name"] == tool)
    params = json.loads(call["function"]["arguments"])
    params["limit"] = 99
    call["function"]["arguments"] = json.dumps(params, ensure_ascii=False)
    reply.update(usage={}, model="synthetic-tool-parameter-fault")
    return reply


def assess(case, record, spec, query_attempts):
    """Allow only the exact predeclared injected error; leave new errors red."""
    from backend.evaluation.v2.checks import evaluate
    checks = evaluate(case, record)
    record["checksBeforeInjectionAllowance"] = deepcopy(checks)
    calls = record.get("runtimeTrace", {}).get("toolCalls", [])
    target = next((c for c in calls if c["callId"] == spec["faultCallId"]), None)
    expected_code = "invalid_tool_arguments" if spec["kind"] == "invalid_limit" else "tool_execution_failed"
    observed = bool(target and target["tool"] == spec["tool"] and target.get("code") == expected_code)
    injected_ids = {spec["faultCallId"]}
    if spec["control"]:
        injected_ids = {c["callId"] for c in calls if c["tool"] == spec["tool"] and c.get("code") == expected_code}
    for check in checks:
        prefix = "tool_returned_valid_observation:"
        if check["name"].startswith(prefix) and check["name"][len(prefix):] in injected_ids and observed:
            check.update(passed=True, detail="Exact predeclared injected tool failure, retained in trace")
    checks.append({"name": "declared_tool_fault_observed", "passed": observed, "detail": spec["kind"]})
    successful = [c for c in calls if c["tool"] == spec["tool"] and c.get("status") == "ok" and c.get("evidenceIds")]
    if not spec["control"]:
        checks.append({"name": "required_tool_recovered_with_evidence", "passed": bool(successful), "detail": spec["tool"]})
        # This suite asks for one correction/retry, rather than counting an
        # unlimited eventually-successful chain as recovery.
        checks.append({"name": "target_tool_at_most_one_retry", "passed": sum(c["tool"] == spec["tool"] for c in calls) <= 2, "detail": "first attempt plus one retry"})
        checks.append({"name": "injected_parameter_did_not_execute", "passed": spec["kind"] != "invalid_limit" or all(a["params"].get("limit") != 99 for a in query_attempts), "detail": "real validator blocks invalid parameters"})
    record["toolFaultObserved"] = observed
    record["toolRecoveryObtainedEvidence"] = bool(successful)
    return checks


def is_recovered(record):
    return bool(record.get("passed") and record.get("toolFaultObserved")
        and record.get("toolRecoveryObtainedEvidence") and record.get("runtimeTrace", {}).get("status") == "completed")


def freeze(output):
    from datetime import datetime, timezone
    import uuid
    from backend.evaluation.rag_ablation import source_fingerprint
    from backend import config
    assert config.LLM_ENABLED
    root = Path(output).resolve() / ("tool_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    parent, format_root = Path(PARENT), Path(FORMAT_SUITE)
    report = it.read(parent / "iteration_report.json")
    cases = []
    for case_id, tool in CONTEXTS.items():
        row = next(r for r in report["rows"] if r["version"] == "baseline" and r["split"] == "dev" and r["caseId"] == case_id)
        assert row["taskPassed"]
        path = Path(row["recordPath"])
        cassette = it.read(path.parent / "cassette.json")
        assert any(c["function"]["name"] == tool for c in cassette[0]["reply"]["message"]["tool_calls"])
        for kind in (*FAULTS, "persistent_query_error"):
            prefix = deepcopy(cassette[:1])
            if kind == "invalid_limit":
                prefix[0]["reply"] = inject_limit(prefix[0]["reply"], tool)
            call_id = next(c["id"] for c in prefix[0]["reply"]["message"]["tool_calls"] if c["function"]["name"] == tool)
            cases.append({"key": case_id + "_" + kind, "caseId": case_id, "tool": tool,
                "kind": kind, "faultCallId": call_id, "control": kind == "persistent_query_error",
                "prefix": prefix, "prefixHash": content_hash(prefix),
                "originalRecord": str(path), "originalRecordHash": content_hash(it.read(path)),
                "originalCassetteHash": content_hash(cassette), "request": it.read(path.parent / "request.json")})
    manifest = {"schemaVersion": "trafficmind-tool-recovery-1", "formatSuite": str(format_root),
        "formatReportHash": content_hash(it.read(format_root / "report.json")), "configuration": it.public_config()["agent"],
        "contexts": list(CONTEXTS), "cases": cases, "batches": 3, "workers": 3, "recoveryTrials": 18, "baselineTrials": 6, "controlTrials": 3,
        "scope": "synthetic fault-injection conditional recovery; first model turn replayed, subsequent decisions live",
        "primaryMetric": "all original task checks except exact declared fault + fault seen + required tool returned evidence + one retry maximum + runtime completed / 18 scheduled trials",
        "denominatorPolicy": "All scheduled trials retained, including provider errors, extra retries, citation failures and downstream failures. No reruns or sample substitution.",
        "controlPolicy": "Persistent errors are outside recoverable denominator. Report fail-closed planning block/no action effects separately; no success assumed.",
        "productionPolicy": "Baseline uses old error feedback; candidate adds trusted parameter-correction and SQLite BUSY/LOCKED retry hints, preserving all budgets/guards/system prompts. max_output_repairs=1 in both.",
        "timeoutPolicy": "No true in-flight thread timeout injection or timeout-retry claim; transient fault is an immediate first-call SQLite OperationalError.",
        "poolingPolicy": "Micro average of recovered execution counts across format and tool suites, preserving both component denominators. Not estimated production fault frequencies."}
    manifest["source"] = freeze_source(it.PROJECT, root / "source")
    old_hashes = it.read(format_root / "source/source_manifest.json")
    compared, changed = [], []
    for rel in old_hashes:
        if rel.endswith(".py") and not rel.startswith(("backend/evaluation/", "backend/tests/")):
            if (root / "source" / rel).read_bytes() == (format_root / "source" / rel).read_bytes():
                compared.append(rel)
            else:
                assert rel == "backend/agent/llm_accident/runtime.py", rel
                changed.append(rel)
    manifest["productionFilesIdenticalToFormatSuite"] = compared
    manifest["productionChangedFromFormatSuite"] = changed + ["backend/agent/llm_accident/tool_recovery.py"]
    # Same driver and typed injected faults in both arms; only the runtime's
    # recovery-feedback branch differs. New helper is present but unused in A.
    import hashlib
    manifest["baselineSource"] = freeze_source(it.PROJECT, root / "baseline_source")
    runtime_rel = "backend/agent/llm_accident/runtime.py"
    (root / "baseline_source" / runtime_rel).write_bytes((format_root / "source" / runtime_rel).read_bytes())
    hashes = it.read(root / "baseline_source/source_manifest.json")
    hashes[runtime_rel] = hashlib.sha256((root / "baseline_source" / runtime_rel).read_bytes()).hexdigest()
    it.write(root / "baseline_source/source_manifest.json", hashes)
    manifest["baselineSource"]["sourceManifestHash"] = content_hash(hashes)
    manifest["formatCompatibilityPolicy"] = "Strict offline replay of all original 120 correction cassettes under candidate production code; request hashes and original outcomes must match before pooling. Not fresh model trials."
    manifest["manifestHash"] = content_hash(manifest)
    it.write(root / "manifest.json", manifest)
    paths = it.read(format_root / "source_store_before.json")["paths"]
    it.write(root / "source_store_before.json", {"paths": paths, "fingerprint": source_fingerprint(paths)})
    print("FROZEN " + str(root), flush=True)
    return root


def worker(request_path, spec_path):
    from unittest.mock import patch
    import sqlite3
    from backend.evaluation.v2 import worker as normal
    from backend.evaluation.v2 import checks as check_module
    from backend.agent.llm_accident.client import AccidentLLMClient
    from backend.agent.llm_accident.tools import EventQueries
    spec = it.read(spec_path)
    prefix, query_attempts = spec["prefix"], []
    class Client:
        def __init__(self):
            self.entries = []; self.offset = 0; self.fresh = 0; self.live = None
        async def complete(self, messages, tools, **kwargs):
            request = {"messages": deepcopy(messages), "tools": deepcopy(tools), **kwargs}
            if self.offset < len(prefix):
                entry = deepcopy(prefix[self.offset])
                assert content_hash(request) == entry["requestHash"], "tool_prefix_request_mismatch"
                self.entries.append(entry); self.offset += 1
                return deepcopy(entry["reply"])
            self.fresh += 1
            entry = {"requestHash": content_hash(request), "request": safe_value(request)}
            self.entries.append(entry)
            if self.live is None:
                self.live = AccidentLLMClient()
            try:
                reply = await self.live.complete(messages, tools, **kwargs)
                entry["reply"] = safe_value(reply)
                return reply
            except Exception as exc:
                entry["error"] = type(exc).__name__
                raise
        def assert_consumed(self):
            assert self.offset == len(prefix)
    original_query = EventQueries.execute
    target_count = 0
    def query(queries, name, params):
        nonlocal target_count
        query_attempts.append({"tool": name, "params": deepcopy(params), "injectedError": False})
        if name == spec["tool"]:
            target_count += 1
            if spec["kind"] == "persistent_query_error" or spec["kind"] == "transient_query_error" and target_count == 1:
                query_attempts[-1]["injectedError"] = True
                error = sqlite3.OperationalError("synthetic database lock")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
        return original_query(queries, name, params)
    # Keep the original assertion implementation accessible to assess(), even
    # while the worker resolves its imported evaluate() to this narrow wrapper.
    original_evaluate = check_module.evaluate
    def evaluate(case, record):
        with patch.object(check_module, "evaluate", original_evaluate):
            return assess(case, record, spec, query_attempts)
    client = Client()
    with patch.object(normal, "EvaluationClient", lambda *a, **k: client), patch.object(EventQueries, "execute", query), patch.object(check_module, "evaluate", evaluate):
        sys.argv = [sys.argv[0], request_path]
        normal.main()
    directory = Path(it.read(request_path)["directory"])
    record = it.read(directory / "record.json")
    record.update(transport="tool_fault_prefix_then_live", injectedToolFault=spec["kind"],
        toolFaultControl=spec["control"], freshEndToEndQualityEligible=False,
        historicalPrefixFullyMatched=client.offset == len(prefix), freshModelRequests=client.fresh,
        queryExecutionAttempts=query_attempts)
    it.write(directory / "record.json", record)


def trial(root, manifest, spec, batch, version="candidate"):
    import subprocess
    directory = root / ("baseline" if version == "baseline" else "controls" if spec["control"] else f"batch{batch}") / spec["key"]
    directory.mkdir(parents=True, exist_ok=False)
    request = deepcopy(spec["request"])
    request.update(directory=str(directory), transport="live")
    request.pop("cassette", None)
    it.write(directory / "request.json", request); it.write(directory / "fault_spec.json", spec)
    snapshot = manifest["baselineSource"] if version == "baseline" else manifest["source"]
    env = {**os.environ, "ACCIDENT_OUTPUT_REPAIR_ATTEMPTS": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPATH": snapshot["directory"]}
    with (directory / "worker.log").open("wb") as log:
        try:
            process = subprocess.run([sys.executable, "-m", "backend.evaluation.v2.tool_recovery_benchmark", "worker",
                str(directory / "request.json"), str(directory / "fault_spec.json")], cwd=snapshot["directory"],
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        except subprocess.TimeoutExpired:
            process = subprocess.CompletedProcess([], -1)
    if process.returncode:
        it.write(directory / "infrastructure_failure.json", {"exitCode": process.returncode})
    return {"key": spec["key"], "caseId": spec["caseId"], "kind": spec["kind"], "batch": batch,
        "control": spec["control"], "version": version, "directory": str(directory)}


def audit(root):
    from backend.evaluation.rag_ablation import source_fingerprint
    root = Path(root); manifest = it.read(root / "manifest.json")
    saved = deepcopy(manifest); stamp = saved.pop("manifestHash"); assert content_hash(saved) == stamp
    verify_source(manifest["source"])
    verify_source(manifest["baselineSource"])
    ah = it.read(Path(manifest["baselineSource"]["directory"]) / "source_manifest.json")
    bh = it.read(Path(manifest["source"]["directory"]) / "source_manifest.json")
    assert ah.keys() == bh.keys()
    assert [rel for rel in ah if ah[rel] != bh[rel]] == ["backend/agent/llm_accident/runtime.py"]
    before = it.read(root / "source_store_before.json")
    assert source_fingerprint(before["paths"]) == before["fingerprint"]
    rows = []
    for trial_row in it.read(root / "trials.json"):
        spec = next(c for c in manifest["cases"] if c["key"] == trial_row["key"])
        path = Path(trial_row["directory"]) / "record.json"
        if not path.exists():
            rows.append({**trial_row, "recovered": False, "failClosed": False, "infrastructureFailure": True}); continue
        record, cassette = it.read(path), it.read(path.parent / "cassette.json")
        trace = record["runtimeTrace"]
        assert record["historicalPrefixFullyMatched"] and cassette[:1] == spec["prefix"]
        assert len(cassette) == len(trace["llmCalls"]) <= 4
        assert trace["toolAttempts"] <= 4 and trace["toolRounds"] <= 3 and trace["outputRepairAttempts"] <= 1
        assert record["freshModelRequests"] == len(cassette) - 1
        assert all(content_hash(e["request"]) == e["requestHash"] for e in cassette)
        assert all(not c["toolsAllowed"] for c in trace["llmCalls"] if c["purpose"] == "output_repair")
        # Recompute adjusted assertions from raw execution data, not the stored grade.
        checks = assess(spec["request"]["case"], deepcopy(record), spec, record["queryExecutionAttempts"])
        assert record["checks"] == checks and record["passed"] == all(c["passed"] for c in checks)
        fresh = cassette[1:]
        fail_closed = bool(record.get("toolFaultObserved") and not record.get("actions") and not record.get("isolatedEffects") and record.get("workflowStatus") == "planning_blocked")
        rows.append({**trial_row, "recovered": is_recovered(record), "taskPassed": record["passed"],
            "faultObserved": record.get("toolFaultObserved"), "obtainedRequiredEvidence": record.get("toolRecoveryObtainedEvidence"),
            "failClosed": fail_closed, "failureReason": trace["failureReason"], "workflowStatus": record["workflowStatus"],
            "failedChecks": [c["name"] for c in checks if not c["passed"]],
            "freshModelRequests": len(fresh), "freshTokens": sum(e.get("reply", {}).get("usage", {}).get("total_tokens", 0) for e in fresh),
            "recordPath": str(path)})
    candidates = [r for r in rows if not r["control"] and r["version"] == "candidate"]
    baseline = [r for r in rows if r["version"] == "baseline"]
    controls = [r for r in rows if r["control"]]
    assert len(candidates) == 18 and len(baseline) == 6 and len(controls) == 3 and len({(r["version"], r["key"], r["batch"]) for r in rows}) == 27
    def stats(values):
        return {"trials": len(values), "recovered": sum(r["recovered"] for r in values), "recoveryRate": sum(r["recovered"] for r in values)/len(values)}
    report = {"scope": manifest["scope"], "candidate": stats(candidates), "baseline": stats(baseline),
        "byBatch": {str(b): stats([r for r in candidates if r["batch"] == b]) for b in (1, 2, 3)},
        "byFault": {k: stats([r for r in candidates if r["kind"] == k]) for k in FAULTS},
        "persistentControls": {"trials": 3, "failClosed": sum(r["failClosed"] for r in controls)},
        "freshModelRequests": sum(r.get("freshModelRequests", 0) for r in rows),
        "freshTokens": sum(r.get("freshTokens", 0) for r in rows),
        "infrastructureFailures": sum(bool(r.get("infrastructureFailure")) for r in rows),
        "sourceRagStoresUnchanged": True, "productionFilesSameAsFormatSuite": len(manifest["productionFilesIdenticalToFormatSuite"]), "rows": rows}
    it.write(root / "report.json", report)
    return report


def run(root):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from backend import config
    root = Path(root); manifest = it.read(root / "manifest.json")
    assert not (root / "state.json").exists(), "campaign_already_started_no_reruns"
    saved = deepcopy(manifest); stamp = saved.pop("manifestHash"); assert content_hash(saved) == stamp
    assert it.public_config()["agent"] == manifest["configuration"]
    verify_source(manifest["source"])
    it.write(root / "state.json", {"status": "running"}); trials = []
    try:
        for batch in (-1, 1, 2, 3, 0):
            specs = [c for c in manifest["cases"] if c["control"] == (batch == 0)]
            with ThreadPoolExecutor(max_workers=3) as pool:
                for future in as_completed([pool.submit(trial, root, manifest, c, max(0, batch), "baseline" if batch == -1 else "candidate") for c in specs]):
                    row = future.result(); trials.append(row); it.write(root / "trials.json", trials)
                    path = Path(row["directory"]) / "record.json"
                    record = it.read(path) if path.exists() else {}
                    print(f"version={row['version']} batch={batch} {row['key']} recovered={is_recovered(record)} {len(trials)}/27", flush=True)
        report = audit(root)
        it.write(root / "state.json", {"status": "complete", "result": report["candidate"]})
        print("COMPLETE " + json.dumps(report["candidate"]), flush=True)
    except Exception:
        it.write(root / "state.json", {"status": "failed", "completedTrials": len(trials)}); raise


def format_worker(request_path):
    from backend.evaluation.v2 import worker as normal
    sys.argv = [sys.argv[0], request_path]
    normal.main()


def replay_format(root):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import subprocess
    from backend import config
    root = Path(root); manifest = it.read(root / "manifest.json")
    assert it.read(root / "state.json")["status"] == "complete"
    directory = root / "format_compatibility"
    directory.mkdir(exist_ok=False)
    old_root = Path(manifest["formatSuite"])
    old_rows = [r for r in it.read(old_root / "report.json")["rows"] if r["arm"] == "candidate"]
    assert len(old_rows) == 120
    def one(row):
        original_dir = Path(row["directory"])
        target = directory / f"batch{row['batch']}" / row["key"]
        target.mkdir(parents=True)
        request = it.read(original_dir / "request.json")
        request.update(directory=str(target), transport="replay", cassette=str(original_dir / "cassette.json"))
        it.write(target / "request.json", request)
        env = {**os.environ, "ACCIDENT_OUTPUT_REPAIR_ATTEMPTS": "1", "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1", "PYTHONPATH": manifest["source"]["directory"]}
        with (target / "worker.log").open("wb") as log:
            process = subprocess.run([sys.executable, "-m", "backend.evaluation.v2.tool_recovery_benchmark", "format_worker",
                str(target / "request.json")], cwd=manifest["source"]["directory"], env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        assert process.returncode == 0, str(target)
        record, original = it.read(target / "record.json"), it.read(original_dir / "record.json")
        replayed = it.read(target / "cassette.json")
        assert replayed == it.read(original_dir / "cassette.json")
        assert record["passed"] == original["passed"]
        assert record["runtimeTrace"]["failureReason"] == original["runtimeTrace"]["failureReason"]
        assert record["runtimeTrace"]["outputRepairValidated"] == original["runtimeTrace"]["outputRepairValidated"]
        assert record["checks"] == original["checks"]
        return {"key": row["key"], "batch": row["batch"], "passed": record["passed"],
            "recovered": bool(record["passed"] and record["runtimeTrace"]["outputRepairValidated"] and record["runtimeTrace"]["status"] == "completed"),
            "requestHashesMatched": True, "freshModelRequests": 0, "originalRecord": str(original_dir / "record.json"),
            "recordPath": str(target / "record.json"), "cassetteHash": content_hash(replayed)}
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(one, r) for r in old_rows]):
            rows.append(future.result())
            it.write(directory / "progress.json", {"completed": len(rows), "rows": rows})
            if len(rows) % 10 == 0:
                print(f"format strict replay {len(rows)}/120", flush=True)
    result = {"status": "PASS", "trials": 120, "recovered": sum(r["recovered"] for r in rows),
        "allOutcomesAndRequestHashesUnchanged": True, "freshModelRequests": 0, "rows": rows}
    assert result["recovered"] == 114
    it.write(root / "format_compatibility.json", result)
    print("FORMAT_COMPATIBILITY_PASS 114/120, no new model calls", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "freeze": freeze(sys.argv[2])
    elif sys.argv[1] == "run": run(sys.argv[2])
    elif sys.argv[1] == "audit": print(json.dumps(audit(sys.argv[2])["candidate"]))
    elif sys.argv[1] == "format_replay": replay_format(sys.argv[2])
    elif sys.argv[1] == "format_worker": format_worker(sys.argv[2])
    else: worker(sys.argv[2], sys.argv[3])
