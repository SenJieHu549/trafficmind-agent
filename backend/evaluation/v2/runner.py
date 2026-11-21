"""CLI orchestrator. Each case runs serving entry points in a fresh child.

Default is OFFLINE scripted acceptance. Use --transport live explicitly for
real model calls; --transport replay strictly replays recorded model I/O.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from backend.evaluation.v2.providers import content_hash
from backend.evaluation.v2.checks import comparable

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT / "backend/evaluation_data/trafficmind_eval_v2_smoke.json"


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("schemaVersion") != "trafficmind-eval-v2": raise ValueError("unsupported_dataset")
    allowed_scenarios = {"normal", "approve_repeat", "reject", "facts_changed", "provider_failure"}
    seen = set()
    for case in data["cases"]:
        cid = case["caseId"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", cid) or cid in seen: raise ValueError("invalid_case_id")
        if case["scenario"] not in allowed_scenarios: raise ValueError("invalid_scenario")
        entry = case.get("entryPoint", "orchestrator")
        if entry not in {"orchestrator", "routed_api"}: raise ValueError("invalid_entry_point")
        if any(k in case for k in ("memoryItems", "reviewDecision", "roleFixtures", "selectedAgents")) and entry != "routed_api":
            raise ValueError("entry_coverage_requires_routed_api")
        if "reviewDecision" in case:
            policy = case["reviewDecision"]
            if not isinstance(policy, dict) or policy.get("action") not in {"wait", "select", "edit", "reject"}:
                raise ValueError("invalid_review_fixture")
            if policy["action"] == "select" and not policy.get("selectedAgent"): raise ValueError("invalid_review_fixture")
            if policy["action"] == "edit" and not policy.get("resolution"): raise ValueError("invalid_review_fixture")
        if any(item.get("fixtureScope", "current") not in {"current", "foreign_session", "foreign_thread"}
               for item in case.get("memoryItems", [])): raise ValueError("invalid_memory_fixture_scope")
        seen.add(cid)
    if not seen: raise ValueError("empty_dataset")
    return data


def source_fingerprint():
    digest = hashlib.sha256()
    paths = []
    for directory, children, filenames in os.walk(PROJECT / "backend"):
        children[:] = [name for name in children if name not in {".venv", "__pycache__"}]
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT).as_posix().encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def run_suite(dataset_path, output_root, *, modes=("llm_tools",), transport="scripted", case_ids=(), replay_from=None,
              compare_to=None, deadline=1200):
    if transport not in {"scripted", "live", "replay"}: raise ValueError("invalid_transport")
    if not modes or any(m not in {"rule", "llm_context", "llm_tools"} for m in modes): raise ValueError("invalid_mode")
    if transport == "replay" and not replay_from: raise ValueError("replay_from_required")
    dataset = load_dataset(dataset_path)
    cases = [c for c in dataset["cases"] if not case_ids or c["caseId"] in case_ids]
    if not cases or set(case_ids) - {c["caseId"] for c in cases}: raise ValueError("unknown_case_id")
    if transport == "live" and any(c.get("roleFixtures") or c.get("selectedAgents") for c in cases):
        raise ValueError("scripted_role_or_routing_fixtures_forbidden_in_live")
    evaluation_id = "eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = Path(output_root).resolve() / evaluation_id
    # New directory only: never mutate/reuse an old evaluation DB.
    root.mkdir(parents=True, exist_ok=False)
    (root / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    if transport == "replay":
        previous = json.loads((Path(replay_from) / "summary.json").read_text(encoding="utf-8"))
        if previous["datasetHash"] != content_hash(dataset): raise ValueError("replay_dataset_mismatch")
    if compare_to:
        previous = json.loads((Path(compare_to) / "summary.json").read_text(encoding="utf-8"))
        if previous["datasetHash"] != content_hash(dataset): raise ValueError("comparison_dataset_mismatch")
    manifest = {"evaluationRunId": evaluation_id, "schemaVersion": "trafficmind-eval-summary-v2",
        "datasetHash": content_hash(dataset), "sourceHash": source_fingerprint(), "transport": transport,
        "modes": list(modes), "purpose": dataset.get("purpose", "infrastructure_acceptance_not_quality_benchmark"),
        "datasetVersion": dataset.get("datasetVersion"), "split": dataset.get("split", "acceptance"),
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
        "profile": {"knowledge": "frozen_fixture_service", "workflow": "real_RunDriver_and_WorkflowExecutor",
                    "actions": "isolated_adapters", "fusion": "production_template_branch"}, "results": []}
    # Persist before starting, so a cancelled parent leaves a discoverable run.
    (root / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        for mode in modes:
            directory = root / f"{case['caseId']}-{mode}"
            directory.mkdir()
            request = {"evaluationRunId": evaluation_id, "dataset": dataset, "case": case, "mode": mode,
                       "transport": transport, "directory": str(directory)}
            if transport == "replay":
                request["cassette"] = str(Path(replay_from).resolve() / directory.name / "cassette.json")
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            print(f"[eval-v2] {case['caseId']} {mode} {transport}", flush=True)
            try:
                process = subprocess.run([sys.executable, "-m", "backend.evaluation.v2.worker", str(request_path)],
                    cwd=PROJECT, env=env, capture_output=True, timeout=deadline)
                # Child logs cannot include API credentials; worker stores only
                # exception types and never dumps config or raw provider errors.
                (directory / "worker.log").write_bytes(process.stdout + process.stderr)
                result_path = directory / "record.json"
                if process.returncode != 0 or not result_path.exists():
                    raise RuntimeError("worker_process_failed")
                record = json.loads(result_path.read_text(encoding="utf-8"))
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                record = {"caseId": case["caseId"], "modeRequested": mode, "passed": False,
                    "workflowStatus": "worker_failed", "error": {"type": type(exc).__name__},
                    "checks": [{"name": "worker_completed", "passed": False}]}
                (directory / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            item = {"caseId": case["caseId"], "mode": mode, "actualMode": record.get("actualMode"),
                "passed": record["passed"], "workflowStatus": record.get("workflowStatus"),
                "recordPath": f"{directory.name}/record.json", "degraded": record.get("degraded", False),
                "llmPathCompleted": record.get("llmPathCompleted", False),
                "injectedFault": record.get("injectedFault", False),
                "entryPoint": case.get("entryPoint", "orchestrator"),
                "collaborationStatus": record.get("collaboration", {}).get("status"),
                "humanReviewStatus": (record.get("humanReview", {}).get("after") or {}).get("status"),
                "failedChecks": [c["name"] for c in record.get("checks", []) if not c["passed"]]}
            reference = compare_to or (replay_from if transport == "replay" else None)
            if reference:
                prior = Path(reference) / directory.name / "record.json"
                item["behaviorMatchesReference"] = prior.exists() and comparable(record) == comparable(json.loads(prior.read_text(encoding="utf-8")))
            manifest["results"].append(item)
            manifest["total"] = len(manifest["results"])
            manifest["passed"] = sum(r["passed"] for r in manifest["results"])
            manifest["unexpectedFallbacks"] = sum(r["degraded"] and not r["injectedFault"] for r in manifest["results"])
            manifest["workflowCompleted"] = sum(r["workflowStatus"] == "completed" for r in manifest["results"])
            manifest["awaitingHumanReview"] = sum(r["workflowStatus"] == "awaiting_review" for r in manifest["results"])
            manifest["humanReviewRejected"] = sum(r["workflowStatus"] == "review_rejected" for r in manifest["results"])
            manifest["comparisonPassed"] = all(r.get("behaviorMatchesReference", True) for r in manifest["results"])
            (root / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(root), "passed": manifest["passed"], "total": manifest["total"],
                      "comparisonPassed": manifest["comparisonPassed"]}, ensure_ascii=False), flush=True)
    return root, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(PROJECT.parent / "TrafficMind-cache/evaluation-v2"))
    parser.add_argument("--modes", nargs="+", choices=["rule", "llm_context", "llm_tools"], default=["llm_tools"])
    parser.add_argument("--transport", choices=["scripted", "live", "replay"], default="scripted")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--replay-from")
    parser.add_argument("--compare-to")
    parser.add_argument("--deadline", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.deadline <= 3600: parser.error("deadline must be between 1 and 3600 seconds")
    _, summary = run_suite(args.dataset, args.output, modes=args.modes, transport=args.transport,
        case_ids=args.case, replay_from=args.replay_from, compare_to=args.compare_to, deadline=args.deadline)
    raise SystemExit(0 if summary["passed"] == summary["total"] and summary["comparisonPassed"] else 1)


if __name__ == "__main__": main()
