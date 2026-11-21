"""Read recorded executions, grade without re-executing Agent or actions."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from backend.evaluation.v2.providers import content_hash
from backend.evaluation.v2.quality.inputs import build_payload, hard_audit
from backend.evaluation.v2.quality.judge import Judge
from backend.evaluation.v2.quality.report import aggregate


async def assess_runs(run_dirs, output, *, transport="scripted", split="dev", repeat=1, case_ids=(),
                      include_synthetic=False, replay_from=None, judge=None):
    if not 1 <= repeat <= 5: raise ValueError("repeat_out_of_range")
    if transport == "replay" and replay_from is None: raise ValueError("judge_replay_from_required")
    judge = judge or Judge()
    root = Path(output) / ("quality_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    results, excluded, seen_records = [], [], set()
    for source in map(Path, run_dirs):
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        dataset = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
        cases = {c["caseId"]: c for c in dataset["cases"]}
        for execution in summary["results"]:
            case = cases[execution["caseId"]]
            case_split = case.get("split", dataset.get("split", "acceptance"))
            if case_ids and case["caseId"] not in case_ids: continue
            if split != "all" and case_split != split: continue
            path = (source / execution["recordPath"]).resolve()
            if not path.is_relative_to(source.resolve()): raise ValueError("record_path_outside_run")
            record = json.loads(path.read_text(encoding="utf-8"))
            record_hash = content_hash(record)
            if record_hash in seen_records:
                excluded.append({"caseId": case["caseId"], "reason": "duplicate_record", "path": str(path)})
                continue
            seen_records.add(record_hash)
            if record.get("transport") != "live" and not include_synthetic:
                excluded.append({"caseId": case["caseId"], "reason": "non_live_generation", "path": str(path)})
                continue
            for ordinal in range(repeat):
                aid = record_hash[:20] + f"_r{ordinal + 1}"
                item = {"assessmentId": aid, "recordPath": str(path), "recordHash": record_hash,
                    "caseId": case["caseId"], "split": case_split, "mode": record.get("actualMode"),
                    "requestedMode": record.get("modeRequested"),
                    "scenario": case["scenario"], "generationTransport": record.get("transport"),
                    "workflowStatus": record.get("workflowStatus"),
                    "isException": case["scenario"] in {"reject", "facts_changed", "provider_failure"} or bool(case.get("toolFault")),
                    "judgeTransport": transport, "degraded": bool(record.get("degraded")),
                    "fallbackReason": record.get("runtimeTrace", {}).get("fallbackReason"),
                    "primaryEligible": record.get("transport") == "live" and transport == "live" and not record.get("degraded")}
                try:
                    checks = hard_audit(record, case)
                    payload = build_payload(record, case)
                    previous = None
                    if transport == "replay":
                        previous = json.loads((Path(replay_from) / (aid + ".json")).read_text(encoding="utf-8"))["assessment"]
                    assessment = await judge.grade(payload, replay=previous, scripted=transport == "scripted")
                    item.update(hardChecks=checks, hardPassed=all(c["passed"] for c in checks),
                        payload=payload, assessment=assessment,
                        toolAttempts=len(record.get("runtimeTrace", {}).get("toolCalls", [])),
                        legalToolRequests=sum(c["passed"] for c in checks if c["name"].startswith("tool_schema:")))
                except Exception as exc:
                    item.update(hardChecks=[{"name": "quality_input_valid", "passed": False}], hardPassed=False,
                        assessment={"status": "error", "verdict": None, "errorType": type(exc).__name__})
                item["accepted"] = item["hardPassed"] and item["assessment"]["status"] == "scored" and not (
                    item["assessment"].get("verdict") or {}).get("unableToJudge", True)
                (root / (aid + ".json")).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(item)
                report = {"schemaVersion": "trafficmind-quality-report-1", "results": results,
                          "excluded": excluded, **aggregate(results)}
                (root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[quality] {case['caseId']} repeat={ordinal + 1} {item['assessment']['status']} hard={item['hardPassed']}", flush=True)
    if not results:
        (root / "quality_report.json").write_text(json.dumps({"results": [], "excluded": excluded, "groups": []}, indent=2), encoding="utf-8")
        raise ValueError("no_records_selected_for_quality")
    return root, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", nargs="+", required=True)
    p.add_argument("--output", default=str(Path(__file__).resolve().parents[5] / "TrafficMind-cache/evaluation-quality"))
    p.add_argument("--transport", choices=["scripted", "live", "replay"], default="scripted")
    p.add_argument("--split", choices=["dev", "holdout", "acceptance", "all"], default="dev")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--include-synthetic", action="store_true")
    p.add_argument("--replay-from")
    args = p.parse_args()
    root, report = asyncio.run(assess_runs(args.run_dir, args.output, transport=args.transport, split=args.split,
        repeat=args.repeat, case_ids=args.case, include_synthetic=args.include_synthetic, replay_from=args.replay_from))
    print(json.dumps({"directory": str(root), "ratings": len(report["results"]),
                      "humanCalibration": "pending_independent_human_labels"}))
    raise SystemExit(1 if any(r["assessment"]["status"] == "error" for r in report["results"]) else 0)


if __name__ == "__main__": main()
