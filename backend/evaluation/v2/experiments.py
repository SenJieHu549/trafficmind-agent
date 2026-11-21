"""Small frozen-checkout experiment driver; serving state is never touched."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from backend.evaluation.v2.providers import content_hash
from backend.evaluation.v2.quality.judge import JudgeConfig, prompt
from backend.evaluation.v2.quality import rubric
from backend.evaluation.v2.runner import PROJECT

MODES = ("rule", "llm_context", "llm_tools")


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_source(project, target):
    project, target = Path(project), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for directory, children, names in os.walk(project / "backend"):
        children[:] = [n for n in children if n not in {".venv", "__pycache__", "data", ".pytest_cache"}]
        for name in names:
            source = Path(directory) / name
            if source.suffix not in {".py", ".json", ".yaml", ".yml", ".md", ".sql", ".toml"}: continue
            rel = source.relative_to(project)
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Read-only rule resources; do not clone any SQLite/vector stores or .env.
    rules = project / "backend/data/rules"
    if rules.exists():
        for source in rules.glob("*.md"):
            rel = source.relative_to(project)
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(target / "source_manifest.json", hashes)
    return {"directory": str(target.resolve()), "files": len(hashes), "sourceManifestHash": content_hash(hashes)}


def verify_source(snapshot):
    root = Path(snapshot["directory"])
    hashes = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if content_hash(hashes) != snapshot["sourceManifestHash"]: raise ValueError("snapshot_manifest_changed")
    for rel, expected in hashes.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected: raise ValueError("frozen_source_changed:" + rel)


def freeze_experiment(output):
    from backend import config  # Load configured credentials into this process, never persist them.
    if not config.LLM_ENABLED: raise ValueError("configured_model_required")
    root = Path(output) / ("phase3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    baseline = freeze_source(PROJECT, root / "baseline_source")
    datasets = {}
    for split in ("dev", "holdout"):
        source = PROJECT / f"backend/evaluation/v2/datasets/quality_{split}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        destination = root / f"{split}.json"
        write(destination, data)
        tool_cases = [c["caseId"] for c in data["cases"] if c["expected"].get("requiredTools") or c["expected"].get("requiredToolAttempts")]
        datasets[split] = {"path": str(destination.resolve()), "hash": content_hash(data),
            "caseIds": [c["caseId"] for c in data["cases"]], "toolSpecificCases": tool_cases,
            "commonCases": [c["caseId"] for c in data["cases"] if c["caseId"] not in tool_cases]}
    protocol = {"repetitionsPerCase": 1, "humanCalibration": "deferred_not_claimed",
        "arms": {"A": "rule", "B": "llm_context", "C": "llm_tools"},
        "candidateChangeScope": "LLM citation presentation and exact server-owned reference mapping only",
        "budgetsUnchanged": True, "allowFuzzyReferenceMatching": False,
        "qualityPolicy": "conditional means with coverage; compare paired scored cases and retain every fallback in task denominators",
        "claimScope": "synthetic small-set engineering pilot; no significance or general traffic accuracy claim",
        "holdoutPolicy": "freeze candidate first; run baseline ABC and candidate BC once; no tuning after inspecting holdout"}
    manifest = {"schemaVersion": "trafficmind-ablation-1", "root": str(root.resolve()), "baseline": baseline,
        "datasets": datasets, "protocol": protocol, "protocolHash": content_hash(protocol),
        "judgeConfig": JudgeConfig.from_env().public(), "judgePromptHash": content_hash(prompt({})),
        "rubricHash": content_hash([rubric.ANCHORS, rubric.POLICY]),
        "agentProvider": {"model": config.DEEPSEEK_MODEL, "temperature": .1, "maxTokens": 1800},
        "runs": {}}
    write(root / "experiment.json", manifest)
    print(json.dumps({"experiment": str(root.resolve()), "protocolHash": manifest["protocolHash"]}), flush=True)
    return root


def launch(root, version, split):
    from backend import config
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if JudgeConfig.from_env().public() != manifest["judgeConfig"]: raise ValueError("judge_configuration_changed")
    if content_hash(prompt({})) != manifest["judgePromptHash"]: raise ValueError("judge_prompt_changed")
    if config.DEEPSEEK_MODEL != manifest["agentProvider"]["model"]: raise ValueError("agent_model_changed")
    if version == "candidate" and "candidate" not in manifest: raise ValueError("candidate_not_frozen")
    if split == "holdout" and "candidate" not in manifest: raise ValueError("freeze_candidate_before_holdout")
    key = version + "_" + split
    if key in manifest["runs"]: raise ValueError("experiment_arm_already_run")
    snapshot = manifest[version]
    verify_source(snapshot)
    data = manifest["datasets"][split]
    if content_hash(json.loads(Path(data["path"]).read_text(encoding="utf-8"))) != data["hash"]:
        raise ValueError("frozen_dataset_changed")
    modes = MODES if version == "baseline" else MODES[1:]
    output = root / (key + "_runs")
    output.mkdir()
    command = [sys.executable, "-m", "backend.evaluation.v2.runner", "--dataset", data["path"],
               "--output", str(output.resolve()), "--transport", "live", "--modes", *modes]
    result = subprocess.run(command, cwd=snapshot["directory"])
    runs = list(output.glob("eval_*"))
    if len(runs) != 1: raise ValueError("experiment_run_not_found")
    run = runs[0]
    record = {"directory": str(run.resolve()), "exitCode": result.returncode, "modes": list(modes)}
    manifest["runs"][key] = record
    write(root / "experiment.json", manifest)
    # Grade all versions with the SAME frozen Judge and hard auditor.
    judge_output = root / (key + "_quality")
    judge_output.mkdir()
    result = subprocess.run([sys.executable, "-m", "backend.evaluation.v2.quality.runner", "--run-dir", str(run.resolve()),
        "--output", str(judge_output.resolve()), "--transport", "live", "--split", split], cwd=manifest["baseline"]["directory"])
    quality = list(judge_output.glob("quality_*"))
    if len(quality) != 1: raise ValueError("experiment_quality_report_not_found")
    record.update(qualityDirectory=str(quality[0].resolve()), judgeExitCode=result.returncode)
    verify_source(snapshot)
    write(root / "experiment.json", manifest)
    print(json.dumps({"arm": key, **record}), flush=True)


def freeze_candidate(root):
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if "candidate" in manifest: raise ValueError("candidate_already_frozen")
    if "baseline_dev" not in manifest["runs"] or "qualityDirectory" not in manifest["runs"]["baseline_dev"]:
        raise ValueError("finish_baseline_before_candidate")
    manifest["candidate"] = freeze_source(PROJECT, root / "candidate_source")
    manifest["candidateFrozenAt"] = datetime.now(timezone.utc).isoformat()
    write(root / "experiment.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "run", "freeze-candidate"])
    p.add_argument("--output", default="D:/TrafficMind-cache/phase3")
    p.add_argument("--experiment")
    p.add_argument("--version", choices=["baseline", "candidate"], default="baseline")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    args = p.parse_args()
    if args.action == "freeze": freeze_experiment(args.output)
    elif args.action == "freeze-candidate": freeze_candidate(args.experiment)
    else: launch(args.experiment, args.version, args.split)


if __name__ == "__main__": main()
