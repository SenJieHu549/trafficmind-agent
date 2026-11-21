"""Export blind human packets; compare only independently completed labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from backend.evaluation.v2.quality.rubric import DIMENSIONS, VERSION, ANCHORS, POLICY
from backend.evaluation.v2.providers import content_hash

THRESHOLDS = {"minimumHumanCases": 10, "maximumMAE": 1.0, "minimumWithinOneAgreement": .8}


def quadratic_kappa(pairs):
    if not pairs: return None
    n = len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum((a-b)**2 for a,b in pairs) / n
    expected = sum(left[a] * right[b] * (a-b)**2 for a in left for b in right) / n**2
    return round(1-observed/expected, 4) if expected else None


def export_packet(report_path, packet_path, labels_path, *, limit=12):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    # Exclude holdout from human-guided development calibration.
    unique = {r["recordHash"]: r for r in report["results"] if r["split"] == "dev" and r.get("payload")
              and r.get("primaryEligible") and r["assessment"]["status"] == "scored"
              and not r["assessment"]["verdict"]["unableToJudge"]}
    ordered = sorted(unique.values(), key=lambda r: r["caseId"])
    if len(ordered) > limit:
        ordered = [ordered[round(i * (len(ordered)-1) / (limit-1))] for i in range(limit)] if limit > 1 else ordered[:1]
    labels = {"schemaVersion": "trafficmind-human-labels-1", "origin": "human", "reviewer": "",
        "independent": False, "rubricVersion": VERSION, "rubricHash": content_hash([ANCHORS, POLICY]), "labels": []}
    lines = ["# 26 Judge 人工盲评样本与评分表", "", "这些样本未附 Judge 分数、被测模式及实现版本。请先独立评分，再进行对照。",
             "", "每维为 1～5 整数：研判合理性、证据支持性、处置可行性。未知信息正确保留不应直接扣分；具体锚点见第 25 号文档及 rubric.py。",
             "", "在同目录 JSON 评分表填写 reviewer，确认独立评分后将 independent 设为 true，再填写每条 scores 和 notes。尚未填写的分数保持 null，不要填 0。", ""]
    for index, row in enumerate(ordered, 1):
        payload = row["payload"]
        labels["labels"].append({"sampleId": f"B{index:02d}", "inputHash": content_hash(payload),
            "scores": {d: None for d in DIMENSIONS}, "notes": ""})
        lines.extend([f"## B{index:02d}", "", "### 任务及事实", "", payload["task"], "", "```json",
            json.dumps({"facts": payload["facts"], "known": payload.get("referenceFacts"), "unknown": payload.get("unknownFacts")}, ensure_ascii=False, indent=2),
            "```", "", "### 实际可用证据", "", "```json", json.dumps(payload["evidence"], ensure_ascii=False, indent=2),
            "```", "", "### 待评输出", "", "```json", json.dumps(payload["candidate"], ensure_ascii=False, indent=2), "```", ""])
    Path(packet_path).write_text("\n".join(lines), encoding="utf-8")
    Path(labels_path).write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(ordered)


def compare_labels(labels, results):
    if labels.get("rubricVersion") != VERSION or labels.get("rubricHash") != content_hash([ANCHORS, POLICY]):
        raise ValueError("human_rubric_mismatch")
    if labels.get("origin") != "human" or not labels.get("reviewer") or labels.get("independent") is not True:
        return {"status": "pending_independent_human_labels", "thresholds": THRESHOLDS}
    ratings = {}
    for row in results:
        verdict = row["assessment"].get("verdict")
        if row["split"] == "dev" and row.get("primaryEligible") and verdict and not verdict["unableToJudge"]:
            ratings.setdefault(row["assessment"]["inputHash"], []).append(verdict)
    pairs = {d: [] for d in DIMENSIONS}
    completed = 0
    disagreements = []
    seen = set()
    for label in labels["labels"]:
        identity = label["inputHash"]
        if identity in seen: raise ValueError("duplicate_human_label")
        seen.add(identity)
        scores = label["scores"]
        if all(v is None for v in scores.values()): continue
        if set(scores) != set(DIMENSIONS) or any(type(v) is not int or not 1 <= v <= 5 for v in scores.values()):
            raise ValueError("invalid_human_scores")
        if identity not in ratings: continue
        completed += 1
        # First predeclared rating per input, not whichever repeat fits labels.
        verdict = ratings[identity][0]
        for d in DIMENSIONS:
            actual = verdict[d]["score"]
            pairs[d].append((scores[d], actual))
            if abs(scores[d] - actual) >= 2:
                disagreements.append({"inputHash": identity, "dimension": d,
                    "humanScore": scores[d], "judgeScore": actual, "humanNotes": label.get("notes", "")})
    metrics = {}
    for d, values in pairs.items():
        metrics[d] = {"MAE": round(mean(abs(a-b) for a,b in values), 4) if values else None,
            "exactAgreement": round(mean(a == b for a,b in values), 4) if values else None,
            "withinOneAgreement": round(mean(abs(a-b) <= 1 for a,b in values), 4) if values else None,
            "quadraticWeightedKappa": quadratic_kappa(values)}
    enough = completed >= THRESHOLDS["minimumHumanCases"]
    passed = enough and all(m["MAE"] <= THRESHOLDS["maximumMAE"] and
                           m["withinOneAgreement"] >= THRESHOLDS["minimumWithinOneAgreement"] for m in metrics.values())
    return {"status": "passed" if passed else "needs_review" if enough else "insufficient_human_pairs",
            "pairedCases": completed, "dimensions": metrics, "thresholds": THRESHOLDS,
            "disagreements": disagreements,
            "note": "Engineering pilot thresholds, not a certification of domain correctness."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["export", "compare"])
    p.add_argument("--report", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--packet")
    p.add_argument("--output")
    args = p.parse_args()
    if args.action == "export":
        if not args.packet: p.error("export requires --packet")
        # Never overwrite a user's already entered labels.
        if Path(args.labels).exists() or Path(args.packet).exists(): raise ValueError("human_packet_already_exists")
        print(json.dumps({"exported": export_packet(args.report, args.packet, args.labels)}))
    else:
        result = compare_labels(json.loads(Path(args.labels).read_text(encoding="utf-8")),
            json.loads(Path(args.report).read_text(encoding="utf-8"))["results"])
        if args.output: Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
