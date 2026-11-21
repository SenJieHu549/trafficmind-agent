"""Run the opt-in eight-case business baseline and retain real failures.

This runner never changes production code, invokes a model, or executes actions.
Exit 1 means business acceptance failures; exit 2 means invalid/incomplete setup.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SUITE = "backend/tests/validation_human_review_business_quality.py"
TITLES = {
    "V01": "选择原方案、隔离被替代建议（融合及快照）",
    "V02": "人工修改进入实际 Planning 输入",
    "V03": "未受冲突影响的急救通道要求保留",
    "V04": "伤情未知及核实要求保留",
    "V05": "违反冻结明确约束时不能完成或复用",
    "V06": "未经核实的新事实不能直接完成或复用",
    "V07": "证据身份可解析、来源变化后阻断",
    "V08": "Planning 各输入通道一致且保留有效要求",
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts" / "evaluation" / f"human_review_business_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True)
    source_files = [SUITE, "scripts/run_human_review_business_validation.py",
        "backend/agent/collaboration/human_review.py", "backend/agent/collaboration/orchestrator.py",
        "backend/planning/analysis_handoff.py", "backend/planning/agent_planning_adapter.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files}
    manifest = {"createdAt": stamp, "cases": TITLES, "sourceSha256": hashes,
        "mode": "scripted_roles_real_orchestrator_api_sqlite_handoff_planning_adapter",
        "newModelCalls": 0, "externalActionExecutions": 0,
        "scope": "业务接受标准基线；不是模型准确率或线上成功率",
        "semanticScope": "固定中文输入人工定义预期；结构、身份和引用检查，不声称理解任意自然语言",
        "factGateExpectation": "V06 是待验证的更严格业务要求；当前接口未承诺事实抽取/核实功能"}
    dump(output / "manifest.json", manifest)
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "DEEPSEEK_API_KEY": "",
                   "HUMAN_REVIEW_VALIDATION_DIR": str(output)}
    command = [sys.executable, "-m", "pytest", SUITE, "-q", "--disable-warnings", "--tb=short",
               f"--junitxml={output / 'junit.xml'}"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8")
    (output / "pytest.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    rows = []
    if (output / "junit.xml").exists():
        for test in ET.parse(output / "junit.xml").iter("testcase"):
            case_id = test.attrib["name"].split('_')[1].upper()
            error, failure, skipped = test.find("error"), test.find("failure"), test.find("skipped")
            observed_path = output / f"{case_id}_observed.json"
            observed = json.loads(observed_path.read_text(encoding="utf-8")) if observed_path.exists() else {}
            # Fixture/schema/unexpected exceptions must not count as a business
            # acceptance failure merely because pytest labels call errors fail.
            acceptance_failure = bool(failure is not None and observed.get("checks")
                                      and any(not c["passed"] for c in observed["checks"]))
            outcome = ("error" if error is not None or (failure is not None and not acceptance_failure)
                       else "failed" if acceptance_failure else "skipped" if skipped is not None else "passed")
            rows.append({"caseId": case_id, "title": TITLES.get(case_id), "outcome": outcome,
                "durationSeconds": float(test.attrib.get("time", 0)),
                "message": (error if error is not None else failure).attrib.get("message", "")
                    if error is not None or failure is not None else "",
                "checks": observed.get("checks", []), "observedFile": observed_path.name if observed else None})
    counts = {name: sum(r["outcome"] == name for r in rows) for name in ("passed", "failed", "error", "skipped")}
    unchanged = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    report = {"manifest": "manifest.json", "pytestExitCode": completed.returncode,
        "counts": counts, "sourceUnchangedDuringRun": unchanged,
        "complete": len(rows) == 8 and counts["error"] == 0 and counts["skipped"] == 0,
        "results": rows}
    dump(output / "report.json", report)
    lines = ["# 人工裁决后的业务质量验证基线", "", f"模式：{manifest['mode']}", "",
             "这里只运行固定业务接受标准；失败原样保留，运行器不修改业务代码。", "",
             f"结果：{counts['passed']} 通过，{counts['failed']} 失败，{counts['error']} 错误，{counts['skipped']} 跳过。", "",
             "| 场景 | 结果 | 未满足检查 |", "|---|---|---|"]
    for row in rows:
        missing = "；".join(c["id"] for c in row["checks"] if not c["passed"])
        lines.append(f"| {row['caseId']}：{row['title']} | {row['outcome']} | {missing or '-'} |")
    lines += ["", "V01 检查融合/快照中的当前推荐；V08 进一步检查 Planning 并行通道，二者不能互相替代。",
              "V06 分别检查事件未被覆盖和新事实是否被阻断；不能因为前者通过就推断后者通过。",
              "V07 仅证明引用身份/适用性检查，不能代替每个自然语言结论的语义支持审查。",
              "", "完整输入、响应、最终结论和 Planning 请求见各 Vxx_observed.json；错误详情见 pytest.log 和 junit.xml。"]
    (output / "report.md").write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print(json.dumps({"output": str(output), **counts, "complete": report["complete"],
                      "sourceUnchanged": unchanged}, ensure_ascii=False))
    return 2 if not report["complete"] or not unchanged else 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
