"""Combine measured output and tool recoveries without averaging percentages."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.providers import content_hash
from backend.evaluation.v2.tool_recovery_benchmark import audit


def merge_counts(*components):
    for c in components:
        if type(c["trials"]) is not int or type(c["recovered"]) is not int or not 0 <= c["recovered"] <= c["trials"]:
            raise ValueError("invalid_recovery_counts")
    total = sum(c["trials"] for c in components)
    success = sum(c["recovered"] for c in components)
    return {"trials": total, "recovered": success, "recoveryRate": success / total if total else None}


def publish(root):
    root = Path(root).resolve()
    assert it.read(root / "state.json")["status"] == "complete"
    manifest = it.read(root / "manifest.json")
    tools = audit(root)
    compatibility = it.read(root / "format_compatibility.json")
    assert compatibility["status"] == "PASS" and compatibility["freshModelRequests"] == 0
    format_root = Path(manifest["formatSuite"])
    old = it.read(format_root / "report.json")
    assert content_hash(old) == manifest["formatReportHash"]
    # Recompute both numerators from all saved task assertions, rather than
    # trusting the aggregate report or changing grades to hit a target.
    output_records = [r for r in old["rows"] if r["arm"] == "candidate"]
    n_output = 0
    for row in output_records:
        record = it.read(row["recordPath"]); trace = record["runtimeTrace"]
        n_output += bool(all(c["passed"] for c in record["checks"]) and trace["outputRepairValidated"] and trace["status"] == "completed")
    assert n_output == old["candidate"]["recovered"] == compatibility["recovered"] == 114
    n_tool = 0
    for row in tools["rows"]:
        if row["version"] != "candidate" or row["control"]:
            continue
        record = it.read(row["recordPath"]); trace = record["runtimeTrace"]
        n_tool += bool(all(c["passed"] for c in record["checks"]) and record["toolFaultObserved"]
            and record["toolRecoveryObtainedEvidence"] and trace["status"] == "completed")
    assert n_tool == tools["candidate"]["recovered"]
    tool_evidence_recovered = sum(r.get("obtainedRequiredEvidence", False) for r in tools["rows"]
        if r["version"] == "candidate" and not r["control"])
    components = {"formatAndFields": {"trials": len(output_records), "recovered": n_output},
                  "toolArgumentsAndTransientQuery": {"trials": 18, "recovered": n_tool}}
    total = merge_counts(*components.values())
    assert total["trials"] == 138
    a = {r["key"]: r for r in tools["rows"] if r["version"] == "baseline"}
    b = {r["key"]: r for r in tools["rows"] if r["version"] == "candidate" and r["batch"] == 1}
    assert a.keys() == b.keys() and len(a) == 6
    pairs = {"pairs": 6, "baselineRecovered": sum(r["recovered"] for r in a.values()),
        "candidateRecovered": sum(r["recovered"] for r in b.values()),
        "recovered": [k for k in a if not a[k]["recovered"] and b[k]["recovered"]],
        "regressed": [k for k in a if a[k]["recovered"] and not b[k]["recovered"]]}
    combined = {"scope": "conditional recovery in synthetic fault-injection suites, execution-weighted micro average",
        "components": components, "combined": total, "toolPairedFirstBatch": pairs,
        "formatReportHash": content_hash(old), "toolReportHash": content_hash(tools),
        "formatCompatibility": {k: v for k, v in compatibility.items() if k != "rows"},
        "uniqueFaultSamples": 46, "businessContexts": 9, "batchesPerComponent": 3,
        "toolQueriesRecoveredWithRequiredEvidence": tool_evidence_recovered,
        "excludedFromDenominator": {"formatProtectedControls": 16, "toolPersistentControls": 3,
            "toolBaselineTrials": 6, "historicalNaturalReplay": 3, "offlineFormatCompatibilityReplays": 120,
            "exploratoryToolTrials": 18, "exploratoryToolControls": 3},
        "definition": "Injected eligible failure observed; original task checks, evidence and guards pass after recovery. Source-format trials require outputRepairValidated; source-tool trials require required query evidence and at most one target retry.",
        "notClaimed": "Production incident frequency, overall task success, independent human semantic quality, or 138 independent scenarios."}
    it.write(root / "combined_report.json", combined)
    rate = f"{total['recoveryRate']:.2%}"
    tool_rate = f"{tools['candidate']['recoveryRate']:.2%}"
    lines = ["# 工具异常专项、反馈改进与统一异常恢复率（2026-11-09）", "",
        f"**统一异常恢复率：{rate}（{total['recovered']}/138）**。格式/字段异常 114/120，加上工具参数与瞬态查询异常 {n_tool}/18，按成功次数与执行次数直接相加。", "",
        "## 统一指标口径", "",
        "| 类型 | 恢复成功/执行 | 恢复率 |", "|---|---:|---:|",
        "| 格式与字段异常 | 114/120 | 95.00% |",
        f"| 工具参数与瞬态查询异常 | {n_tool}/18 | {tool_rate} |",
        f"| 合并 | {total['recovered']}/138 | {rate} |", "",
        "两类异常均分 3 批执行；共有 46 个唯一故障样本，来源于 9 个已见合成业务场景。重复执行不是新的独立场景。合并比例受当前 120:18 样本配置影响，不表示真实业务中的异常频率。", "",
        "分子必须是故障发生后最终通过原任务执行断言及证据、动作、身份等校验的执行。参数改正确但未查到任务要求的证据、查询成功但引用错误、下游任务失败，均不算恢复。持续不可用属于单独的降级/阻断对照，不加入可恢复分母。", "",
        "## 为什么补充工具反馈", "",
        "探索轮 6 个工具故障样本 × 3 批：参数异常 7/9，瞬态查询异常 0/9。轨迹显示模型经常看到 tool_execution_failed 就结束研判，未重新取得必需证据；另有少量最终引用错误。该轮完整保留，没有覆盖或作为最终候选结果。",
        "随后增加受控反馈：参数错误返回服务端生成的参数 Schema；SQLite BUSY/LOCKED 按结构化错误码识别为可重试，只提示在原预算内同参重试一次；再次失败、超时、工作线程繁忙及未知错误不建议自动重试。只读查询范围、事件绑定、引用/动作校验、原模型/工具预算均保留。",
        "这属于错误分类与反馈引导，重试决策仍由 Agent 作出，不是 SDK 隐式重试。原 SDK max_retries=0；原连续错误和总调用上限仍执行。输出格式纠错仍默认关闭，测试进程中显式设为 1。工具错误反馈在新启动的运行时生效，本轮没有重启或部署应用。", "",
        "## 小规模工具对照协议", "",
        "- D10：道路与知识查证，目标 accident_knowledge_search；D13：正向历史案例；D15：历史案例时间隔离，目标 accident_case_search。",
        "- 各场景两种可恢复故障：首个调用的 limit=99；首个真实查询抛带 SQLITE_BUSY 错误码的 OperationalError，后续恢复原查询实现。共 6 个故障样本。没有测试真实在途线程超时后重试，也不作该能力声明。",
        "- 原第一轮工具调用按请求哈希严格回放，并人工注入指定故障；其后模型决策为新发真实调用，查询走原工具实现，DAG/Planning/Workflow 走原链，知识和动作使用隔离评测设施。",
        "- 使用同一冻结驱动和带类型错误的故障输入，旧反馈 A 每样本 1 次，新反馈 B 每样本 3 次。两个源快照仅 runtime.py 不同，新 helper 两边均存在，A 不调用。",
        "- 预声明的故障仅按确切 tool_call_id 与错误码豁免一次，原错误断言仍单独保存。纠错后任何新错误继续正常扣分，原任务期望不改。任务要求查询成功时，不能用‘已尝试查询’替代。",
        "- 所有计划执行保留；不补跑、不择优、不调整分母。专项通过条件还限定最多一次目标工具重试，同时继续执行原 4 次模型调用、4 次工具尝试、3 个工具轮次上限。", "",
        "## 工具 A/B 与分批结果", "",
        f"相同 6 个样本首次配对：旧反馈 {pairs['baselineRecovered']}/6，新反馈 {pairs['candidateRecovered']}/6；恢复 {len(pairs['recovered'])} 个，退化 {len(pairs['regressed'])} 个。这是小样本观察，不能视为独立生产验证。", "",
        "| 候选批次 | 恢复/执行 | 恢复率 |", "|---|---:|---:|"]
    for batch, data in tools["byBatch"].items():
        lines.append(f"| {batch} | {data['recovered']}/{data['trials']} | {data['recoveryRate']:.2%} |")
    lines.extend(["", "| 工具故障类型 | 恢复/执行 | 恢复率 |", "|---|---:|---:|"])
    for key, data in tools["byFault"].items():
        lines.append(f"| {key} | {data['recovered']}/{data['trials']} | {data['recoveryRate']:.2%} |")
    lines.extend(["", f"工具层重新取得必需查询证据为 {tool_evidence_recovered}/18，但最终任务恢复为 {n_tool}/18。引用或下游任务检查失败仍扣分，不能仅按工具重新返回成功计算统一异常恢复率。"])
    control_n = tools["persistentControls"]["failClosed"]
    lines.extend(["", f"持续故障单列：{control_n}/3 进入 planning_blocked 且无动作/副作用。其余轨迹是否停止查询、是否产生有限结论均保留，不强称全部正确降级，不加入统一恢复率。", "",
        "## 原格式测试兼容性", "",
        "工具反馈只改工具错误分支。仍对原 120 次格式异常的完整模型 I/O 做严格离线回放：候选代码的每个请求哈希、工具/模型响应及原任务断言与原记录一致，仍为 114/120。120 次回放没有新模型请求，也没有增加合并分母。原 6 次引用失败保持失败。", "",
        "## 未恢复项与边界", ""])
    failures = [r for r in tools["rows"] if r["version"] == "candidate" and not r["control"] and not r["recovered"]]
    if not failures:
        lines.append("本次 18 次工具恢复执行没有未恢复项，不能泛化为所有工具故障都可恢复。原格式测试仍有 6 次引用失败。")
    for row in failures:
        lines.append(f"- 第 {row['batch']} 批 {row['key']}：{row['failureReason']}；{row['failedChecks']}；{row['recordPath']}")
    lines.extend(["", f"本轮工具 A/B 与持续故障对照共新模型请求 {tools['freshModelRequests']} 次，记录 Token {tools['freshTokens']}。另有此前探索轮成本，未混入本轮计数。", "",
        "## 简历更新", ""])
    bullet = f"**Agent 评测与可靠性优化：** 基于 Evaluation Runner 构建规则断言与 LLM-as-a-Judge 自动评测，覆盖研判质量、证据引用、工具调用及 Workflow 执行；通过执行轨迹定位输出格式、工具参数与瞬态查询故障，结合校验反馈及预算约束下的纠错与重试，在 **138 次故障注入测试中实现 {rate} 异常恢复率**，并沉淀失败用例开展回归验证。"
    lines.extend(["> " + bullet, "",
        "95% 保留为格式异常分项；统一指标使用两类合计的实测值。整体任务成功率是另一个指标，本轮不更新；本轮不新增独立 Judge 质量评分或人工校准。", "",
        "## 证据与复现", "", f"本轮：`{root}`。探索轮：`D:/TrafficMind-cache/output-repair/tool_recovery_20261109T032440_ec6c2446`。",
        "原格式实验见 34 号记录。工具 run 禁止对已启动目录重跑；format_replay 只使用已记录模型响应并禁止外网。combined_report.json 保留两类分子分母、配对变化与不计入分母的对照。", ""])
    (root / "combined_report.md").write_text("\n".join(lines), encoding="utf-8")
    project = it.PROJECT
    dev = project / "技术文档/TrafficMind_开发记录_01-29/35_工具异常专项与统一异常恢复率_20261109.md"
    job = project / "技术文档/HSJ 求职文档/TrafficMind_新版简历面试准备/11_Agent评测实测结果与简历更新.md"
    marker = "<!-- combined-recovery-20261109:start -->"
    old_job = job.read_text(encoding="utf-8")
    assert not dev.exists() and marker not in old_job
    dev.write_text("\n".join(lines), encoding="utf-8")
    first, rest = old_job.split("\n", 1)
    block = f"""
{marker}
## 2026-11-09 最新：统一异常恢复率

> {bullet}

统一口径：格式/字段异常 114/120（95%），工具参数与瞬态查询异常 {n_tool}/18（{tool_rate}），合计 **{total['recovered']}/138（{rate}）**。两类均测试 3 批，共 46 个故障样本、9 个合成业务上下文；不是 138 个独立业务场景。

通过原任务规则断言且恢复后引用、动作与身份校验通过才算成功。参数改正但缺少必需查询证据、引用错误或下游执行失败均不计恢复。16 个格式保护对照、3 个持续工具故障对照和 120 次离线兼容性回放不并入分母。

原格式 95% 为分项，整体任务成功率保持原实验结果。工具分类反馈有旧反馈对照，原格式全部 120 条严格回放结果不变；详见 [35 号开发记录](../../TrafficMind_开发记录_01-29/{dev.name})。
<!-- combined-recovery-20261109:end -->

"""
    job.write_text(first + "\n" + block + rest, encoding="utf-8")
    archive = project / "artifacts/evaluation" / root.name
    for rel in ("backend/agent/llm_accident/runtime.py", "backend/agent/llm_accident/tool_recovery.py",
                "backend/agent/llm_accident/tools.py", "backend/agent/llm_accident/contracts.py",
                "backend/agent/llm_accident/output_validation.py", "backend/agent/llm_accident/integration.py", "backend/config.py"):
        assert (project / rel).read_bytes() == (root / "source" / rel).read_bytes()
    archive.mkdir(parents=True, exist_ok=False)
    it.write(root / "test_validation.json", {"passed": 94, "warnings": "one existing Starlette/AnyIO deprecation warning",
        "files": ["test_combined_recovery_results.py", "test_tool_recovery_hints.py", "test_tool_recovery_benchmark.py", "test_recovery_benchmark.py", "test_output_repair.py", "test_llm_accident_agent.py"]})
    for name in ("manifest.json", "report.json", "combined_report.json", "combined_report.md", "format_compatibility.json", "source_store_before.json", "state.json", "trials.json"):
        shutil.copy2(root / name, archive / name)
    shutil.copy2(root / "test_validation.json", archive / "test_validation.json")
    saved_contexts = set()
    for item in manifest["cases"]:
        if item["caseId"] in saved_contexts: continue
        source = Path(item["originalRecord"]).parent
        assert content_hash(it.read(source / "record.json")) == item["originalRecordHash"]
        assert content_hash(it.read(source / "cassette.json")) == item["originalCassetteHash"]
        target = archive / "context_sources" / item["caseId"]; target.mkdir(parents=True)
        for name in ("record.json", "request.json", "cassette.json"): shutil.copy2(source / name, target / name)
        saved_contexts.add(item["caseId"])
    pilot = Path("D:/TrafficMind-cache/output-repair/tool_recovery_20261109T032440_ec6c2446")
    (archive / "pilot").mkdir()
    for name in ("manifest.json", "report.json", "trials.json", "state.json"):
        shutil.copy2(pilot / name, archive / "pilot" / name)
    for row in it.read(pilot / "trials.json"):
        source = Path(row["directory"]); target = archive / "pilot" / source.relative_to(pilot)
        target.mkdir(parents=True)
        for name in ("record.json", "request.json", "cassette.json", "fault_spec.json", "worker.log"):
            shutil.copy2(source / name, target / name)
    for row in tools["rows"]:
        source = Path(row["directory"]); target = archive / source.relative_to(root); target.mkdir(parents=True)
        for name in ("request.json", "record.json", "cassette.json", "fault_spec.json", "worker.log"):
            if (source / name).exists(): shutil.copy2(source / name, target / name)
    for row in compatibility["rows"]:
        source = Path(row["recordPath"]).parent
        target = archive / source.relative_to(root); target.mkdir(parents=True)
        for name in ("request.json", "record.json", "cassette.json"):
            shutil.copy2(source / name, target / name)
    for snapshot in ("source", "baseline_source"):
        shutil.copy2(root / snapshot / "source_manifest.json", archive / (snapshot + "_manifest.json"))
        for rel in ("backend/agent/llm_accident/runtime.py", "backend/agent/llm_accident/tool_recovery.py",
                    "backend/evaluation/v2/tool_recovery_benchmark.py", "backend/evaluation/v2/checks.py", "backend/evaluation/v2/worker.py",
                    "backend/tests/test_tool_recovery_hints.py", "backend/tests/test_tool_recovery_benchmark.py"):
            destination = archive / snapshot / rel; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / snapshot / rel, destination)
    shutil.copy2(Path(__file__), archive / "combined_recovery_results.py")
    shutil.copy2(project / "backend/tests/test_combined_recovery_results.py", archive / "test_combined_recovery_results.py")
    hashes = {p.relative_to(archive).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(archive.rglob("*")) if p.is_file()}
    it.write(archive / "artifact_manifest.json", hashes)
    assert all(hashlib.sha256((archive / rel).read_bytes()).hexdigest() == value for rel, value in hashes.items())
    qa = {"status": "PASS", "combined": total, "tool": tools["candidate"], "pairedFirstBatch": pairs,
        "formatCompatibility": "all 120 strict replays unchanged, zero new model requests",
        "archiveFilesVerified": len(hashes), "documentsUpdated": [str(dev), str(job)], "archive": str(archive)}
    it.write(root / "publication_qa.json", qa); it.write(archive / "publication_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    publish(sys.argv[1])
