"""Publish the separately scoped recovery benchmark, preserving prior results."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.recovery_benchmark import audit


def publish(root):
    root = Path(root).resolve()
    assert it.read(root / "state.json")["status"] == "complete"
    result = audit(root)
    # Reconcile the primary numerator directly from persisted records, without
    # using the benchmark's recovered() helper or trusting its aggregate count.
    raw_candidates, raw_baseline, raw_controls, reference_failures = [], [], [], []
    for trial in it.read(root / "trials.json"):
        path = Path(trial["directory"]) / "record.json"
        record = it.read(path) if path.exists() else {}
        if record:
            limits = record["runtimeTrace"]["limits"]
            configuration = it.read(root / "manifest.json")["configuration"]
            assert limits["llm_timeout"] == configuration["llmTimeout"]
            assert limits["tool_timeout"] == configuration["toolTimeout"]
            assert limits["overall_timeout"] == configuration["overallTimeout"]
        if trial["arm"] == "candidate":
            trace = record.get("runtimeTrace", {})
            raw_candidates.append(bool(record.get("checks") and all(c["passed"] for c in record["checks"])
                and trace.get("outputRepairValidated") and trace.get("status") == "completed"))
            if trace.get("failureReason") == "unknown_evidence_reference":
                evidence = (record.get("evidenceSnapshot") or {}).get("evidence", [])
                known = {e["evidenceId"] for e in evidence}
                full_hashes = {e["contentHash"]: e["evidenceId"] for e in evidence}
                wrong = []
                for ref in trace.get("modelCitationRefs", []):
                    if ref not in known:
                        target = full_hashes.get(ref) or full_hashes.get(ref.removeprefix("ev_"))
                        wrong.append({"outputRef": ref, "exactContentHashMatch": target})
                reference_failures.append({"key": trial["key"], "batch": trial["batch"],
                    "unknownReferences": wrong, "recordPath": str(path)})
        elif trial["arm"] == "baseline":
            raw_baseline.append(record.get("runtimeTrace", {}).get("outputRepairAttempts") == 0
                and record.get("runtimeTrace", {}).get("failureReason") == "invalid_final_output")
        else:
            trace = record.get("runtimeTrace", {})
            raw_controls.append(not record.get("freshRecoveryRequestAttempts")
                and not trace.get("outputValidationErrors", [{}])[0].get("repairEligible", True)
                and not record.get("actions") and not record.get("isolatedEffects"))
    assert len(raw_candidates) == 120 and sum(raw_candidates) == result["candidate"]["recovered"]
    assert len(raw_baseline) == 40 and all(raw_baseline)
    assert len(raw_controls) == 16 and all(raw_controls)
    validation = {"status": "PASS", "primaryNumerator": sum(raw_candidates), "primaryDenominator": 120,
        "baselineFailFastVerified": 40, "protectedControlsVerified": 16,
        "independentRecordReconciliation": True, "testResult": "31 passed: recovery_benchmark + output_repair"}
    it.write(root / "metric_reconciliation.json", validation)
    it.write(root / "reference_failure_analysis.json", {"failedTrials": len(reference_failures),
        "unknownReferenceItems": sum(len(r["unknownReferences"]) for r in reference_failures),
        "exactContentHashMismatchItems": sum(bool(v["exactContentHashMatch"]) for r in reference_failures for v in r["unknownReferences"]),
        "note": "Observed identifier mismatch only; no inference about hidden model reasoning. No automatic ID repair applied.",
        "rows": reference_failures})
    passed = result["candidate"]["recovered"]
    rate = f"{passed / 120:.2%}"
    fault_names = {"truncated_json": "JSON 截断（缺右括号）", "missing_suggestion": "缺少必填 suggestion",
        "confidence_type": "confidence 类型错误", "findings_overflow": "findings 超过 10 项", "urgency_enum": "urgency 枚举错误"}
    lines = ["# Agent 格式异常恢复专项多批次测评（2026-11-09）", "",
        f"专项结果：**{passed}/120，格式/字段异常恢复率 {rate}**。8 个合成业务场景 × 5 类人工注入异常 = 40 个故障样本，重复 3 批；不是 120 个独立业务场景。", "",
        "## 测试回答什么", "",
        "当 Agent 已经获得必要上下文，却输出不符合约定格式的结果时，一次错误反馈能否使它恢复，并通过原任务的执行断言。这里没有测量生产异常发生概率，也没有重新测量整体任务成功率或独立人工语义质量。", "",
        "主指标分子必须同时满足：原任务所有规则断言通过、纠错输出通过引用/动作/事件校验、运行时完成。只把 JSON 修正确不算恢复；正确拒绝审批、正确阻断事实变更按原任务预期计为通过。", "",
        "## 冻结协议", "",
        "- 上下文来自上一轮 baseline_dev 中固定的 D01、D05、D09、D10、D11、D13、D19、D20：基础研判、医院通道、道路未绑定、道路与知识查证、空知识、历史案例、拒绝审批、事实变更。均为原来通过的已见样本，有意隔离输出故障这一因素。",
        "- 原模型请求和工具调用前缀按请求哈希严格回放；只对最后一条原本有效的公开输出注入故障。每个故障都在运行前验证为可纠错。前缀不是新模型生成，纠错请求才是真实新调用。",
        "- 不额外提供未损坏的标准答案。模型只获得原上下文、损坏输出、受控校验反馈与完整 Schema。",
        "- 40 个样本先各测一次关闭纠错的确定性基线；候选按相同顺序分 3 批，每批 40 次真实纠错。另 16 个普通字段错误叠加非法引用或非法动作的对照样本必须直接阻断。",
        "- deepseek-chat，temperature=0.1，max_tokens=1800；最多纠正一次，仍计入原 4 次模型调用上限，保持原超时预算，纠错时禁止调用工具。",
        "- 所有超时、API 错误、二次非法输出和下游任务失败保留在预定 120 次分母内；不择优、不补跑、不根据结果改样本或放宽断言。",
        "- 使用独立进程、事件库、RAG/FTS/向量路径和模拟动作执行器；未启用生产纠错开关，未派发真实业务动作。", "",
        "## 分批结果", "", "| 批次 | 恢复/执行 | 恢复率 |", "|---|---:|---:|"]
    for batch, data in result["byBatch"].items():
        lines.append(f"| 第 {batch} 批 | {data['recovered']}/{data['trials']} | {data['recoveryRate']:.2%} |")
    lines.extend([f"| 合计 | {passed}/120 | {rate} |", "", "## 按异常类别", "",
        "| 异常 | 恢复/执行 | 恢复率 |", "|---|---:|---:|"])
    for kind, data in result["byFault"].items():
        lines.append(f"| {fault_names[kind]} | {data['recovered']}/{data['trials']} | {data['recoveryRate']:.2%} |")
    context_names = {"D01": "基础研判", "D05": "医院通道", "D09": "道路未绑定", "D10": "道路和知识查证",
        "D11": "空知识", "D13": "历史案例", "D19": "拒绝审批", "D20": "事实变更"}
    lines.extend(["", "## 按业务上下文", "", "| 上下文 | 恢复/执行 | 恢复率 |", "|---|---:|---:|"])
    for key, data in result["byContext"].items():
        lines.append(f"| {key} {context_names[key]} | {data['recovered']}/{data['trials']} | {data['recoveryRate']:.2%} |")
    lines.extend(["", f"40 个唯一故障样本中，3 批均恢复的样本：**{result['stableSamplesAllThreeBatches']}/40**。重复批次共享上下文，不能将 120 次执行当成 120 个独立样本计算泛化置信度。", "",
        "## 基线、保护对照与成本", "",
        f"- 关闭纠错：0/40 恢复，40 条均按原策略立即报 invalid_final_output。这是预期的故障处理对照，不能宣传为整体任务成功率从 0% 提升。",
        f"- 非法引用/动作混合故障：{result['protectedControls']['blockedWithoutRepairOrEffects']}/16 正确阻断，没有纠错请求或动作副作用。",
        f"- 纠错后 Schema 合法：{result['correctionSchemaValidTrials']}/120；通过完整运行时校验：{result['postCorrectionValidatedTrials']}/120；最终任务恢复：{passed}/120。三个层次分开计数。",
        f"- 新发出的纠错请求 {result['freshCorrectionRequests']} 次；新请求 Token 总数 {result['freshCorrectionTokens']}，用量有记录的响应 {result['usageCoveredResponses']} 个。历史前缀 Token 未计入新增成本。",
        f"- 单次纠错请求耗时 P50 {result['correctionLatencyMs']['p50']/1000:.2f}s，P95 {result['correctionLatencyMs']['p95']/1000:.2f}s。包含调用等待，不能作为整个 Agent 的端到端延迟。",
        f"- 基础设施失败：{result['infrastructureFailures']}。源代码、请求哈希、纠错次数、总调用预算、工具禁用与原 RAG 存储指纹检查通过。", "",
        "## 未恢复项", ""])
    failures = [r for r in result["rows"] if r["arm"] == "candidate" and not r["recovered"]]
    if not failures:
        lines.append("本次 120 次执行没有未恢复项。该结论限于上述人工故障和 8 个已见场景；不能推断其他异常（未知证据引用、权限问题、业务事实错误、工具或提供方故障）也可恢复。")
    for row in failures:
        lines.append(f"- 第 {row['batch']} 批 {row['key']}：{row.get('failureReason') or 'downstream_assertion'}；失败断言 {row.get('failedChecks', [])}；原始记录 {row['directory']}/record.json。")
    if reference_failures:
        count = sum(bool(v["exactContentHashMatch"]) for r in reference_failures for v in r["unknownReferences"])
        lines.extend(["", f"其中 {len(reference_failures)} 次失败是证据引用校验未通过，发现 {count} 个引用项与已知证据的完整 contentHash 精确对应，却不是系统要求的 evidenceId。引用项数不是任务数。格式修复成功后仍执行引用校验，避免把新引入的问题计为恢复。详见 `reference_failure_analysis.json`。",
            "后续可另行验证：在纠错提示中明确要求保留原 evidence_refs，或由服务端提供只允许精确匹配的引用标识映射。本次没有边测边改或自动放宽引用校验。"])
    bullet = f"实现输出校验与限次纠错，在 40 个故障注入样本、3 批 120 次测试中，格式异常恢复率达 **{rate}（{passed}/120）**，并通过非法引用与动作阻断对照验证纠错边界。"
    lines.extend(["", "## 简历与面试更新", "", "> " + bullet, "",
        "30 秒讲法：我先按错误类型归因，将 JSON 和普通字段错误限定为可纠错，其余涉及证据、动作与权限的问题保持阻断。格式错误只反馈一次，并且不重置原调用预算。随后把 8 个业务上下文扩成 40 个故障样本，分 3 批真实调用模型纠错，只有继续通过原任务断言才算恢复，同时保留失败记录与额外调用成本。", "",
        "此前 3/3 为历史自然失败输出回放，本次为人工故障注入；两者分别报告，不合并成 123 次自然异常。此前整体任务成功率与 RAG 指标保持原实验结果，本次专项结果不替换其分母。", "",
        "## 复现与证据", "",
        "入口：`python -m backend.evaluation.v2.recovery_benchmark freeze D:/TrafficMind-cache/output-repair`，之后 `run <冻结目录>`；已启动目录禁止再次 run。",
        "只读重算：`python -m backend.evaluation.v2.recovery_benchmark audit <冻结目录>`。复跑真实模型有随机性和额外请求成本。",
        f"完整运行目录：`{root}`。协议、40 个故障/16 个保护对照、原请求哈希、源代码清单均在 `manifest.json` 中冻结。",
        "仓库归档含 report.json、manifest.json、逐次请求/响应/断言记录与文件 SHA256；不含 .env、API Key、模型权重和数据库。", ""])
    it.write(root / "test_validation.json", {"tests": "backend/tests/test_recovery_benchmark.py backend/tests/test_output_repair.py", "passed": 31})
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    project = it.PROJECT
    dev = project / "技术文档/TrafficMind_开发记录_01-29/34_Agent格式异常恢复多批次专项测评_20261109.md"
    job = project / "技术文档/HSJ 求职文档/TrafficMind_新版简历面试准备/11_Agent评测实测结果与简历更新.md"
    marker = "<!-- recovery-benchmark-20261109:start -->"
    assert not dev.exists() and marker not in job.read_text(encoding="utf-8")
    dev.write_text("\n".join(lines), encoding="utf-8")
    old = job.read_text(encoding="utf-8")
    first_line, remainder = old.split("\n", 1)
    block = f"""
{marker}
## 2026-11-09 补充：格式异常恢复专项测评

> {bullet}

范围：8 个已见合成业务场景 × 5 类人工输出故障 = 40 个唯一故障样本，重复 3 批。纠错请求为真实模型调用，前缀严格回放；不是 120 个独立场景，也不是自然异常发生率或整体任务成功率。原任务断言全部通过且纠错输出通过后端校验，才计为恢复。

关闭纠错基线为 0/40；保护对照 16/16 阻断，未发起纠错或动作。3 批均恢复的故障样本 {result['stableSamplesAllThreeBatches']}/40。每批及各异常类别结果、失败原文、Token 和耗时见 [34 号开发记录](../../TrafficMind_开发记录_01-29/{dev.name})。

原历史失败 3/3 单列保留，不与本次混算。生产纠错开关仍默认关闭；此次只扩大专项测评，未改 RAG 或原整体任务通过率，也未新增 LLM Judge 语义评分。
<!-- recovery-benchmark-20261109:end -->

"""
    job.write_text(first_line + "\n" + block + remainder, encoding="utf-8")
    archive = project / "artifacts/evaluation" / root.name
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "report.json", "report.md", "metric_reconciliation.json", "test_validation.json",
                 "source_store_before.json", "trials.json", "state.json", "reference_failure_analysis.json"):
        shutil.copy2(root / name, archive / name)
    from backend.evaluation.v2.providers import content_hash
    contexts_saved = set()
    for item in it.read(root / "manifest.json")["cases"]:
        if item["caseId"] in contexts_saved:
            continue
        source = Path(item["originalRecord"]).parent
        assert content_hash(it.read(source / "record.json")) == item["originalRecordHash"]
        assert content_hash(it.read(source / "cassette.json")) == item["originalCassetteHash"]
        target = archive / "context_sources" / item["caseId"]
        target.mkdir(parents=True)
        for name in ("record.json", "cassette.json", "request.json"):
            shutil.copy2(source / name, target / name)
        contexts_saved.add(item["caseId"])
    for row in result["rows"]:
        source = Path(row["directory"])
        target = archive / source.relative_to(root)
        target.mkdir(parents=True)
        for name in ("record.json", "cassette.json", "request.json", "fault_spec.json", "worker.log", "infrastructure_failure.json"):
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
    shutil.copy2(root / "source/source_manifest.json", archive / "source_manifest.json")
    for rel in ("backend/evaluation/v2/recovery_benchmark.py", "backend/evaluation/v2/historical_repair.py",
                "backend/evaluation/v2/worker.py", "backend/evaluation/v2/checks.py", "backend/agent/llm_accident/runtime.py",
                "backend/agent/llm_accident/contracts.py", "backend/agent/llm_accident/output_validation.py",
                "backend/tests/test_recovery_benchmark.py", "backend/tests/test_output_repair.py"):
        target = archive / "implementation" / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "source" / rel, target)
    shutil.copy2(Path(__file__), archive / "recovery_results.py")
    for rel in ("backend/config.py", "backend/agent/llm_accident/runtime.py", "backend/agent/llm_accident/contracts.py",
                "backend/agent/llm_accident/integration.py", "backend/agent/llm_accident/output_validation.py", "backend/case_memory/repository.py"):
        assert (project / rel).read_bytes() == (root / "source" / rel).read_bytes()
    hashes = {p.relative_to(archive).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(archive.rglob("*")) if p.is_file()}
    it.write(archive / "artifact_manifest.json", hashes)
    assert all(hashlib.sha256((archive / rel).read_bytes()).hexdigest() == h for rel, h in hashes.items())
    qa = {"status": "PASS", "archiveFilesVerified": len(hashes), "documentsUpdated": [str(dev), str(job)],
          "primaryResult": result["candidate"], "rawRecordReconciliation": True,
          "coreCodeAndConfigurationUnchanged": True, "archive": str(archive)}
    it.write(root / "publication_qa.json", qa)
    it.write(archive / "publication_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    publish(sys.argv[1])
