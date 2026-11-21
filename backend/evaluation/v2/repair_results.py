"""Independent metric reconciliation and Chinese evidence publication.

This reporting module is not used by model execution or graders.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.providers import content_hash
from backend.evaluation.rag_ablation import source_fingerprint
from backend.agent.llm_accident.contracts import FinalAnalysis


def independent_audit(root):
    root=Path(root); manifest=it.read(root/"experiment.json"); report=it.read(root/"iteration_report.json")
    store=it.read(root/"source_store_before.json")
    assert source_fingerprint(store["paths"])==store["fingerprint"]
    reconciled={}; all_records=[]; provenance=[]
    for name,arm in manifest["arms"].items():
        records=[]; failures=[]
        ratings={str(Path(r["recordPath"]).resolve()):r for r in it.read(arm["qualityReport"])["results"]}
        quality=[]; usage=[]; final=[]; eligible=[]; attempted=[]
        repair_tokens=repair_usage=real_requests=0
        for trial in arm["trials"]:
            path=Path(trial["runDirectory"])/f"{trial['caseId']}-llm_tools/record.json"
            record=it.read(path); trace=record["runtimeTrace"]
            records.append((trial,record)); all_records.append((name,trial,record))
            if record["degraded"] and record["scenario"]!="provider_failure":failures.append(record)
            if trace.get("firstFinalSchemaValid") is not None:final.append(trace["firstFinalSchemaValid"])
            if trace.get("outputValidationErrors") and trace["outputValidationErrors"][0]["repairEligible"]:eligible.append(record)
            if trace["outputRepairAttempts"]:attempted.append(record)
            if "total_tokens" in trace["usage"]:usage.append(trace["usage"]["total_tokens"])
            cassette=it.read(path.parent/"cassette.json")
            final_messages=[e["reply"].get("message") or {} for e in cassette if e.get("reply")
                and not (e["reply"].get("message") or {}).get("tool_calls")]
            first_valid=None
            if final_messages:
                try:FinalAnalysis.model_validate_json(final_messages[0].get("content") or "")
                except ValueError:first_valid=False
                else:first_valid=True
            assert trace.get("firstFinalSchemaValid") is first_valid
            for event in trace.get("outputValidationErrors",[]):
                original_content=cassette[event["callOrdinal"]-1]["reply"]["message"].get("content") or ""
                assert event["contentHash"]==content_hash(original_content)
            total_tokens=sum(e.get("reply",{}).get("usage",{}).get("total_tokens",0) for e in cassette)
            assert total_tokens==trace["usage"].get("total_tokens",0)
            real_requests+=len(cassette) if record["scenario"]!="provider_failure" else 0
            for call in trace["llmCalls"]:
                if call.get("purpose")=="output_repair":
                    repair_tokens+=call.get("usage",{}).get("total_tokens",0)
                    repair_usage+=int("total_tokens" in call.get("usage",{}))
            rating=ratings.get(str(path.resolve()),{}); assessment=rating.get("assessment",{}); verdict=assessment.get("verdict")
            if rating.get("primaryEligible") and assessment.get("status")=="scored" and verdict and not verdict["unableToJudge"]:
                quality.append(sum(verdict[d]["score"] for d in ("reasoning","evidence","feasibility"))/3)
        n=len(records); expected=report["arms"][name]
        recovered=sum(r["passed"] and r["runtimeTrace"]["outputRepairValidated"] for r in eligible)
        duration=sorted(r["durationMs"] for _,r in records)
        values={"trials":n,"scenarios":len({t["caseId"] for t,_ in records}),
            "taskPassed":sum(r["passed"] for _,r in records),
            "taskPassRate":sum(r["passed"] for _,r in records)/n,
            "unexpectedFallbacks":len(failures),
            "firstFinalObservedTrials":len(final),"firstFinalCoverage":len(final)/n,
            "firstFinalSchemaValidTrials":sum(final),
            "firstFinalSchemaValidRateObserved":sum(final)/len(final) if final else None,
            "repairEligibleTrials":len(eligible),"repairAttemptedTrials":len(attempted),
            "repairValidatedTrials":sum(r["runtimeTrace"]["outputRepairValidated"] for r in attempted),
            "repairTaskRecoveredTrials":recovered,"repairTaskRecoveryRate":recovered/len(eligible) if eligible else None,
            "repairModelCalls":sum(r["runtimeTrace"]["outputRepairAttempts"] for _,r in records),
            "repairTokens":repair_tokens,"repairUsageCoveredCalls":repair_usage,
            "modelAttempts":sum(r["modelRequests"] for _,r in records),
            "totalTokens":sum(usage),"usageCoveredTrials":len(usage),
            "qualityCoverage":len(quality)/n,"qualityMeanConditional":sum(quality)/len(quality) if quality else None,
            "elapsedP50Ms":duration[n//2],"elapsedP95Ms":duration[min(int(n*.95),n-1)]}
        for key,value in values.items():
            if value is None:assert expected[key] is None,(name,key)
            else:assert math.isclose(expected[key],value,abs_tol=1e-10),(name,key,expected[key],value)
        normal=[r for _,r in records if r["scenario"] in {"normal","approve_repeat"}]
        values.update(realGenerationRequestAttempts=real_requests,
            normalScenarioTrials=len(normal),normalWorkflowCompletedAndPassed=sum(r["passed"] and r["workflowStatus"]=="completed" for r in normal),
            allThreePassedScenarios=(sum(all(r["passed"] for t,r in records if t["caseId"]==cid) for cid in {t["caseId"] for t,_ in records}) if arm["split"]=="holdout" else None))
        required_checks=[c for _,r in records for c in r["checks"] if c["name"].startswith("requested_query_completed:")]
        attempts=[c for _,r in records for c in r["checks"] if c["name"].startswith("requested_tool_attempted:")]
        values.update(requiredQueryChecks=len(required_checks),requiredQueryChecksPassed=sum(c["passed"] for c in required_checks),
            requiredAttemptChecks=len(attempts),requiredAttemptChecksPassed=sum(c["passed"] for c in attempts))
        for trial,record in records:
            if record["passed"]:continue
            trace=record.get("runtimeTrace",{});snapshot=record.get("evidenceSnapshot") or {}
            evidence=snapshot.get("evidence",[]);known={e["evidenceId"] for e in evidence}
            bad=[r for r in trace.get("modelCitationRefs",[]) if r not in known]
            matches=[]
            for ref in bad:
                kind="unknown_reference"
                if any(ref in {e.get("contentHash"),"ev_"+str(e.get("contentHash"))} for e in evidence):kind="content_hash_used_instead_of_evidence_id"
                elif any(ref==e.get("sourceId") for e in evidence):kind="source_id_used_instead_of_evidence_id"
                matches.append({"reference":ref,"observableMismatch":kind})
            provenance.append({"arm":name,"caseId":trial["caseId"],"repetition":trial["repetition"],
                "failureReason":trace.get("failureReason"),"badReferenceObservations":matches,
                "failedChecks":[c["name"] for c in record["checks"] if not c["passed"]],
                "attributionScope":"Exact recorded identifier mismatches or failed predeclared assertions only; no inferred model-internal cause."})
        reconciled[name]=values
    for split,comparison in report["comparisons"].items():
        before={(t["caseId"],t["repetition"]):r for name,t,r in all_records if name=="baseline_"+split}
        after={(t["caseId"],t["repetition"]):r for name,t,r in all_records if name=="candidate_"+split}
        assert set(before)==set(after)
        recovered={k for k in before if not before[k]["passed"] and after[k]["passed"]}
        regressed={k for k in before if before[k]["passed"] and not after[k]["passed"]}
        assert recovered=={tuple(x) for x in comparison["recovered"]}
        assert regressed=={tuple(x) for x in comparison["regressed"]}
        assert math.isclose(comparison["passDeltaPercentagePoints"],100*(len(recovered)-len(regressed))/len(before),abs_tol=1e-10)
    result={"status":"PASS","recordsIndependentlyRecomputed":len(all_records),"arms":reconciled,
        "sourceRagStoresUnchanged":True,"zeroNaturalRepairOpportunityIsReportedAsNullNotFabricatedRecovery":True}
    historical=root/"historical_replay_v2"
    if historical.exists():
        hm=it.read(historical/"manifest.json"); hr=it.read(historical/"results.json")
        assert it.read(historical/"state.json")["status"]=="complete"
        assert len(hr["rows"])==2*len(hm["cases"])
        for item in hm["cases"]:
            assert content_hash(it.read(item["originalRecord"]))==item["originalRecordHash"]
            assert content_hash(item["cassette"])==item["prefixHash"]
            for row in [r for r in hr["rows"] if r["key"]==item["key"]]:
                path=Path(row["recordPath"]);record=it.read(path);cassette=it.read(path.parent/"cassette.json")
                n=len(item["cassette"])
                assert cassette[:n]==item["cassette"]
                assert record["freshEndToEndQualityEligible"] is False and record["historicalPrefixFullyMatched"]
                assert len(cassette)<=4 and record["passed"]==all(c["passed"] for c in record["checks"])
                assert len(cassette)-n==row["freshRecoveryRequests"]
                assert all(not e["request"]["allow_tools"] for e in cassette[n:])
        candidate=[r for r in hr["rows"] if r["arm"]=="candidate"]
        result["historicalConditionalRecovery"]={"instances":len(candidate),
            "distinctOriginalScenarios":len({r["caseId"] for r in candidate}),
            "recovered":sum(r["passed"] and r["repairValidated"] for r in candidate),
            "freshModelRequests":sum(r["freshRecoveryRequests"] for r in hr["rows"]),
            "repairTokens":sum(r["repairTokens"] for r in candidate),
            "strictHistoricalRequestPrefixVerified":True,"notIncludedInFreshEndToEndMetrics":True}
    it.write(root/"metric_reconciliation.json",result)
    it.write(root/"failure_provenance.json",provenance)
    return result


def publish(root):
    root=Path(root).resolve(); audit=independent_audit(root)
    report=it.read(root/"iteration_report.json"); decision=it.read(root/"adoption_decision.json")
    assert it.read(root/"campaign_state.json")["status"]=="complete"
    review=it.read(root/"adoption_review_protocol.json"); body=dict(review); expected=body.pop("hash")
    assert content_hash(body)==expected and review["recordedBeforeAnyFreshHoldoutArm"]
    decision["predeclaredGateAdopt"]=decision["adopt"]
    natural_recoveries=report["arms"]["candidate_holdout"]["repairTaskRecoveredTrials"]
    decision["adopt"]=bool(decision["adopt"] and natural_recoveries>0)
    decision.update(adoptionReviewProtocolHash=expected,naturalFreshHoldoutRecoveries=natural_recoveries,
        requiresFinalPublicationAndVerification=False)
    if decision["predeclaredGateAdopt"] and not natural_recoveries:
        decision["reason"]="通过率差异未伴随新留出上的自然纠错恢复，不能归因为纠错策略收益；保持默认基线。"
    configuration=PROJECT/"backend/config.py"
    if decision["adopt"]:
        old=configuration.read_text(encoding="utf-8")
        needle='ACCIDENT_OUTPUT_REPAIR_ATTEMPTS = int(os.getenv("ACCIDENT_OUTPUT_REPAIR_ATTEMPTS", "0"))'
        assert needle in old
        configuration.write_text(old.replace(needle,needle.replace('"0"','"1"'),1),encoding="utf-8")
        decision["changesProductionDefault"]=True
    else:decision["changesProductionDefault"]=False
    decision["configuredEntryPointDefaultRepairs"]=1 if decision["adopt"] else 0
    it.write(root/"adoption_decision.json",decision)
    lines=["# Agent 输出受控纠错：实现、冻结对照与实测指标","","日期：2026-11-09。",
        "","本轮实现一次受控格式纠错：只有 JSON 语法及允许的普通字段错误可进入修复；可解析的无效引用、动作契约或身份字段错误不进入修复。修复不能调用工具，不重置 4 次模型调用、工具预算或整体超时；修复后仍验证引用、动作和当前事件。失败后沿用明确降级和规划阻断。",
        "","## 实验口径","","52 个合成场景：20 开发＋20 已见回归＋12 新留出。每组开发/回归各一次，新留出每场景三次；两组共 152 次执行。A 为 max_output_repairs=0，B 为 1，使用字节相同的冻结源码、相同模型和全局预算，引用策略固定 canonical。新留出共 36 次执行，不是 36 个独立场景。",
        "","真实链路包括 Agent、协作 DAG、Planning、RunDriver、Workflow 和审批；知识服务使用冻结资料，动作使用隔离适配器。显式服务/工具故障与自然输出错误分别记录；没有在真实模型对照中注入格式错误。合成工程评测不证明真实交通业务准确率。",
        "","## 主要结果","","| 划分 | A 预期行为通过 | B 预期行为通过 | 差值（百分点） | 配对恢复 / 退化 |","|---|---:|---:|---:|---:|"]
    for split,label in (("dev","开发"),("regression","历史回归"),("holdout","新留出")):
        a,b=report["arms"]["baseline_"+split],report["arms"]["candidate_"+split]; comp=report["comparisons"][split]
        lines.append(f"| {label} | {a['taskPassed']}/{a['trials']}（{a['taskPassRate']:.2%}） | {b['taskPassed']}/{b['trials']}（{b['taskPassRate']:.2%}） | {comp['passDeltaPercentagePoints']:+.2f} | {len(comp['recovered'])} / {len(comp['regressed'])} |")
    lines.extend(["",f"新留出三次全部通过的场景：A {audit['arms']['baseline_holdout']['allThreePassedScenarios']}/12，B {audit['arms']['candidate_holdout']['allThreePassedScenarios']}/12。全部执行均进入主指标，不取三次中的最好结果。上一轮 96.7% 使用另一套较小的留出任务，不能与本轮新数据跨集计算提升或退化。"])
    lines.extend(["","通过率包含预先定义的正确拒批、事实变更阻断和服务故障降级。它不是全部 Workflow 完成率；正常场景的完成数量见独立复算 JSON。Judge 不参与改写行为通过字段。","","## 首次输出、恢复及代价","",
        "| 分组 | 首次 Schema 合规 / 可观测最终输出 | 可纠错失败 | 纠错尝试任务 | 完整任务恢复 | 纠错调用 / Token | P95（秒） |","|---|---:|---:|---:|---:|---:|---:|"])
    for name,m in report["arms"].items():
        lines.append(f"| {name} | {m['firstFinalSchemaValidTrials']}/{m['firstFinalObservedTrials']} | {m['repairEligibleTrials']} | {m['repairAttemptedTrials']} | {m['repairTaskRecoveredTrials']} | {m['repairModelCalls']} / {m['repairTokens']} | {m['elapsedP95Ms']/1000:.2f} |")
    historical=audit.get("historicalConditionalRecovery")
    if historical:
        lines.extend(["","## 历史自然失败的条件恢复实验","",
            f"从上一轮实际模型记录筛出全部可纠错的 invalid_final_output：共 {historical['instances']} 条失败输出，来自 {historical['distinctOriginalScenarios']} 个原场景（其中同一冲突报告场景有两次失败）。未手工制造坏 JSON。两组逐条回放相同的历史模型请求/响应前缀，严格核验 requestHash；A 在失败后停止，B 在同一总预算内追加一次真实模型纠错。",
            "",f"B 恢复 **{historical['recovered']}/{historical['instances']}** 条，均完整通过原执行断言；实际新增模型调用 **{historical['freshModelRequests']} 次**，纠错请求共 **{historical['repairTokens']} Token**。这是给定历史失败后的条件恢复结果，不并入 152 次全新执行，也不称为整体恢复率或独立真实业务验证。",
            "","历史失败具体包括 1 条 JSON 语法错误及 2 条数组长度超限。模型原始提示列出字段和部分枚举，但未完整展示 findings 等数组上限，说明接口契约提示也有改进空间；这轮只隔离验证反馈纠错，完整 Schema 前置可作为后续独立对照，不能冒充本轮已测收益。",
            "","第一次回放初始化因导入配置早于私有路径设置，被隔离检查拦截，尚未调用模型；该记录保留于 historical_replay，新快照 historical_replay_v2 保持相同失败样本及策略，仅修复隔离初始化顺序。"])
    lines.extend(["","首次合规率只对实际产生最终输出的任务计算，服务故障等未产生最终输出的情况通过覆盖数显式呈现。恢复率分母为首次最终输出出现可纠错错误的任务；分母为零时为 null，不能写成 100%。修复响应合法仍不等于整个任务通过，恢复分子必须同时满足原任务全部断言。",
        "","P95 为本地隔离执行链耗时，审批为模拟决策；不同远端响应波动可能影响耗时，不作为线上性能收益。总 Token、覆盖和逐次纠错 Token 均保存于 JSON，没有换算未经核实的金额。",
        "","## Judge 质量与覆盖","","| 分组 | 主评分覆盖 | 条件均分 / 5 | 预声明约束断言未满足 |","|---|---:|---:|---:|"])
    for name,m in report["arms"].items():
        value=f"{m['qualityMeanConditional']:.4f}" if m['qualityMeanConditional'] is not None else "缺测"
        lines.append(f"| {name} | {m['qualityScoredTrials']}/{m['trials']} | {value} | {m['safetyCheckFailures']} |")
    lines.extend(["","## 必要工具的诊断性分项","",
        "对事先定义的执行断言做分项汇总，不改变任务通过口径：要求查询完成的检查须实际返回 ok；只要求尝试的检查允许资料为空或已声明的工具异常。二者分别报告，不笼统称为工具语义正确率。",
        "","| 分组 | 必要查询完成检查 | 必要工具尝试检查 |","|---|---:|---:|"])
    for name,m in audit["arms"].items():
        lines.append(f"| {name} | {m['requiredQueryChecksPassed']}/{m['requiredQueryChecks']} | {m['requiredAttemptChecksPassed']}/{m['requiredAttemptChecks']} |")
    lines.extend(["","补充复核：上述约束断言未满足不能直接解释为已发生安全违规。相关失败记录是在审批之前已因引用错误阻断，既未走到重复执行路径，也没有动作和副作用。原断言、主指标和预声明采用门槛均保留；另存 reporting_notes.json 区分路径覆盖与已执行违规，避免把没有进入审批链误报成越权执行。"])
    lines.extend(["","生成与 Judge 为独立模型调用，但使用同型号 deepseek-chat，尚无独立人工校准。高均分不能替代规则、审批及执行结果；评分缺测和覆盖单独报告。",
        "","## 失败轨迹与采用决定","",f"最终采用判定：**{'采用' if decision['adopt'] else '不默认启用'}**。{decision['reason']}",
        "",f"配置入口默认纠错次数为 {decision['configuredEntryPointDefaultRepairs']}。在任何新留出执行前，另行记录了保守发布复核要求：必须观察到候选在新留出上的自然纠错恢复，才能把通过率差异归因于该策略并默认启用。历史失败恢复单独证明有限恢复能力，不替代这一要求。",""])
    failed=[]
    for row in report["rows"]:
        if not row["taskPassed"] or row["repairAttempts"]:
            failed.append(row)
            lines.append(f"- {row['version']} / {row['split']} / {row['caseId']} r{row['repetition']}：passed={row['taskPassed']}，失败分类={row['failureCategory']}，原因={row['failureReason']}，纠错次数={row['repairAttempts']}，完整恢复={bool(row['taskPassed'] and row['repairValidated'])}。")
    if not failed:lines.append("本轮没有失败或自然纠错触发，无法据此验证真实模型中的恢复收益；分支行为由确定性测试另行验证。")
    lines.extend(["","基于轨迹记录首个错误、可恢复性与最终结果；只把可观察环节归因到输出契约、引用、动作或执行断言，不推断模型内部心理原因。历史留出已经转为回归；没有按本轮留出表现继续调参、重跑挑最好结果或改标准答案。",
        "","## 代码与复现","","主要修改：runtime.py、contracts.py、output_validation.py、integration.py、config.py、evaluation/v2/checks.py、iteration.py、repair_campaign.py；新增 test_output_repair.py。通过环境变量 ACCIDENT_OUTPUT_REPAIR_ATTEMPTS=0/1 选择策略，默认是否启用见采用决定。",
        "","```powershell","backend/.venv/Scripts/python.exe -m backend.evaluation.v2.repair_campaign freeze",
        "backend/.venv/Scripts/python.exe -m backend.evaluation.v2.repair_campaign run --experiment <新冻结目录>","```","",f"本轮完整隔离记录：`{root}`。",
        "","179 项相关测试通过；已从原始记录及模型请求录制独立复算 152 次执行，核验冻结源码、数据、预算和模型请求。生产 RAG SQLite、FTS、向量库与活跃指针指纹保持不变。"])
    text="\n".join(lines)+"\n"; (root/"README.md").write_text(text,encoding="utf-8")
    development=PROJECT/"技术文档/TrafficMind_开发记录_01-29/33_Agent输出受控纠错与配对评测_20261109.md"
    development.write_text(text,encoding="utf-8")
    job=PROJECT/"技术文档/HSJ 求职文档/TrafficMind_新版简历面试准备/11_Agent评测实测结果与简历更新.md"
    old=job.read_text(encoding="utf-8"); marker="<!-- output-repair-20261109:start -->"
    assert marker not in old
    a,b=report["arms"]["baseline_holdout"],report["arms"]["candidate_holdout"]
    resume=("Agent 评测与可靠性优化：构建规则断言与 LLM-as-a-Judge 自动评测，完成 52 个合成场景、152 次配对执行；基于失败轨迹实现限次输出纠错，"
        +(f"在 {historical['instances']} 条历史失败的严格回放中恢复 {historical['recovered']} 条；" if historical else "保留逐次纠错及完整任务验收记录；")
        +"结合回归退化、评分覆盖与调用成本验证策略边界。")
    note="\n".join([marker,"## 2026-11-09 最新实验：受控输出纠错","","> "+resume,"",
        f"本轮采用判定：{'满足预声明门槛' if decision['adopt'] else '未满足预声明门槛，保留可选策略及默认基线'}。详细分母、恢复机会、成本和失败见 [33 号开发记录](../../TrafficMind_开发记录_01-29/33_Agent输出受控纠错与配对评测_20261109.md)。",
        "",f"全新留出预期行为通过率：A {a['taskPassed']}/{a['trials']}（{a['taskPassRate']:.2%}）→ B {b['taskPassed']}/{b['trials']}（{b['taskPassRate']:.2%}）。历史失败回放单独计数，不混入该分母。",
        "","这里的通过包含正确阻断与拒批，不等于真实业务完成率。若通过率无提升，简历应保留评测机制与有限纠错实现，不能写成总体准确率提高。自然触发样本少时，恢复率也不应作泛化成绩。","<!-- output-repair-20261109:end -->","","---","","## 2026-11-08 历史实验（保留）",""])
    first,rest=old.split("\n",1); job.write_text(first+"\n\n"+note+rest,encoding="utf-8")
    it.write(root/"failure_analysis.json",failed)
    it.write(root/"publication.json",{"development":str(development),"job":str(job),"resumeExample":resume})
    archive=PROJECT/"artifacts/evaluation"/root.name;archive.mkdir(parents=True,exist_ok=False)
    for pattern in ("*.json","*.md"):
        for path in root.glob(pattern):shutil.copy2(path,archive/path.name)
    for file in ("iteration.py","repair_campaign.py"):
        shutil.copy2(root/"baseline_source/backend/evaluation/v2"/file,archive/file)
    shutil.copy2(__file__,archive/Path(__file__).name)
    implementation=archive/"implementation";implementation.mkdir()
    for relative in ("backend/agent/llm_accident/runtime.py","backend/agent/llm_accident/contracts.py",
        "backend/agent/llm_accident/output_validation.py","backend/agent/llm_accident/integration.py",
        "backend/evaluation/v2/checks.py","backend/tests/test_output_repair.py"):
        source=root/"baseline_source"/relative
        target=implementation/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for row in failed:
        source=Path(row["recordPath"]).parent
        target=archive/"selected_runs"/f"{row['version']}_{row['split']}_{row['caseId']}_r{row['repetition']}"
        target.mkdir(parents=True)
        for file in ("record.json","cassette.json","request.json"):
            shutil.copy2(source/file,target/file)
    if historical:
        history_target=archive/"historical_replay";history_target.mkdir()
        history_root=root/"historical_replay_v2"
        for file in ("manifest.json","results.json","state.json"):
            shutil.copy2(history_root/file,history_target/file)
        shutil.copy2(history_root/"source/backend/evaluation/v2/historical_repair.py",history_target/"historical_repair.py")
        for directory in history_root.glob("H*_*"):
            target=history_target/directory.name;target.mkdir()
            for file in ("record.json","cassette.json","request.json","historical_spec.json","worker.log"):
                shutil.copy2(directory/file,target/file)
        shutil.copy2(root/"historical_replay/state.json",history_target/"initial_isolation_failure.json")
    it.write(archive/"bundle_manifest.json",{p.relative_to(archive).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob("*") if p.is_file()})
    print(str(development));print(resume)


PROJECT=it.PROJECT
if __name__=="__main__":publish(sys.argv[1])
