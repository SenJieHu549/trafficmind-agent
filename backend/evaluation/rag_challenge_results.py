"""Archive completed stress-test evidence and write a compact Chinese result note."""
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import sys

from backend.evaluation import rag_ablation as ab


def publish(root):
    root = Path(root).resolve()
    report = ab.read(root/"ablation_report.json")
    audit = ab.read(root/"independent_audit.json")
    assert ab.read(root/"state.json")["status"] == "complete" and audit["status"] == "PASS"
    assert report["sourceUnchanged"]
    manifest = ab.read(root/"experiment.json")
    prior = ab.read(Path(manifest["parentPilot"])/"ablation_report.json")
    corpus = ab.read(root/"corpus.json")
    n = report["arms"]["A"]["queries"]
    labels = {"A":"Dense 单路", "B":"Dense＋BM25＋RRF", "C":"B＋Rerank"}
    def table(arms):
        rows = ["| 组别 | Recall@5 | MRR@5 | Hit@5 | nDCG@5 | Precision@5 | 候选召回率 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, value in arms.items():
            rows.append(f"| {arm} {labels[arm]} | {value['recallAt5']:.2%} | {value['mrrAt5']:.4f} | {value['hitAt5']:.2%} | {value['ndcgAt5']:.4f} | {value['precisionAt5']:.2%} | {value['candidateRecall']:.2%} |")
        return rows
    lines = ["# TrafficMind RAG 三组消融：相似预案压力测试实测", "",
        f"本轮使用 {len(corpus['documents'])} 篇文档、{len(corpus['chunks'])} 个分块及 {n} 条冻结查询。其中 25 篇为原有知识文档，60 篇为仅用于评测的合成处置卡。三类查询各 8 条：编号匹配、条件转述、跨文档条件组合。",
        "", "标准答案根据合成场景键在模型检索之前确定，由助手编写，未经独立人工标注。适用卡片需要同时满足地点、时段、触发条件；通用部分相关文档不计为完整命中。本轮属于小样本检索压力测试，不能表述为真实业务整体正确率或 Agent 端到端成功率。", "",
        "## 本轮实测指标", "", *table(report["arms"]), "", "Recall@5 为逐查询文档召回率的宏平均；MRR@5 衡量首个正确文档的排序；nDCG@5 同时关注全部标准文档及其排序。每个查询在排名分块中按文档首次出现去重，然后取前 5；失败查询按零分保留。", "",
        "## 配对对照", ""]
    for key, delta in report["comparisons"].items():
        left, right = key.split("_to_")
        lines.append(f"- {left} → {right}：Recall@5 {delta['recallDeltaPercentagePoints']:+.2f} 个百分点；MRR@5 {delta['mrrDelta']:+.4f}；nDCG@5 {delta['ndcgDelta']:+.4f}。召回改善/退化 {len(delta['recallImproved'])}/{len(delta['recallRegressed'])} 条，排序改善/退化 {len(delta['rankingImproved'])}/{len(delta['rankingRegressed'])} 条。")
    for category, arms in report["byCategory"].items():
        lines.extend(["", "## 分组："+category, "", *table(arms)])
    raw = {r["caseId"]:r for r in ab.read(root/"retrieval_rows.json")}
    reranked = {r["caseId"]:r for r in ab.read(root/"rerank_rows.json")}
    documents = {d["document_id"]:d for d in corpus["documents"]}
    regressions = []
    for cid in report["comparisons"]["B_to_C"]["rankingRegressed"]:
        item = raw[cid]
        ranks = {}
        for arm in "ABC":
            ids = list(dict.fromkeys(c["document_id"] for c in (reranked[cid]["C"] if arm=="C" else item[arm])))
            ranks[arm] = [{"documentId":gold,"title":documents[gold]["title"],
                "rank":ids.index(gold)+1 if gold in ids else None} for gold in item["goldDocumentIds"]]
        regressions.append({"caseId":cid,"query":item["query"],"goldRanks":ranks,
            "attribution":"All gold documents are present in the B/C candidate pool. The observed regression is introduced by reranking, not missing recall. Model-internal cause has not been experimentally established.",
            "rerankedTopFive":[{"documentId":identity,"title":documents[identity]["title"],
                "content":documents[identity]["content"]} for identity in list(dict.fromkeys(c["document_id"] for c in reranked[cid]["C"]))[:5]]})
    ab.write(root/"failure_analysis.json",regressions)
    dataset = ab.read(root/"queries.json")
    ab.write(root/"regression_cases.json", {"status":"saved_for_future_regression_not_a_new_executed_fix",
        "cases":[c for c in dataset["cases"] if c["caseId"] in {r["caseId"] for r in regressions}]})
    lines.extend(["", "## 保留的失败案例", "",
        "C 相比 B 有 2 条排序退化：S18 的两份正确资料从第 1/3 名变为第 2/5 名；S24 从第 1/2 名变为第 6/3 名，造成 1 条多文档查询召回不完整。两题的正确资料均已进入 B/C 候选池，所以可定位到重排序阶段，而不是召回缺失。",
        "", "S24 的前两名是同地点、同事件但触发条件不同的卡片，说明本轮模型排序未能始终保持条件匹配。此为排名和文档内容的可观察现象，尚未验证模型内部原因。已保留 `failure_analysis.json` 与 `regression_cases.json`，作为后续修复的回归用例；本轮没有继续调参或宣称已修复。"])
    lines.extend(["", "## 原有语料上的上一轮基准", "", "以下为另一套固定语料及查询上的独立结果，不能将两轮数值之差解释为优化收益。上一轮 25 篇文档 / 58 个分块 / 30 条查询，出现召回率天花板，因此追加了本轮压力测试。", "", *table(prior["arms"]), "",
        "## 运行与核验", "", "- 使用真实本地 Qwen3-Embedding-0.6B 与 Qwen3-Reranker-0.6B；无 Fake/Hash 降级。全部原始和新增分块重新以 float32 编码，查询与重排序也使用 float32。",
        "- 固定 K=5、候选池 12；A 输入 Dense 24，B 输入 Dense 12＋BM25 12，经 RRF(k=60) 截取 12；C 只重排 B 的同一候选集。控制的是排名输入预算，不是等耗时；Dense24 共用计算。",
        "- 同一语料、查询、有效期过滤；未加入 Query Rewrite、Structured 通道或权威度加权。未依据本轮结果调整参数、标签或选择样本。",
        f"- 检索异常 A/B/C：{report['arms']['A']['errors']}/{report['arms']['B']['errors']}/{report['arms']['C']['errors']}；稀疏检索实际后端：{report['sparseBackends']}。",
        f"- 独立从原始排名复算 {audit['rankingRows']} 条组别记录，核验 {audit['rerankPairs']} 个真实重排序对；模型、源码、语料、协议和标准答案哈希一致。生产 SQLite/FTS/向量库及活跃指针保持不变。",
        "- 保留先前 bf16 构建中断记录；中断发生在任何检索评分之前。完整新运行没有复用旧压力测试预测。",
        "", "## 复现入口", "", "```powershell", "backend/.venv/Scripts/python.exe -m backend.evaluation.rag_challenge prepare --base D:/TrafficMind-cache/rag-ablation/rag_20261108T144543_36af1b8d", "# 使用 prepare 输出的新实验目录；不要覆盖已完成运行", "backend/.venv/Scripts/python.exe -m backend.evaluation.rag_ablation run --experiment <新实验目录>", "```", "", f"本轮完整隔离运行目录：`{root}`。", ""])
    text = "\n".join(lines)
    (root/"README.md").write_text(text,encoding="utf-8")
    archive = ab.PROJECT/"artifacts/evaluation"/root.name
    archive.mkdir(parents=True,exist_ok=False)
    # No credentials, environment files, database or model weights in the portable archive.
    for pattern in ("*.json","*.log","*.md"):
        for path in root.glob(pattern): shutil.copy2(path,archive/path.name)
    for script in ("rag_ablation.py","rag_challenge.py"):
        shutil.copy2(root/"source/backend/evaluation"/script,archive/script)
    shutil.copy2(__file__,archive/Path(__file__).name)
    audit_script = root.parent/"audit_rag_challenge_20261108.py"
    shutil.copy2(audit_script,archive/audit_script.name)
    interrupted = root.parent/"stress_20261108T151307_48d23a90"
    history = archive/"construction_interruption"
    history.mkdir()
    for name in ("interruption.json","challenge_spec.json","construction.json","build.log"):
        if (interrupted/name).exists(): shutil.copy2(interrupted/name,history/name)
    file_hashes = {p.relative_to(archive).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in archive.rglob("*") if p.is_file()}
    ab.write(archive/"bundle_manifest.json",file_hashes)
    development = ab.PROJECT/"技术文档/TrafficMind_开发记录_01-29/32_RAG相似预案压力测试与实测指标_20261108.md"
    development.write_text(text,encoding="utf-8")
    job = ab.PROJECT/"技术文档/HSJ 求职文档/TrafficMind_新版简历面试准备/12_RAG检索消融实测与简历更新.md"
    original = job.read_text(encoding="utf-8")
    marker = "<!-- rag-stress-20261108:start -->"
    if marker in original: raise ValueError("job_result_already_published")
    a, c = report["arms"]["A"], report["arms"]["C"]
    resume = (f"构建文档级 RAG 自动化评测，在 85 篇文档、24 条冻结合成压力查询上完成 Dense／混合召回／重排消融，"
        f"文档级 Recall@5 {a['recallAt5']:.2%} → {c['recallAt5']:.2%}，MRR@5 {a['mrrAt5']:.4f} → {c['mrrAt5']:.4f}；保留原始排名、失败案例和独立复算记录。")
    latest = "\n".join([marker,"## 最新补充：相似预案压力测试","",
        "85 篇文档 / 118 分块 / 24 查询。新增 60 张合成卡，标准答案在评分前冻结；未经独立人工标注，仅支持小样本检索层结论。详见 [32 号开发记录](../../TrafficMind_开发记录_01-29/32_RAG相似预案压力测试与实测指标_20261108.md)。","",
        *table(report["arms"]),"","可用于面试材料的限定表述：","","> "+resume,"",
        "该对比是在本轮相同冻结数据上的配置消融，不是跨两轮语料的前后对比，也不等于 Agent 总体执行正确率。下面保留第一轮基准和解释。",
        "<!-- rag-stress-20261108:end -->","","---","","## 第一轮历史基准","",])
    original = original.replace("日期：2026-11-08。最新 RAG 数字以本文件和", "日期：2026-11-08。第一轮 RAG 数字见本节和",1)
    first,rest = original.split("\n",1)
    job.write_text(first+"\n\n"+latest+rest,encoding="utf-8")
    ab.write(root/"published_documents.json", {"development":str(development),"job":str(job),
        "archive":str(archive),"resumeExample":resume})
    print(str(archive/"README.md"))


if __name__ == "__main__": publish(sys.argv[1])
