"""One frozen synthetic near-neighbour stress test, separate from the base pilot.

Build only in a copied index. Labels are generated from explicit scenario keys
before retrieval. No tuning or case selection is performed using predictions.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

from backend.evaluation import rag_ablation as ab


def specification():
    sites = [
        ("育才学校东门", "小学放学接送口", "安排护学人员保护学生过街，接送车即停即走"),
        ("仁和医院北门", "医院急诊入口", "保留救护车进出通道，安排人员引导就诊车辆"),
        ("新城体育馆西门", "体育场馆观众出口", "分批疏散观众，引导社会车辆前往外围停车场"),
    ]
    families = [
        ("信号灯异常", "灯控故障", [
            ("全路口信号灯全部熄灭", "各方向的红绿灯都不亮", "请求维修并由交警现场交替放行"),
            ("单一进口绿灯持续常亮", "一个方向绿灯一直不变而其他方向正常", "隔离异常进口，人工指挥该方向并联系信号维护")]),
        ("拥堵", "车辆排队", [
            ("队列长度150米且未溢出上游路口", "排队约一百五十米，还没有堵住上个交叉口", "保持路口清空，现场疏导并观察队列变化"),
            ("队列长度350米且已溢出上游路口", "车龙三百五十米，已经回堵到上个交叉口", "启动上游分流并协调信号配时，避免继续放车进入堵段")]),
        ("车辆故障滞留", "汽车抛锚", [
            ("故障车可以自行移动", "车坏了但仍能够挪到不碍事的位置", "开启危险报警闪光灯，将车辆移至安全区域再检修"),
            ("故障车完全无法移动", "抛锚车彻底开不动且仍占用车道", "设置来车方向警告，保护现场并联系拖移力量")]),
        ("违法停车", "车辆乱停", [
            ("车辆未阻断急救通道", "乱停的车尚未挡住救护车通行路线", "联系驾驶人驶离，记录现场并恢复停车秩序"),
            ("车辆已经阻断急救通道", "停放车辆把救护车通行路线完全堵死", "优先清空救援通道，驾驶人不到场时联系交警依法处置")]),
        ("暴雨积水", "强降雨道路积水", [
            ("积水深度15厘米且车辆尚可低速通过", "路面积水约十五公分，汽车仍能缓慢通行", "设置警示并联系排水，持续监测积水，不引导行人涉水"),
            ("积水深度35厘米且车辆已经无法通过", "积水三十五公分，汽车已经过不去", "封控积水路段并组织绕行，联系市政排水和现场救援")]),
    ]
    documents = []
    lookup = {}
    rng = random.Random(20261108)
    codes = rng.sample(range(100000, 999999), 60)
    for fi, (event, alias, states) in enumerate(families):
        for si, (site, site_alias, site_action) in enumerate(sites):
            for state, (condition, condition_alias, action) in enumerate(states):
                for phase in range(2):
                    code = str(codes[len(documents)])
                    period = ["早高峰", "平峰"][phase]
                    phase_action = ["提前增派一组疏导人员，协调相邻路口控流", "保持常规巡查力量，需要增援时再申请调度"][phase]
                    source = f"stress:{fi}:{si}:{state}:{phase}"
                    title = f"演练处置卡 {code}：{site}{event}（{period}）"
                    content = (f"{title}。仅用于软件评测的合成预案，不是实际交通指令。"
                        f"适用地点：{site}；适用时段：{period}；必要触发条件：{condition}。"
                        f"现场措施：{action}。地点配套措施：{site_action}。时段调度：{phase_action}。"
                        "选择此卡必须同时匹配地点、时段和触发条件；其他条件的卡片不能替代本卡。")
                    item = dict(source=source, code=code, title=title, content=content, event=event,
                        site=site, period=period, condition=condition, siteAlias=site_alias,
                        conditionAlias=condition_alias, eventAlias=alias)
                    documents.append(item)
                    lookup[(fi, si, state, phase)] = item
    cases = []
    # Fixed balanced sampling of sites, event families and conditions. No model runs here.
    for i in range(8):
        item = lookup[(i % 5, i % 3, (i // 2) % 2, i % 2)]
        cases.append(dict(caseId=f"S{i+1:02}", category="exact_identifier",
            query=f"请检索编号{item['code']}的演练处置卡，确认该卡的适用条件和现场措施。",
            goldSources=[item["source"]], rationale="编号唯一对应指定演练处置卡，其他编号不满足任务。"))
    for i in range(8):
        item = lookup[((i+2) % 5, (i+1) % 3, i % 2, (i//2) % 2)]
        period_alias = "早晨车流最密集的时候" if item["period"] == "早高峰" else "非高峰的普通时段"
        cases.append(dict(caseId=f"S{i+9:02}", category="paraphrased_conditions",
            query=f"在{period_alias}，{item['siteAlias']}出现{item['eventAlias']}，{item['conditionAlias']}。请找出同时符合这三个条件的演练卡，应该怎么处理？",
            goldSources=[item["source"]], rationale="地点语义、时段和触发条件三项都必须满足；通用处置仅为部分相关。"))
    for i in range(8):
        fi, si, state = (i+1) % 5, (i+2) % 3, i % 2
        peak, offpeak = lookup[(fi, si, state, 0)], lookup[(fi, si, state, 1)]
        cases.append(dict(caseId=f"S{i+17:02}", category="multi_document_conditions",
            query=f"对{peak['site']}的{peak['event']}，已知{peak['condition']}，请同时找出早高峰和平峰两张适用的演练处置卡，对照人员调度与现场措施。",
            goldSources=[peak["source"], offpeak["source"]], rationale="需要相同地点、事件及触发条件下的两种时段卡；必须召回两份完整证据。"))
    return {"schemaVersion": "trafficmind-rag-stress-1", "annotationPolicy":
        "Assistant-authored synthetic scenario keys and deterministic gold mapping, frozen before any ranking. No independent human annotation. Exact card selection requires all scenario conditions; generic partial matches are not gold.",
        "purpose": "Near-neighbour stress test after base pilot ceiling; not a production traffic accuracy benchmark or evidence of natural query frequency.",
        "seed": 20261108, "documents": documents, "cases": cases}


def prepare(base, output):
    from backend.evaluation.v2.experiments import freeze_source
    base = Path(base).resolve()
    previous = ab.read(base/"experiment.json")
    if ab.source_fingerprint(previous["sourcePaths"]) != previous["sourceFingerprintBefore"]:
        raise ValueError("production_corpus_changed_since_pilot")
    root = Path(output).resolve()/("stress_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    spec = specification()
    ab.write(root/"challenge_spec.json", spec)
    for name in ("rag.db", "fts.db"):
        ab.sqlite_snapshot(base/name, root/name)
    shutil.copytree(base/"vectors", root/"vectors", ignore=shutil.ignore_patterns("*-wal", "*-shm"))
    ab.sqlite_snapshot(base/"vectors/chroma.sqlite3", root/"vectors/chroma.sqlite3")
    for name in ("corpus.json", "queries.json"):
        shutil.copy2(base/name, root/name)
    manifest = deepcopy(previous)
    manifest.update(root=str(root), retrievalReuse=None,
        source=freeze_source(ab.PROJECT, root/"source"),
        parentPilot=str(base), challengeSpecHash=ab.digest(spec))
    manifest["protocol"].update(scope=spec["purpose"], annotationPolicy=spec["annotationPolicy"],
        embeddingDtype="float32", vectorPrecisionPolicy="Re-encode all original and synthetic chunks in float32; queries use the same precision. No mixed-precision vector provenance.",
        tuningPolicy="One frozen 24-query/60-card challenge; no prediction-based selection or parameter tuning; retain all regressions. Previous base pilot is reported separately.")
    manifest["protocolHash"] = ab.digest(manifest["protocol"])
    ab.write(root/"experiment.json", manifest)
    ab.write(root/"construction.json", {"status":"spec_frozen_before_indexing", "specHash":ab.digest(spec)})
    env = {**os.environ, "PYTHONIOENCODING":"utf-8", "PYTHONUTF8":"1", "PYTHONPATH":manifest["source"]["directory"]}
    with (root/"build.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable,"-m","backend.evaluation.rag_challenge","build","--experiment",str(root)],
            cwd=manifest["source"]["directory"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in process.stdout:
            log.write(line); log.flush(); print(line.rstrip(), flush=True)
        if process.wait() != 0: raise RuntimeError("challenge_index_build_failed")
    # Only corpus construction was performed above; no queries have been embedded or ranked.
    corpus = {"documents":ab.sql_rows(root/"rag.db","rag_documents","document_id"),
              "chunks":ab.sql_rows(root/"rag.db","rag_chunks","chunk_id")}
    sources = {d["source_id"]:d["document_id"] for d in corpus["documents"]}
    dataset = {k:v for k,v in spec.items() if k != "documents"}
    for case in dataset["cases"]:
        case["goldDocumentIds"] = [sources[s] for s in case["goldSources"]]
    ab.write(root/"corpus.json",corpus); ab.write(root/"queries.json",dataset)
    manifest.update(corpusHash=ab.digest(corpus), datasetHash=ab.digest(dataset))
    ab.write(root/"experiment.json",manifest)
    ab.write(root/"construction.json", {"status":"sealed_before_retrieval", "specHash":ab.digest(spec),
        "documents":len(corpus["documents"]), "chunks":len(corpus["chunks"]), "queries":len(dataset["cases"])})
    print("SEALED "+str(root), flush=True)
    return root


def build(root):
    root, manifest = ab.setup(root)
    spec = ab.read(root/"challenge_spec.json")
    if ab.digest(spec) != manifest["challengeSpecHash"]: raise ValueError("challenge_spec_changed")
    from backend.rag.v2 import document_repository as repo, dense_index, sparse_index
    from backend.rag.v2.models import RagDocument, RagChunk
    from backend.rag.v2.providers import get_embedding_provider
    provider = get_embedding_provider()
    if provider.is_degraded() or provider.get_resolved_model_name() != manifest["models"]["embedding"]:
        raise ValueError("real_embedding_required")
    provider._model.float()
    collection = dense_index.get_active_collection_name()
    # Rebuild original vectors too, to keep all corpus/query vectors at one precision.
    originals = ab.read(root/"corpus.json")["chunks"]
    for i, row in enumerate(originals):
        chunk = RagChunk(**row)
        vector = provider.embed_documents([chunk.contextual_content])
        if provider.is_degraded() or not dense_index.upsert_chunks([chunk],vector,collection_name=collection):
            raise ValueError("original_chunk_reembedding_failed")
        print(f"reencoded original {i+1}/{len(originals)}",flush=True)
    timings = []
    for item in spec["documents"]:
        identity = hashlib.sha256(item["source"].encode()).hexdigest()[:16]
        document = RagDocument(document_id="stress_doc_"+identity, source_id=item["source"],
            title=item["title"], content=item["content"], event_type=item["event"],
            road_name=item["site"], grounding_scope="SYNTHETIC_EVALUATION_ONLY")
        chunk = RagChunk(chunk_id="stress_chunk_"+identity, document_id=document.document_id,
            raw_content=item["content"], contextual_content=item["content"],
            section_path=item["title"], event_type=item["event"], road_name=item["site"],
            grounding_scope="SYNTHETIC_EVALUATION_ONLY")
        t = time.perf_counter()
        vector = provider.embed_documents([chunk.contextual_content])
        if provider.is_degraded() or len(vector)!=1 or len(vector[0])!=1024: raise ValueError("embedding_invalid")
        repo.upsert_document(document); repo.upsert_chunks([chunk])
        if not dense_index.upsert_chunks([chunk],vector,collection_name=collection): raise ValueError("dense_write_failed")
        if not sparse_index.upsert_chunks_fts([chunk]): raise ValueError("sparse_write_failed")
        timings.append((time.perf_counter()-t)*1000)
        print(f"indexed {len(timings)}/{len(spec['documents'])} {item['source']}",flush=True)
    ab.write(root/"index_build.json", {"documentsAdded":len(timings),"originalChunksReencoded":len(originals),"embeddingResolved":provider.get_resolved_model_name(),
        "degraded":provider.is_degraded(),"dimension":provider.get_dimension(),"perDocumentMs":timings,
        "embeddingDtype":str(next(provider._model.parameters()).dtype),"collection":collection})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","build"))
    parser.add_argument("--base"); parser.add_argument("--experiment")
    parser.add_argument("--output",default="D:/TrafficMind-cache/rag-ablation")
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.base,args.output)
    else: build(args.experiment)


if __name__ == "__main__": main()
