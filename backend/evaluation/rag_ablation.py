"""Frozen retrieval-only A/B/C pilot using real local providers and isolated stores.

A: Dense 24 -> pool 12; B: Dense 12 + BM25 12 -> RRF -> pool 12;
C: rerank the exact B pool. Metrics collapse chunks to unique document IDs.
No Agent generation, query rewriting, or final EvidencePolicy authority boosts.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sql_rows(path, table, order):
    with sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}")]


def sqlite_snapshot(source, target):
    with sqlite3.connect(Path(source).resolve().as_uri()+"?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)


def source_fingerprint(paths):
    return {"documents": digest(sql_rows(paths["rag"], "rag_documents", "document_id")),
            "chunks": digest(sql_rows(paths["rag"], "rag_chunks", "chunk_id")),
            "versions": digest(sql_rows(paths["rag"], "rag_index_versions", "version_id")),
            "fts": digest(sql_rows(paths["fts"], "rag_fts", "chunk_id")),
            "vectorFiles": {p.relative_to(paths["vector"]).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(paths["vector"]).rglob("*"))
                if p.is_file() and not p.name.endswith(("-shm", "-wal"))}}


def freeze(output, dataset, *, reuse_retrieval=None):
    from backend import config  # Load local non-secret settings, never serialize .env.
    from backend.rag.v2 import config as rc
    from backend.evaluation.v2.experiments import freeze_source
    paths = {"rag": rc.RAG_V2_DB_PATH, "fts": rc.RAG_V2_FTS_PATH, "vector": rc.RAG_V2_VECTOR_DB_PATH}
    before = source_fingerprint(paths)
    root = Path(output).resolve() / ("rag_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    sqlite_snapshot(paths["rag"], root/"rag.db")
    sqlite_snapshot(paths["fts"], root/"fts.db")
    shutil.copytree(paths["vector"], root/"vectors", ignore=shutil.ignore_patterns("*-shm", "*-wal"))
    sqlite_snapshot(Path(paths["vector"])/"chroma.sqlite3", root/"vectors/chroma.sqlite3")
    if source_fingerprint(paths) != before:
        raise ValueError("source_changed_during_snapshot")
    corpus = {"documents": sql_rows(root/"rag.db", "rag_documents", "document_id"),
              "chunks": sql_rows(root/"rag.db", "rag_chunks", "chunk_id")}
    write(root/"corpus.json", corpus)
    data = read(dataset)
    sources = {d["source_id"]: d for d in corpus["documents"]}
    if len(sources) != len(corpus["documents"]):
        raise ValueError("duplicate_source_id")
    if len({c["caseId"] for c in data["cases"]}) != len(data["cases"]):
        raise ValueError("duplicate_case_id")
    for case in data["cases"]:
        gold = case["goldSources"]
        if not gold or len(gold) != len(set(gold)) or not set(gold) <= set(sources):
            raise ValueError("invalid_gold:"+case["caseId"])
        case["goldDocumentIds"] = [sources[s]["document_id"] for s in gold]
    write(root/"queries.json", data)
    protocol = {"scope": "retrieval-only synthetic source-checked pilot; no independent annotation or external validation",
        "k": 5, "poolSize": 12, "denseBudgetA": 24, "denseBudgetB": 12, "sparseBudgetB": 12,
        "rrfK": 60, "rrfWindow": 40, "rerankTextLimit": 500,
        "primaryMetric": "macro document Recall@5 across all frozen queries; failed rows score zero",
        "secondaryMetrics": ["Hit@5", "Precision@5", "MRR@5", "binary nDCG@5", "candidateRecall"],
        "documentCollapse": "Keep first occurrence of each document in ranked chunk pool, then take 5; pad precision denominator to 5",
        "controls": "Same corpus, query, status/effective-date eligibility and final K; B and C use identical chunk pool. No source/risk/region filtering differences, query rewrite, Structured channel or authority/diversity boosts.",
        "budgetScope": "A exposes 24 dense candidates, B exposes 12 dense + 12 sparse. Dense24 is computed once and shared; this controls ranking inputs, not equal compute or end-to-end latency.",
        "tuningPolicy": "One frozen query set and fixed ranking settings; retain all regressions, no relevance/parameter tuning after results",
        "rerankDtype": "float32",
        "annotationPolicy": data["annotationPolicy"], "asOf": "2026-11-08T00:00:00+00:00"}
    cache = Path(rc.RAG_MODEL_CACHE_DIR)
    model_files = {}
    for folder in ("models--Qwen--Qwen3-Embedding-0.6B", "models--Qwen--Qwen3-Reranker-0.6B"):
        for p in sorted((cache/folder/"snapshots").glob("*/*")):
            if p.is_file():
                model_files[p.relative_to(cache).as_posix()] = {"bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    reuse = None
    if reuse_retrieval:
        previous = Path(reuse_retrieval).resolve()
        parent = read(previous/"experiment.json")
        if parent["datasetHash"] != digest(data) or parent["corpusHash"] != digest(corpus):
            raise ValueError("reused_retrieval_input_mismatch")
        for key in ("k","poolSize","denseBudgetA","denseBudgetB","sparseBudgetB","rrfK","rrfWindow","rerankTextLimit","asOf"):
            if parent["protocol"][key] != protocol[key]: raise ValueError("reused_retrieval_protocol_mismatch")
        completed=read(previous/"retrieval_rows.json")
        if len(completed)!=len(data["cases"]): raise ValueError("reused_retrieval_incomplete")
        for filename in ("retrieval_rows.json","preflight.json"):
            shutil.copy2(previous/filename,root/filename)
        reuse={"experiment":str(previous),"retrievalRowsHash":digest(completed),
            "reason":"CPU reranking took 183.514 seconds for the first 12 pairs; previous attempt interrupted and preserved. Explicit float32 CPU execution recorded in a new source/protocol snapshot. Frozen labels and A/B ranks unchanged; all C queries, including the first, rerun under the new execution precision."}
    manifest = {"schemaVersion": "trafficmind-rag-ablation-1", "root": str(root),
        "protocol": protocol, "protocolHash": digest(protocol), "datasetHash": digest(data), "corpusHash": digest(corpus),
        "sourcePaths": paths, "sourceFingerprintBefore": before, "retrievalReuse": reuse,
        "models": {"embedding": rc.RAG_EMBEDDING_MODEL, "reranker": rc.RAG_RERANKER_MODEL,
                   "cache": str(cache), "device": "cpu", "torchThreads": 4, "files": model_files},
        "source": freeze_source(PROJECT, root/"source")}
    write(root/"experiment.json", manifest)
    print(str(root), flush=True)
    return root


def setup(root):
    root = Path(root).resolve()
    m = read(root/"experiment.json")
    os.environ.update({"TRAFFIC_DB_PATH": str(root/"traffic.db"), "RAG_V2_DB_PATH": str(root/"rag.db"),
        "RAG_V2_FTS_PATH": str(root/"fts.db"), "RAG_V2_VECTOR_DB_PATH": str(root/"vectors"),
        "RAG_EMBEDDING_MODEL": m["models"]["embedding"], "RAG_RERANKER_MODEL": m["models"]["reranker"],
        "RAG_MODEL_CACHE_DIR": m["models"]["cache"], "RAG_DEVICE": "cpu", "RAG_RERANK_BATCH_SIZE": "1",
        "RAG_ALLOW_MODEL_DOWNLOAD": "false", "RAG_ALLOW_HASH_FALLBACK": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "ANONYMIZED_TELEMETRY": "false",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    from backend import config
    import torch
    torch.set_num_threads(4)
    if digest(read(root/"queries.json")) != m["datasetHash"] or digest(read(root/"corpus.json")) != m["corpusHash"]:
        raise ValueError("frozen_input_changed")
    return root, m


def eligible_documents(corpus, as_of):
    now = datetime.fromisoformat(as_of)
    def parse(value):
        v = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return sorted(d["document_id"] for d in corpus["documents"] if d["status"] == "active"
        and (not d["effective_from"] or parse(d["effective_from"]) <= now)
        and (not d["effective_to"] or parse(d["effective_to"]) > now))


def retrieve(root):
    root, m = setup(root)
    if (root/"retrieval_rows.json").exists(): raise ValueError("retrieval_already_started")
    from backend.rag.v2.providers import get_embedding_provider
    from backend.rag.v2 import dense_index, sparse_index
    from backend.rag.v2.rrf import reciprocal_rank_fusion
    corpus, data, p = read(root/"corpus.json"), read(root/"queries.json"), m["protocol"]
    allowed = eligible_documents(corpus, p["asOf"])
    if any(not set(c["goldDocumentIds"]) <= set(allowed) for c in data["cases"]):
        raise ValueError("gold_outside_eligible_corpus")
    provider = get_embedding_provider()
    if provider.is_degraded() or provider.get_resolved_model_name() != m["models"]["embedding"]:
        raise ValueError("real_embedding_required")
    if p.get("embeddingDtype") == "float32":
        provider._model.float()
    collection_name = dense_index.get_active_collection_name()
    col = dense_index.get_client().get_collection(collection_name, embedding_function=None)
    actual = col.get(include=["metadatas", "embeddings"])
    expected_chunks = {c["chunk_id"] for c in corpus["chunks"]}
    if set(actual["ids"]) != expected_chunks: raise ValueError("vector_and_sqlite_chunks_differ")
    if len(actual["embeddings"][0]) != provider.get_dimension(): raise ValueError("vector_dimension_mismatch")
    fts = sql_rows(root/"fts.db", "rag_fts", "chunk_id")
    if {c["chunk_id"] for c in fts} != expected_chunks: raise ValueError("fts_and_sqlite_chunks_differ")
    write(root/"preflight.json", {"collection": collection_name, "vectors": len(actual["ids"]),
        "dimension": provider.get_dimension(), "ftsRows": len(fts), "eligibleDocuments": len(allowed),
        "embeddingResolved": provider.get_resolved_model_name(), "degraded": provider.is_degraded()})
    fallback_calls = []
    original = sparse_index._search_bm25_python
    def observe_fallback(*args, **kwargs):
        fallback_calls.append(True)
        return original(*args, **kwargs)
    sparse_index._search_bm25_python = observe_fallback
    rows = []
    for case in data["cases"]:
        row = {"caseId": case["caseId"], "category": case["category"], "query": case["query"],
               "goldDocumentIds": case["goldDocumentIds"]}
        try:
            t = time.perf_counter(); vector = provider.embed_query(case["query"])
            query_ms = (time.perf_counter()-t)*1000
            if provider.is_degraded(): raise ValueError("embedding_degraded")
            t = time.perf_counter()
            dense = dense_index.search_dense(vector, p["denseBudgetA"], where={"document_id":{"$in":allowed}}, collection_name=collection_name)
            dense_ms = (time.perf_counter()-t)*1000
            if not dense: raise ValueError("dense_search_returned_no_candidates")
            count = len(fallback_calls); t = time.perf_counter()
            sparse = sparse_index.search_sparse(case["query"], p["sparseBudgetB"], allowed_document_ids=allowed)
            sparse_ms = (time.perf_counter()-t)*1000
            hybrid = reciprocal_rank_fusion([deepcopy(dense[:p["denseBudgetB"]]), deepcopy(sparse)], k=p["rrfK"], window=p["rrfWindow"])[:p["poolSize"]]
            row.update(status="ok", A=dense[:p["poolSize"]], B=hybrid, BPoolHash=digest(hybrid),
                sparseBackend="python_bm25_fallback" if len(fallback_calls)>count else "fts5_bm25",
                timingsMs={"queryEmbedding": query_ms, "sharedDense24": dense_ms, "sparse12": sparse_ms},
                queryEmbeddingHash=digest(vector), denseCount=len(dense), sparseCount=len(sparse))
        except Exception as exc:
            row.update(status="error", errorType=type(exc).__name__, error=str(exc), A=[], B=[])
        rows.append(row); write(root/"retrieval_rows.json", rows)
        print(f"retrieval {case['caseId']} {row['status']}", flush=True)


def rerank(root):
    root, m = setup(root)
    if (root/"rerank_rows.json").exists(): raise ValueError("rerank_already_started")
    from backend.rag.v2.providers import get_reranker_provider
    provider = get_reranker_provider()
    if provider.is_degraded() or provider.get_model_name() != m["models"]["reranker"]:
        raise ValueError("real_reranker_required")
    import torch
    model = provider._model.model
    original_dtype = str(next(model.parameters()).dtype)
    if m["protocol"].get("rerankDtype") == "float32":
        model.float()
    actual_dtype = str(next(model.parameters()).dtype)
    write(root/"reranker_execution.json", {"loadedDtype":original_dtype,"executionDtype":actual_dtype,
        "torchThreads":torch.get_num_threads(),"batchSize":1,"queryCount":len(read(root/"queries.json")["cases"])})
    print(f"reranker precision {original_dtype} -> {actual_dtype}",flush=True)
    rows = []
    originals = read(root/"retrieval_rows.json")
    if len(originals) != len(read(root/"queries.json")["cases"]): raise ValueError("incomplete_retrieval")
    for item in originals:
        row = {"caseId": item["caseId"], "inputPoolHash": digest(item["B"]), "C": []}
        t = time.perf_counter()
        try:
            if item["status"] != "ok": raise ValueError("upstream_retrieval_failed")
            scores = provider.rerank(item["query"], [c["content"][:m["protocol"]["rerankTextLimit"]] for c in item["B"]], top_k=m["protocol"]["poolSize"])
            if provider.is_degraded() or len(scores) != len(item["B"]) or not all(math.isfinite(s) for s in scores):
                raise ValueError("reranker_degraded_or_invalid_scores")
            ranked = [{**c, "ablationRerankScore": float(score)} for c,score in zip(deepcopy(item["B"]),scores)]
            ranked.sort(key=lambda c:c["ablationRerankScore"], reverse=True)
            row.update(status="ok", C=ranked, scores=scores)
        except Exception as exc:
            row.update(status="error", errorType=type(exc).__name__, error=str(exc))
        row["rerankMs"] = (time.perf_counter()-t)*1000
        rows.append(row); write(root/"rerank_rows.json", rows)
        print(f"rerank {item['caseId']} {row['status']} {row['rerankMs']:.0f}ms", flush=True)
    write(root/"reranker_identity.json", {"model": provider.get_model_name(), "degraded": provider.is_degraded()})


def unique_documents(candidates):
    return list(dict.fromkeys(c["document_id"] for c in candidates))


def score(candidates, gold, k=5):
    if not gold or len(gold) != len(set(gold)): raise ValueError("nonempty_unique_gold_required")
    docs = unique_documents(candidates)
    top = docs[:k]; relevant = set(gold)
    hits = len(set(top)&relevant)
    dcg = sum(1/math.log2(i+2) for i,d in enumerate(top) if d in relevant)
    ideal = sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {"recallAt5": hits/len(relevant), "hitAt5": float(hits>0), "precisionAt5": hits/k,
        "mrrAt5": next((1/(i+1) for i,d in enumerate(top) if d in relevant),0), "ndcgAt5": dcg/ideal,
        "candidateRecall": len(set(docs)&relevant)/len(relevant), "topDocumentIds": top,
        "candidateDocumentCount": len(docs)}


def aggregate(rows):
    keys = ("recallAt5","hitAt5","precisionAt5","mrrAt5","ndcgAt5","candidateRecall")
    return {"queries":len(rows), "errors":sum(r["status"]!="ok" for r in rows),
        **{key:sum(r["metrics"][key] for r in rows)/len(rows) for key in keys}}


def report(root):
    root=Path(root); m=read(root/"experiment.json")
    before, after = read(root/"retrieval_rows.json"), read(root/"rerank_rows.json")
    assert len(before)==len(after)==len(read(root/"queries.json")["cases"])
    after={r["caseId"]:r for r in after}; rows=[]
    for original in before:
        reranked=after[original["caseId"]]
        if original["status"]=="ok" and reranked["inputPoolHash"] != original["BPoolHash"]:
            raise ValueError("rerank_input_pool_changed")
        if reranked["status"]=="ok" and Counter(c["chunk_id"] for c in original["B"]) != Counter(c["chunk_id"] for c in reranked["C"]):
            raise ValueError("rerank_changed_candidate_membership")
        for arm in ("A","B","C"):
            selected = reranked if arm=="C" else original
            rows.append({"caseId":original["caseId"], "category":original["category"], "arm":arm,
                "status":selected["status"], "metrics":score(selected.get(arm,[]), original["goldDocumentIds"])})
    arms={a:aggregate([r for r in rows if r["arm"]==a]) for a in ("A","B","C")}
    by_category={category:{a:aggregate([r for r in rows if r["arm"]==a and r["category"]==category])
        for a in arms} for category in sorted({r["category"] for r in rows})}
    comparisons={}
    for left,right in (("A","B"),("B","C"),("A","C")):
        l={r["caseId"]:r for r in rows if r["arm"]==left}; rr={r["caseId"]:r for r in rows if r["arm"]==right}
        comparisons[left+"_to_"+right]={"recallDeltaPercentagePoints":100*(arms[right]["recallAt5"]-arms[left]["recallAt5"]),
            "mrrDelta":arms[right]["mrrAt5"]-arms[left]["mrrAt5"],
            "ndcgDelta":arms[right]["ndcgAt5"]-arms[left]["ndcgAt5"],
            "rankingImproved":[cid for cid in l if rr[cid]["metrics"]["ndcgAt5"]-l[cid]["metrics"]["ndcgAt5"]>1e-12],
            "rankingRegressed":[cid for cid in l if l[cid]["metrics"]["ndcgAt5"]-rr[cid]["metrics"]["ndcgAt5"]>1e-12],
            "recallImproved":[cid for cid in l if rr[cid]["metrics"]["recallAt5"]>l[cid]["metrics"]["recallAt5"]],
            "recallRegressed":[cid for cid in l if rr[cid]["metrics"]["recallAt5"]<l[cid]["metrics"]["recallAt5"]]}
    unchanged=source_fingerprint(m["sourcePaths"])==m["sourceFingerprintBefore"]
    result={"schemaVersion":m["schemaVersion"],"protocol":m["protocol"], "protocolHash":m["protocolHash"],
        "datasetHash":m["datasetHash"],"corpusHash":m["corpusHash"],"sourceUnchanged":unchanged,
        "arms":arms,"byCategory":by_category,"comparisons":comparisons,"rows":rows,
        "sparseBackends":dict(Counter(r.get("sparseBackend","error") for r in before))}
    write(root/"ablation_report.json",result)
    corpus = read(root/"corpus.json")
    lines=["# RAG A/B/C retrieval-only experiment","",f"{len(before)} synthetic queries; {len(corpus['documents'])} documents / {len(corpus['chunks'])} chunks. No independent annotation or end-to-end Agent quality claim.","",m["protocol"]["scope"],"",
        "| Arm | Queries | Errors | Recall@5 | MRR@5 | Hit@5 | nDCG@5 | Candidate recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a,v in arms.items(): lines.append(f"| {a} | {v['queries']} | {v['errors']} | {v['recallAt5']:.4f} | {v['mrrAt5']:.4f} | {v['hitAt5']:.4f} | {v['ndcgAt5']:.4f} | {v['candidateRecall']:.4f} |")
    lines.extend(["","## Paired comparisons","","```json",json.dumps(comparisons,ensure_ascii=False,indent=2),"```","",f"Source stores unchanged: {unchanged}."])
    (root/"ablation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def run(root):
    from backend.evaluation.v2.experiments import verify_source
    root=Path(root).resolve(); m=read(root/"experiment.json")
    if (root/"state.json").exists(): raise ValueError("campaign_already_started_no_silent_rerun")
    verify_source(m["source"])
    write(root/"state.json",{"status":"running"})
    try:
        if m.get("retrievalReuse") and digest(read(root/"retrieval_rows.json")) != m["retrievalReuse"]["retrievalRowsHash"]:
            raise ValueError("reused_retrieval_changed")
        for phase in (("rerank",) if m.get("retrievalReuse") else ("retrieve","rerank")):
            env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1","PYTHONPATH":m["source"]["directory"]}
            command=[sys.executable,"-m","backend.evaluation.rag_ablation",phase,"--experiment",str(root)]
            with (root/(phase+".log")).open("w",encoding="utf-8") as log:
                process=subprocess.Popen(command,cwd=m["source"]["directory"],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8")
                for line in process.stdout:
                    log.write(line); log.flush(); print(line.rstrip(),flush=True)
                if process.wait()!=0: raise RuntimeError("phase_failed:"+phase)
        verify_source(m["source"])
        result=report(root)
        if not result["sourceUnchanged"]: raise ValueError("source_stores_changed")
        write(root/"state.json",{"status":"complete"})
    except Exception:
        write(root/"state.json",{"status":"failed"})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","retrieve","rerank","report"))
    p.add_argument("--output",default="D:/TrafficMind-cache/rag-ablation")
    p.add_argument("--dataset",default=str(PROJECT/"backend/evaluation/v2/datasets/rag_ablation_queries.json"))
    p.add_argument("--experiment")
    p.add_argument("--reuse-retrieval")
    a=p.parse_args()
    if a.action=="freeze": freeze(a.output,a.dataset,reuse_retrieval=a.reuse_retrieval)
    elif not a.experiment: p.error("--experiment required")
    else: globals()[a.action](a.experiment)


if __name__=="__main__": main()
