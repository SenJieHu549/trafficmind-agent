"""Strict historical model-I/O prefix replay, followed by a live repair call.

Conditional recovery experiment only; never a fresh end-to-end model benchmark.
All eligible historical failures in the named prior campaign are retained.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from pydantic import ValidationError

# Serving configuration is import-time: isolate before importing the client,
# evaluation helpers or any module that can import backend.config.
if __name__=="__main__" and len(sys.argv)>2 and sys.argv[1]=="worker":
    _request=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    _root=Path(_request["directory"]).resolve()
    for _key,_name in {"TRAFFICMIND_DB_PATH":"events.db","RAG_V2_DB_PATH":"rag.db",
        "RAG_V2_FTS_PATH":"fts.db","RAG_V2_VECTOR_DB_PATH":"vectors"}.items():
        os.environ[_key]=str(_root/_name)

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.experiments import freeze_source, verify_source
from backend.evaluation.v2.providers import content_hash, safe_value


def prepare(parent,label="historical_replay"):
    from backend import config
    from backend.agent.llm_accident.contracts import FinalAnalysis, EvidenceLedger
    from backend.agent.llm_accident.output_validation import validation_feedback
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    from backend.agent.llm_accident.citations import resolve_citations
    parent=Path(parent).resolve(); root=parent/label;root.mkdir(exist_ok=False)
    previous=Path("D:/TrafficMind-cache/agent-iteration/iteration_20261108T133501_75648c2d")
    manifest={"scope":"conditional recovery of historical natural validation failures, not fresh end-to-end quality",
        "sourceCampaign":str(previous),"cases":[],"excluded":[],"arms":{"baseline":0,"candidate":1},
        "policy":"Freeze every historical invalid_final_output passing the current repair-eligibility check; replay all original requests/responses exactly by request hash. No synthetic corruption. Only the additional repair completion is live. Equal 4-call global cap includes replayed calls."}
    for row in it.read(previous/"iteration_report.json")["rows"]:
        if row["failureReason"]!="invalid_final_output":continue
        original=Path(row["recordPath"]).parent; record=it.read(original/"record.json")
        cassette=it.read(original/"cassette.json");content=cassette[-1]["reply"]["message"].get("content","")
        snapshot=record["evidenceSnapshot"];ledger=EvidenceLedger(snapshot["eventId"],snapshot["asOf"])
        for e in snapshot["evidence"]:
            assert ledger.add(e["kind"],e["sourceId"],e["data"],e["origins"][0])==e["evidenceId"]
        compact=record["runtimeTrace"]["citationPolicy"]!="canonical-evidence-id-1"
        try:FinalAnalysis.model_validate_json(content)
        except ValidationError as exc:
            feedback=validation_feedback(exc,content,ledger,compact=compact,
                validate_actions=AccidentAgentRuntime._validate_actions,resolve_refs=resolve_citations)
        else:raise ValueError("historical_failure_no_longer_a_validation_failure")
        item={"key":f"H{len(manifest['cases'])+1:02}","originalRecord":str(original/"record.json"),
            "request":it.read(original/"request.json"),"cassette":cassette,
            "citationPolicy":"compact" if compact else "canonical","eligibility":feedback,
            "originalRecordHash":content_hash(record),"prefixHash":content_hash(cassette)}
        if feedback["repairEligible"]:manifest["cases"].append(item)
        else:manifest["excluded"].append(item)
    assert manifest["cases"]
    manifest["source"]=freeze_source(it.PROJECT,root/"source")
    # Reporting additions may differ; generation, guard and transport code must match the main campaign.
    main=it.read(parent/"experiment.json")
    for file in ("runtime.py","contracts.py","output_validation.py","client.py","integration.py"):
        relative=Path("backend/agent/llm_accident")/file
        assert (Path(main["baseline"]["directory"])/relative).read_bytes()==(root/"source"/relative).read_bytes()
    manifest["manifestHash"]=content_hash(manifest)
    it.write(root/"manifest.json",manifest); print(str(root),flush=True)
    return root


def worker(request_path,spec_path):
    from backend.evaluation.v2 import worker as normal
    from backend.agent.llm_accident.client import AccidentLLMClient
    from backend.agent.llm_accident import runtime
    spec=it.read(spec_path); prefix=spec["cassette"]
    class PrefixClient:
        def __init__(self,*args,**kwargs):
            self.entries=[];self.offset=0;self.fresh=0;self.live=None
        async def complete(self,messages,tools,**kwargs):
            request={"messages":deepcopy(messages),"tools":deepcopy(tools),**kwargs}
            if self.offset<len(prefix):
                entry=deepcopy(prefix[self.offset])
                if content_hash(request)!=entry["requestHash"]:raise RuntimeError("historical_request_hash_mismatch")
                self.offset+=1;self.entries.append(entry)
                if entry.get("error"):raise RuntimeError("historical_prefix_error")
                return deepcopy(entry["reply"])
            self.fresh+=1
            if kwargs["allow_tools"]:raise RuntimeError("historical_recovery_must_disable_tools")
            entry={"requestHash":content_hash(request),"request":safe_value(request)};self.entries.append(entry)
            if self.live is None:self.live=AccidentLLMClient()
            try:
                reply=await self.live.complete(messages,tools,**kwargs)
                entry["reply"]=safe_value(reply);return reply
            except Exception as exc:
                entry["error"]=type(exc).__name__;raise
        def assert_consumed(self):
            if self.offset!=len(prefix):raise RuntimeError("historical_prefix_unconsumed")
    client=PrefixClient(); original_runtime=runtime.AccidentAgentRuntime
    def factory(**kwargs):return original_runtime(citation_policy=spec["citationPolicy"],**kwargs)
    with patch.object(normal,"EvaluationClient",lambda *a,**k:client),patch.object(runtime,"AccidentAgentRuntime",factory):
        sys.argv=[sys.argv[0],request_path];normal.main()
    directory=Path(it.read(request_path)["directory"])
    record=it.read(directory/"record.json")
    record.update(transport="historical_prefix_then_live",freshEndToEndQualityEligible=False,
        historicalPrefixCallsReplayed=client.offset,freshRecoveryRequestAttempts=client.fresh,
        historicalPrefixFullyMatched=client.offset==len(prefix),historicalOriginalRecord=spec["originalRecord"])
    it.write(directory/"record.json",record)


def run(root):
    from backend import config  # Load original local credentials into child environment only.
    root=Path(root); manifest=it.read(root/"manifest.json")
    saved=deepcopy(manifest);expected=saved.pop("manifestHash");assert content_hash(saved)==expected
    verify_source(manifest["source"])
    if (root/"state.json").exists():raise ValueError("historical_campaign_already_started")
    it.write(root/"state.json",{"status":"running"});rows=[]
    for item in manifest["cases"]:
        for arm,repairs in manifest["arms"].items():
            directory=root/(item["key"]+"_"+arm);directory.mkdir()
            request=deepcopy(item["request"]);request.update(directory=str(directory),transport="live")
            request.pop("cassette",None)
            it.write(directory/"request.json",request);it.write(directory/"historical_spec.json",item)
            env={**os.environ,"ACCIDENT_OUTPUT_REPAIR_ATTEMPTS":str(repairs),"PYTHONIOENCODING":"utf-8",
                "PYTHONUTF8":"1","PYTHONPATH":manifest["source"]["directory"]}
            with (directory/"worker.log").open("wb") as log:
                process=subprocess.run([sys.executable,"-m","backend.evaluation.v2.historical_repair","worker",
                    str(directory/"request.json"),str(directory/"historical_spec.json")],
                    cwd=manifest["source"]["directory"],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            if process.returncode:raise RuntimeError("historical_worker_failed")
            record=it.read(directory/"record.json")
            assert record["historicalPrefixFullyMatched"]
            trace=record["runtimeTrace"]
            rows.append({"key":item["key"],"arm":arm,"caseId":record["caseId"],"passed":record["passed"],
                "replayedCalls":record["historicalPrefixCallsReplayed"],"freshRecoveryRequests":record["freshRecoveryRequestAttempts"],
                "repairValidated":trace["outputRepairValidated"],"failureReason":trace["failureReason"],
                "repairTokens":sum(c.get("usage",{}).get("total_tokens",0) for c in trace["llmCalls"] if c.get("purpose")=="output_repair"),
                "recordPath":str(directory/"record.json")})
            it.write(root/"results.json",{"scope":manifest["scope"],"rows":rows})
            print(f"historical {item['key']} {arm} passed={record['passed']} repair={trace['outputRepairAttempts']}",flush=True)
    it.write(root/"state.json",{"status":"complete"})


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":prepare(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else "historical_replay")
    elif action=="run":
        try:run(sys.argv[2])
        except Exception:
            it.write(Path(sys.argv[2])/"state.json",{"status":"failed"});raise
    else:worker(sys.argv[2],sys.argv[3])
