"""Frozen A/B comparison of fail-fast vs one output repair, with equal caps.

Uses the existing live Agent/DAG/Planning/Workflow and isolated V2 runner.
Synthetic operational fixtures; not a real-world traffic accuracy benchmark.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from backend.evaluation.v2 import iteration as it
from backend.evaluation.v2.experiments import freeze_source, verify_source
from backend.evaluation.v2.providers import content_hash

PROJECT = it.PROJECT


def freeze(output):
    holdout=PROJECT/"backend/evaluation/v2/datasets/output_repair_holdout.json"
    root=it.freeze(output,holdout,workers=2,holdout_repeats=3)
    manifest=it.read(root/"experiment.json")
    regression=it.read(root/"regression.json")
    previous=it.read(PROJECT/"backend/evaluation/v2/datasets/iteration_holdout.json")
    for case in previous["cases"]:
        case=deepcopy(case)
        case["event"]={**previous["eventDefaults"],**case.get("event",{})}
        case["knowledge"]=deepcopy(case.get("knowledge",previous["knowledge"]))
        case["split"]="regression"
        regression["cases"].append(case)
    regression.update(datasetVersion="output-repair-regression-20261109-1",
        purpose="20_previously_seen_scenarios_including_former_holdout_now_regression")
    it.write(root/"regression.json",regression)
    manifest["datasets"]["regression"].update(hash=content_hash(regression),caseIds=[c["caseId"] for c in regression["cases"]])
    manifest["candidate"]=freeze_source(PROJECT,root/"candidate_source")
    manifest["policyVariants"]={"baseline":{"maxOutputRepairs":0},"candidate":{"maxOutputRepairs":1}}
    assert manifest["baseline"]["sourceManifestHash"]==manifest["candidate"]["sourceManifestHash"]
    cases=[c for split in manifest["datasets"] for c in it.read(root/(split+".json"))["cases"]]
    assert len(cases)==52 and len({c["caseId"] for c in cases})==52 and len({c["question"] for c in cases})==52
    manifest["protocol"].update(candidateScope="Only max_output_repairs 0 vs 1; identical frozen source, canonical citations, models, prompts before failure, max 4 LLM calls, tool limits and deadlines. Repair disables tool permission and revalidates all original guards.",
        dataScope="52 synthetic scenarios: 20 historical development + 20 seen regression + 12 fresh compound holdout; 152 scheduled trials across two arms. Explicit provider/tool faults are labelled separately. Not external validation.",
        secondaryMetrics=["first-final schema compliance with coverage","task recovery among repair-eligible first-final failures","unexpected fallback","paired regressions","P50/P95 total execution time","token and repair-call usage with coverage"],
        holdoutPolicy="Both policy/source variants and all labels frozen before any arm execution. One run per scheduled pair; retain failures. New holdout each has 3 trials, not 36 independent scenarios.",
        interventionPolicy="No injected malformed model output in live quality comparison; separate deterministic tests verify recovery mechanics. Explicit provider/tool faults remain labelled.",
        adoptionPolicy="Adopt only if regression and holdout behavior rates do not regress, unexpected fallbacks do not increase, no observed safety failures, AND holdout pass rate strictly improves. Otherwise keep default zero; no forced gain.")
    manifest["protocolHash"]=content_hash(manifest["protocol"])
    manifest["schemaVersion"]="trafficmind-output-repair-campaign-1"
    it.write(root/"experiment.json",manifest)
    return root


def audit(root):
    root=Path(root); manifest=it.read(root/"experiment.json")
    for source in ("baseline","candidate"):verify_source(manifest[source])
    assert manifest["baseline"]["sourceManifestHash"]==manifest["candidate"]["sourceManifestHash"]
    assert content_hash(manifest["protocol"])==manifest["protocolHash"]
    for split,data in manifest["datasets"].items():
        assert content_hash(it.read(data["path"]))==data["hash"]
    report=it.report(root)
    records=[]
    for name,arm in manifest["arms"].items():
        assert arm["status"]=="complete"
        expected=manifest["policyVariants"][arm["version"]]["maxOutputRepairs"]
        for trial in arm["trials"]:
            path=Path(trial["runDirectory"])/f"{trial['caseId']}-llm_tools/record.json"
            record=it.read(path); trace=record.get("runtimeTrace",{})
            assert trace["limits"]["max_output_repairs"]==expected
            assert trace["limits"]["max_llm_calls"]==4
            assert len(trace["llmCalls"])<=4 and trace["outputRepairAttempts"]<=expected
            repair_calls=[c for c in trace["llmCalls"] if c.get("purpose")=="output_repair"]
            assert len(repair_calls)==trace["outputRepairAttempts"]
            assert all(not c["toolsAllowed"] for c in repair_calls)
            assert record["passed"]==all(c["passed"] for c in record["checks"])
            cassette=it.read(path.parent/"cassette.json")
            assert len(cassette)==len(trace["llmCalls"])==record["modelRequests"]
            for call,entry in zip(trace["llmCalls"],cassette):
                assert content_hash(entry["request"])==entry["requestHash"]
                if call.get("purpose")=="output_repair":assert entry["request"]["allow_tools"] is False
            records.append((name,trial["caseId"],trial["repetition"]))
    assert len(records)==len(set(records))==152
    assert all(r["pairedTrials"]==manifest["protocol"]["repetitions"][split]*len(manifest["datasets"][split]["caseIds"])
        for split,r in report["comparisons"].items())
    totals={"status":"PASS","trials":len(records),"sourceDatasetProtocolAndCassetteHashesVerified":True,
        "equalGlobalBudgetsVerified":True,"repairToolsDisabledVerified":True,
        "providerFaultTrials":sum(r["expectedProviderFault"] for r in report["rows"]),
        "modelRequests":sum(r["modelAttempts"] for r in report["rows"])}
    it.write(root/"independent_audit.json",totals)
    return report


def run(root):
    root=Path(root); manifest=it.read(root/"experiment.json")
    if (root/"campaign_state.json").exists():raise ValueError("campaign_already_started")
    it.write(root/"campaign_state.json",{"status":"running"})
    try:
        for split in ("dev","regression","holdout"):
            for version in ("baseline","candidate"):
                it.launch(root,version,split)
        report=audit(root)
        adopt=bool(report["adoptionGatePassed"] and report["comparisons"]["holdout"]["passDeltaPercentagePoints"]>0)
        it.write(root/"adoption_decision.json",{"adopt":adopt,
            "reason":"Predeclared gate and positive holdout behavior difference" if adopt else "No sufficient gain under the predeclared adoption rule; preserve opt-in implementation and baseline default.",
            "changesProductionDefault":False,"requiresFinalPublicationAndVerification":True})
        it.write(root/"campaign_state.json",{"status":"complete"})
    except Exception:
        it.write(root/"campaign_state.json",{"status":"failed"});raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","audit"))
    p.add_argument("--output",default="D:/TrafficMind-cache/output-repair")
    p.add_argument("--experiment")
    a=p.parse_args()
    if a.action=="freeze":print("FROZEN "+str(freeze(a.output)),flush=True)
    else:globals()[a.action](a.experiment)


if __name__=="__main__":main()
