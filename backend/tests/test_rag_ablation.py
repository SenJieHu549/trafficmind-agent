"""Metric and controlled-pool checks for the retrieval-only experiment."""
import math

import pytest

from backend.evaluation import rag_ablation as ab


def test_document_duplicates_do_not_inflate_retrieval_metrics():
    ranked = [{"document_id":d} for d in ("irrelevant","g1","g1","other","g2")]
    value = ab.score(ranked,["g1","g2"])
    assert value["topDocumentIds"] == ["irrelevant","g1","other","g2"]
    assert value["recallAt5"] == 1 and value["precisionAt5"] == .4
    assert value["mrrAt5"] == .5
    assert value["ndcgAt5"] == pytest.approx((1/math.log2(3)+1/math.log2(5))/(1+1/math.log2(3)))


def test_empty_results_score_zero_and_failure_stays_in_denominator():
    rows=[{"status":"ok","metrics":ab.score([{"document_id":"g"}],["g"])},
          {"status":"error","metrics":ab.score([],["g"])}]
    value=ab.aggregate(rows)
    assert value["queries"]==2 and value["errors"]==1
    assert value["recallAt5"]==.5 and value["mrrAt5"]==.5


def test_missing_or_duplicate_gold_is_not_treated_as_perfect_recall():
    for gold in ([],["g","g"]):
        with pytest.raises(ValueError,match="gold"):
            ab.score([],gold)


def test_eligibility_keeps_same_status_and_time_rules_for_all_arms():
    def doc(identity,**extra):
        return {"document_id":identity,"status":"active","effective_from":None,"effective_to":None,**extra}
    corpus={"documents":[doc("ok"),doc("deleted",status="deleted"),
        doc("future",effective_from="2027-01-01"),doc("old",effective_to="2026-11-08T00:00:00+00:00"),
        doc("offset",effective_from="2026-11-08T07:00:00+08:00")]}
    assert ab.eligible_documents(corpus,"2026-11-08T00:00:00+00:00")==["offset","ok"]


def test_reranker_cannot_add_a_gold_candidate_not_in_the_b_pool(tmp_path):
    pool=[{"chunk_id":"c1","document_id":"d1"}]
    ab.write(tmp_path/"experiment.json",{})
    ab.write(tmp_path/"queries.json",{"cases":[{"caseId":"t"}]})
    ab.write(tmp_path/"retrieval_rows.json",[{"caseId":"t","status":"ok","category":"test",
        "goldDocumentIds":["d2"],"A":pool,"B":pool,"BPoolHash":ab.digest(pool)}])
    ab.write(tmp_path/"rerank_rows.json",[{"caseId":"t","status":"ok", "inputPoolHash":ab.digest(pool),
        "C":[{"chunk_id":"new","document_id":"d2"}]}])
    with pytest.raises(ValueError,match="membership"):
        ab.report(tmp_path)


def test_frozen_query_plan_has_three_groups_and_no_duplicate_questions():
    data=ab.read(ab.PROJECT/"backend/evaluation/v2/datasets/rag_ablation_queries.json")
    assert len(data["cases"])==30
    assert len({c["query"] for c in data["cases"]})==30
    assert all(c["goldSources"] and c["rationale"] for c in data["cases"])
    assert sorted(ab.Counter(c["category"] for c in data["cases"]).values())==[10,10,10]
