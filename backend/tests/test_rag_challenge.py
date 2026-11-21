"""Check scenario-label consistency independently of retrieval model output."""
from collections import Counter

from backend.evaluation.rag_challenge import specification


def test_challenge_gold_requires_all_scenario_facets_and_both_requested_periods():
    spec = specification()
    docs = {d["source"]:d for d in spec["documents"]}
    assert len(docs) == 60 and len({d["code"] for d in docs.values()}) == 60
    assert len({c["query"] for c in spec["cases"]}) == 24
    assert Counter(c["category"] for c in spec["cases"]) == {
        "exact_identifier":8,"paraphrased_conditions":8,"multi_document_conditions":8}
    for case in spec["cases"]:
        selected = [docs[s] for s in case["goldSources"]]
        if case["category"] == "exact_identifier":
            assert len(selected)==1 and selected[0]["code"] in case["query"]
        elif case["category"] == "paraphrased_conditions":
            assert len(selected)==1
            assert selected[0]["siteAlias"] in case["query"]
            assert selected[0]["conditionAlias"] in case["query"]
        else:
            assert len(selected)==2
            assert {d["period"] for d in selected} == {"早高峰","平峰"}
            assert len({(d["event"],d["site"],d["condition"]) for d in selected}) == 1
            assert selected[0]["site"] in case["query"] and selected[0]["condition"] in case["query"]
    # Each event has competing cards at every site, trigger state and period.
    assert sorted(Counter(d["event"] for d in docs.values()).values()) == [12]*5
