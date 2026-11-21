"""Evidence organization and exact mapping; no paid model calls."""
from copy import deepcopy
import pytest

from backend.agent.llm_accident.contracts import EvidenceLedger, AgentRuntimeError
from backend.agent.llm_accident.citations import public_evidence, citation_keys, resolve_citations, render_inline_citations


@pytest.fixture
def ledger():
    l = EvidenceLedger("event", "2026-11-01T10:00:00+08:00")
    l.add("current_event", "event", {"vehicleCount": 2}, "initial_event")
    l.add("knowledge", "doc:chunk", {"excerpt": "核实现场", "sourceStamp": "x"*64,
                                     "documentVersion": 1, "chunkVersion": 1}, "grounding")
    return l


def test_public_keys_hide_hashes_without_modifying_frozen_sources(ledger):
    before = ledger.freeze([])
    view = public_evidence(ledger.items(), ledger)
    assert [e["evidenceId"] for e in view] == ["E1", "E2"]
    assert "contentHash" not in view[0] and "sourceStamp" not in view[1]["data"]
    view[1]["data"]["excerpt"] = "changed"
    assert ledger.freeze([]) == before


def test_new_tool_evidence_appends_keys_and_dedup_does_not_renumber(ledger):
    before = citation_keys(ledger)
    ledger.add("knowledge", "doc:chunk", {"excerpt": "核实现场", "sourceStamp": "x"*64,
                  "documentVersion": 1, "chunkVersion": 1}, "tool:query")
    assert citation_keys(ledger) == before
    ledger.add("case", "past-case", {"summary": "过去案例"}, "tool:case")
    assert citation_keys(ledger)["E1"] == before["E1"]
    assert "E3" in citation_keys(ledger)


def test_exact_aliases_and_canonical_ids_resolve_to_same_durable_refs(ledger):
    keys = citation_keys(ledger)
    assert resolve_citations(["E2", keys["E2"], "E1"], ledger) == [keys["E2"], keys["E1"]]


@pytest.mark.parametrize("bad", ["E0", "E01", "E3", "e1", "E1 ", "invented-id"])
def test_unknown_or_similar_keys_are_rejected(ledger, bad):
    with pytest.raises(AgentRuntimeError, match="unknown_evidence_reference"):
        resolve_citations([bad], ledger)


def test_full_hash_and_shortened_id_remain_rejected(ledger):
    item = ledger.items()[0]
    for bad in ("ev_" + item["contentHash"], item["evidenceId"][:-5]):
        with pytest.raises(AgentRuntimeError, match="unknown_evidence_reference"):
            resolve_citations([bad], ledger)


def test_inline_brackets_render_to_durable_ids_but_road_labels_do_not_change(ledger):
    assert render_inline_citations("道路E1，依据[E2]核实", ledger) == "道路E1，依据[" + citation_keys(ledger)["E2"] + "]核实"
    with pytest.raises(AgentRuntimeError, match="unknown_evidence_reference"):
        render_inline_citations("依据[E99]", ledger)
