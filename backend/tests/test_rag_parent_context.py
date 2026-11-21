"""Canonical parent expansion is bounded, scoped and present in real prompts."""
import pytest

from backend.rag.v2.chunker import TrafficKnowledgeChunker
from backend.rag.v2.context_expansion import resolve_parent_context
from backend.rag.v2.context_packer import ContextPacker
from backend.rag.v2.models import DocType, EvidenceItem, RagDocument

pytestmark = pytest.mark.usefixtures("isolated_phase11_rag_v2_env")


@pytest.fixture
def case():
    from backend.rag.v2 import document_repository as repo
    doc = RagDocument(
        document_id="parent-case", source_id="parent-test", title="A scoped traffic case",
        doc_type=DocType.CASE, version=3, region_id="region-a", grounding_scope="REGION",
        content="事实：No school crossing; keep emergency access.\n"
                "措施：Extend the vehicle green phase.\n"
                "结果：Queue reduced; pedestrian crossing must remain available.",
    )
    parents, children = TrafficKnowledgeChunker().chunk_document(doc)
    repo.upsert_document(doc)
    repo.upsert_chunks(parents + children)
    child = next(c for c in children if "Extend" in c.raw_content)
    evidence = EvidenceItem(
        evidence_id="E1", chunk_id=child.chunk_id, document_id=doc.document_id,
        parent_chunk_id=child.parent_chunk_id, version=doc.version,
        title=doc.title, content=child.raw_content, contextual_content=child.contextual_content,
        doc_type=doc.doc_type, region_id=doc.region_id, grounding_scope=doc.grounding_scope,
    )
    return doc, parents[0], children, child, evidence


def test_case_and_rule_children_have_real_parents(case):
    doc, parent, children, _, _ = case
    assert all(c.parent_chunk_id == parent.chunk_id for c in children)
    rule = doc.model_copy(update={
        "document_id": "rule", "doc_type": DocType.RULE,
        "content": "# Main\nApply only when safe.\n## Action\nExtend green.\n## Exception\nKeep crossing open.",
    })
    parents, children = TrafficKnowledgeChunker().chunk_document(rule)
    action = next(c for c in children if c.raw_content == "Extend green.")
    full = next(p for p in parents if p.chunk_id == action.parent_chunk_id)
    assert "Apply only when safe." in full.raw_content
    assert "Keep crossing open." in full.raw_content


def test_single_action_hit_expands_facts_and_outcome_and_preserves_reference(case):
    _, parent, _, child, evidence = case
    before = evidence.model_dump()
    packed = ContextPacker().pack([evidence])
    assert "[E1]" in packed and child.chunk_id in packed
    assert parent.chunk_id in packed and "v3" in packed
    assert "No school crossing" in packed and "Queue reduced" in packed
    assert evidence.model_dump() == before
    unexpanded = ContextPacker().pack([evidence], include_parents=False)
    assert "No school crossing" not in unexpanded


def test_parent_is_deduplicated_and_pack_has_hard_character_budget(case):
    doc, parent, children, child, evidence = case
    other = children[-1]
    second = evidence.model_copy(update={
        "evidence_id": "E2", "chunk_id": other.chunk_id,
        "content": other.raw_content, "contextual_content": other.contextual_content,
    })
    packed = ContextPacker().pack([evidence, second])
    assert packed.count("父块上下文: " + parent.chunk_id) == 1
    assert len(ContextPacker(max_tokens=90).pack([evidence, second])) <= 180


@pytest.mark.parametrize("change", ["document_version", "parent_version", "region", "expired", "missing"])
def test_stale_or_wrong_scope_parent_is_not_expanded(case, change):
    from backend.rag.v2 import document_repository as repo
    doc, parent, children, child, evidence = case
    if change == "document_version":
        repo.upsert_document(doc.model_copy(update={"version": 4}))
    elif change == "expired":
        repo.upsert_document(doc.model_copy(update={"status": "expired"}))
    elif change == "missing":
        evidence.parent_chunk_id = "missing-parent"
    else:
        parent = parent.model_copy(update={"version": 2} if change == "parent_version"
                                   else {"region_id": "other-region"})
        repo.upsert_chunks([parent] + children)
    packed = ContextPacker().pack([evidence])
    assert "父块上下文:" not in packed
    assert "No school crossing" not in packed


def test_legacy_unlinked_case_can_use_unique_canonical_parent(case):
    from backend.rag.v2 import document_repository as repo
    doc, parent, children, child, evidence = case
    for c in children:
        c.parent_chunk_id = None
    repo.upsert_chunks([parent] + children)
    evidence.parent_chunk_id = None
    assert "No school crossing" in ContextPacker().pack([evidence])


def test_generator_sanitizes_expanded_pack_instead_of_discarding_it(case):
    from backend.rag.v2.grounded_generator import GroundedGenerator
    from backend.rag.v2.models import EvidenceState
    _, _, _, _, evidence = case
    system, user = GroundedGenerator()._build_llm_messages(
        "What can be reused?", [evidence], EvidenceState.SUFFICIENT, "")
    assert "[E1]" in user and "No school crossing" in user and "Queue reduced" in user
    assert "参考数据" in user


def test_accident_tool_source_includes_revalidated_parent_context(case):
    from backend.agent.llm_accident.tools import EventQueries
    doc, parent, _, child, _ = case
    source = EventQueries._knowledge_source({"documentId": doc.document_id, "chunkId": child.chunk_id})
    related = source["data"]["relatedContext"]
    assert related["parentChunkId"] == parent.chunk_id
    assert related["parentVersion"] == 3
    assert "No school crossing" in related["content"] and "Queue reduced" in related["content"]


def test_large_parent_keeps_each_case_section_with_explicit_truncation(case):
    from backend.rag.v2 import document_repository as repo
    doc, parent, children, child, evidence = case
    parent.raw_content = ("事实：FACT " + "a" * 2500 + "\n措施：ACTION " + "b" * 2500
                          + "\n结果：OUTCOME " + "c" * 2500)
    repo.upsert_chunks([parent] + children)
    related = resolve_parent_context(doc.document_id, child.chunk_id, max_chars=1000)
    assert len(related["content"]) <= 1000
    assert all(label in related["content"] for label in ("FACT", "ACTION", "OUTCOME"))
    assert related["truncated"] and "已截断" in related["content"]
