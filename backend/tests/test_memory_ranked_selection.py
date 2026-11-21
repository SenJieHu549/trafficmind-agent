"""Rank all eligible candidates before shared/role budgets; verify real delivery."""
import asyncio
from copy import deepcopy
import json

import pytest

from backend.memory.coordinator import MemoryCoordinator
from backend.memory.injector import MemoryInjector
from backend.memory.models import MemoryItem
from backend.memory.recall_planner import MemoryRecallPlan
from backend.memory.retriever import FilteredItem, MemoryRetriever
from backend.memory.runtime_context import project_runtime_memory
from backend.tests.test_analysis_handoff import env
from backend.tests.test_llm_accident_agent import worker
from backend.tests.test_llm_accident_integration import setup


def item(memory_id, **changes):
    return MemoryItem(
        **{"id": memory_id, "memory_type": "constraint", "memory_key": "constraint." + memory_id,
           "session_id": "session", "event_thread_id": "thread", "status": "active",
           "value": {"text": "Keep access available"}, "authority_level": 0,
           "created_at": "2020-01-01T00:00:00Z", **changes})


class Store:
    """Return a deliberately ordered candidate pool, like a bounded store query."""
    def __init__(self, items):
        self.items = items
        self.limits = []

    def list_session_items(self, **query):
        self.limits.append(query["limit"])
        return [i for i in self.items
                if (not query.get("memory_type") or i.memory_type == query["memory_type"])
                and (not query.get("memory_key") or i.memory_key == query["memory_key"])][:query["limit"]]


def plan(**changes):
    return MemoryRecallPlan(**{
        "session_id": "session", "event_thread_id": "thread", "intent": "continue_event",
        "requested_types": ["constraint"], "max_items": 2, **changes})


def test_high_value_candidate_beyond_old_prefilter_limit_wins():
    store = Store([item(f"low-{i}") for i in range(25)] + [item("important", authority_level=100)])
    result = MemoryRetriever(store).retrieve(plan(), {}, "session")
    assert result["selected"][0].item.id == "important"
    assert result["total_candidates"] == 26
    assert store.limits == [100]
    assert result["selected"][0].rank == 1
    assert result["selected"][0].score_breakdown["authorityScore"] == 1
    assert len(result["selected"]) == 2
    assert all(fi.score > 0 for fi in result["rejected"])


def test_requested_type_order_and_duplicate_query_hits_do_not_change_selection():
    memories = [item("goal", memory_type="session_goal", memory_key="goal.primary"),
                item("important", authority_level=100)]
    results = []
    for types in (["session_goal", "constraint"], ["constraint", "session_goal"]):
        result = MemoryRetriever(Store(memories)).retrieve(
            plan(requested_types=types, requested_keys=["constraint.important"], max_items=1), {}, "session")
        assert result["total_candidates"] == 2
        results.append([fi.item.id for fi in result["selected"]])
    assert results == [["important"], ["important"]]


def test_duplicate_content_keeps_highest_ranked_source_then_fills_next_slot():
    memories = [item("old", memory_key="constraint.access"),
                item("corrected", memory_key="constraint.access", authority_level=100),
                item("different", value={"text": "Do not block the hospital"})]
    result = MemoryRetriever(Store(memories)).retrieve(plan(), {}, "session")
    assert [fi.item.id for fi in result["selected"]] == ["corrected", "different"]
    assert next(fi for fi in result["rejected"] if fi.item.id == "old").rejection_reason == "duplicate_memory"


def test_token_estimate_budget_skips_large_high_score_item_and_continues():
    result = MemoryRetriever(Store([
        item("large", authority_level=100, value={"text": "x" * 3000}), item("small"),
    ])).retrieve(plan(max_token_estimate=500), {}, "session")
    assert [fi.item.id for fi in result["selected"]] == ["small"]
    assert result["rejected"][0].rejection_reason == "token_estimate_budget_exceeded"
    assert sum(fi.token_estimate for fi in result["selected"]) <= 500


@pytest.mark.parametrize("changes,reason", [
    ({"session_id": "foreign"}, "wrong_session"),
    ({"event_thread_id": "other"}, "wrong_event_thread"),
    ({"status": "candidate"}, "inactive_status"),
    ({"valid_until": "2000-01-01T00:00:00Z"}, "invalid_ttl"),
])
def test_ineligible_high_score_item_cannot_displace_valid_memory(changes, reason):
    result = MemoryRetriever(Store([item("blocked", authority_level=100, **changes), item("valid")])).retrieve(
        plan(max_items=1), {}, "session")
    assert [fi.item.id for fi in result["selected"]] == ["valid"]
    assert result["rejected"][0].rejection_reason == reason
    assert result["rejected"][0].rank == 0


def test_each_role_selects_from_ranked_pool_instead_of_shared_top_k():
    recall_plan = plan(requested_types=["stable_fact", "constraint"], max_items=1)
    memories = [item("school", memory_type="stable_fact", memory_key="school.nearby",
                     value={"value": True}, authority_level=100), item("access")]
    result = MemoryRetriever(Store(memories)).retrieve(recall_plan, {}, "session")
    assert [fi.item.id for fi in result["selected"]] == ["school"]
    injected = MemoryInjector().build_injection_context(
        result["selected"], ["AccidentAgent", "SignalAgent"], {}, "run", "session",
        ranked_items=result["ranked"], plan=recall_plan)
    assert [i["memoryId"] for i in injected["agentInjectionMap"]["AccidentAgent"]["items"]] == ["access"]
    assert [i["memoryId"] for i in injected["agentInjectionMap"]["SignalAgent"]["items"]] == ["school"]
    rejected = injected["agentSelection"]["AccidentAgent"][0]
    assert rejected == {"memoryId": "school", "selected": False, "reason": "role_key_not_allowed"}


def test_role_budget_and_duplicate_reasons_allow_smaller_lower_ranked_item():
    raw = [MemoryInjector()._item_dict(FilteredItem(item=i)) for i in [
        item("large", value={"text": "x" * 1900}),
        item("small"), item("duplicate", memory_key="constraint.small"),
        item("extra", value={"text": "Other constraint"}),
    ]]
    decisions = []
    context = project_runtime_memory({"items": raw}, "AccidentAgent", session_id="session",
                                     decisions=decisions, max_items=1, max_chars=1200)
    assert [i["memoryId"] for i in context["items"]] == ["small"]
    assert [i["reason"] for i in decisions] == ["item_chars_exceeded",
        "selected_by_rank_within_role_budget", "duplicate_memory", "role_max_items_exceeded"]
    assert len(json.dumps(context, ensure_ascii=False)) <= 1200


def test_zero_selection_budget_does_not_query_the_store():
    store = Store([item("unused")])
    result = MemoryRetriever(store).retrieve(plan(max_items=0), {}, "session")
    assert not result["selected"] and not store.limits


def test_role_total_json_budget_skips_larger_item_without_losing_smaller_one():
    raw = [MemoryInjector()._item_dict(FilteredItem(item=i)) for i in [
        item("large", value={"text": "x" * 1000}), item("small"),
    ]]
    decisions = []
    context = project_runtime_memory({"items": raw}, "AccidentAgent", session_id="session",
                                     decisions=decisions, max_chars=1000)
    assert [i["memoryId"] for i in context["items"]] == ["small"]
    assert decisions[0]["reason"] == "role_chars_budget_exceeded"
    assert len(json.dumps(context, ensure_ascii=False)) <= 1000


def test_trace_distinguishes_shared_exclusion_from_actual_role_selection(tmp_path, monkeypatch):
    from backend import config
    from backend.memory.store import MemoryRepository, init_memory_tables
    from backend.memory.recall_planner import RecallPlanner
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "memory.db"))
    init_memory_tables()
    repo = MemoryRepository()
    thread = repo.create_event_thread("session", "Accident", "previous")
    school = repo.create_item(memory_type="stable_fact", session_id="session", memory_key="school.nearby",
        value={"value": True}, status="active", source_type="user_correction", authority_level=100)
    access = repo.create_item(memory_type="constraint", session_id="session", memory_key="constraint.access",
        value={"text": "Keep access available"}, status="active", source_type="user_explicit")
    for memory in (school, access):
        repo.update_item(memory.id, event_thread_id=thread.id)
    monkeypatch.setattr(RecallPlanner, "build_plan", lambda *args, **kwargs:
        plan(event_thread_id=thread.id, requested_types=["stable_fact", "constraint"], max_items=1))
    result = MemoryCoordinator(repo=repo).recall_and_inject(
        "session", "role-selection", "继续分析", {}, ["AccidentAgent"])
    assert result["error"] is None
    assert result["selected"][0].item.id == school.id
    trace = repo.get_trace_by_run("role-selection")
    rows = {r["memoryId"]: r for r in json.loads(trace.candidates_json)}
    assert rows[access.id]["sharedContextSelected"] is False
    assert rows[access.id]["sharedContextReason"] == "max_items_exceeded"
    assert rows[access.id]["agentDecisions"]["AccidentAgent"]["selected"] is True
    assert rows[school.id]["agentDecisions"]["AccidentAgent"]["reason"] == "role_key_not_allowed"
    actual = json.loads(trace.injection_map_json)["AccidentAgent"]
    assert [i["memoryId"] for i in actual["items"]] == [access.id]


def test_real_coordinator_ranked_memory_reaches_runtime_and_persisted_audit(setup, worker, monkeypatch):
    from backend.agent.collaboration.orchestrator import CollaborationOrchestrator
    from backend.memory.store import MemoryRepository, init_memory_tables
    from backend.memory.recall_planner import RecallPlanner

    init_memory_tables()
    repo = MemoryRepository()
    thread = repo.create_event_thread("session", "Current accident", "previous")
    ids = []
    for i in range(25):
        memory = repo.create_item(memory_type="constraint", session_id="session",
            memory_key=f"constraint.low-{i}", value={"text": f"Reference {i}"},
            status="active", source_type="user_explicit", authority_level=0)
        ids.append(memory.id)
    important = repo.create_item(memory_type="constraint", session_id="session",
        memory_key="constraint.important", value={"text": "Keep emergency access available"},
        status="active", source_type="user_correction", authority_level=100)
    ids.append(important.id)
    for memory_id in ids:
        repo.update_item(memory_id, event_thread_id=thread.id)
    # Force the important item to appear after the old final-size*2 query bound.
    original_query = repo.list_session_items
    def late_candidate(**kwargs):
        found = original_query(**{**kwargs, "limit": 100})
        found.sort(key=lambda m: m.id == important.id)
        return found[:kwargs["limit"]]
    monkeypatch.setattr(repo, "list_session_items", late_candidate)
    original_plan = RecallPlanner.build_plan
    def small_plan(self, *args, **kwargs):
        result = original_plan(self, *args, **kwargs)
        result.max_items = 1
        result.requested_types = ["constraint"]
        return result
    monkeypatch.setattr(RecallPlanner, "build_plan", small_plan)
    event = deepcopy(setup["event"])
    before = deepcopy(event)
    recalled = MemoryCoordinator(repo=repo).recall_and_inject(
        "session", "ranked-delivery", "继续分析", event, ["AccidentAgent"])
    assert recalled["error"] is None
    assert recalled["selected"][0].item.id == important.id
    trace = repo.get_trace_by_run("ranked-delivery")
    candidates = json.loads(trace.candidates_json)
    selected = next(i for i in candidates if i["memoryId"] == important.id)
    assert selected["rank"] == 1 and selected["scoreBreakdown"]["authorityScore"] == 1
    assert selected["agentDecisions"]["AccidentAgent"]["selected"] is True
    assert all(i["reason"] == "max_items_exceeded" for i in json.loads(trace.rejected_json))
    assert json.loads(trace.injection_map_json) == recalled["agentInjectionMap"]
    assert recalled["agentSelectedCounts"] == {"AccidentAgent": 1}

    grounding = json.loads(setup["collaboration"].get_run("source")["grounding_context"])
    async def execute():
        return [frame async for frame in CollaborationOrchestrator().execute(
            "ranked-delivery", "session", event, ["AccidentAgent"],
            grounding_context=grounding, agent_memory_contexts=recalled["agentInjectionMap"])]
    asyncio.run(execute())
    payload = json.loads(setup["client"].requests[0]["messages"][1]["content"])
    assert [i["memoryId"] for i in payload["memoryContext"]["items"]] == [important.id]
    assert "agentSelection" not in payload["memoryContext"]
    task = next(t for t in setup["collaboration"].list_tasks("ranked-delivery")
                if t["agent_name"] == "AccidentAgent")
    assert json.loads(task["input_snapshot"])["memoryContext"]["items"][0]["memoryId"] == important.id
    assert json.loads(task["output_snapshot"])["runtimeTrace"]["memoryInput"]["memoryIds"] == [important.id]
    assert event == before
