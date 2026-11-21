"""Blocking RAG must not starve leases, timeout delivery or cancellation."""
import asyncio
import copy
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from backend.workflow.models import NodeConfig, NodeType
from backend.workflow.nodes import rag_retrieve as rag
from backend.workflow.state import TrafficWorkflowState


@pytest.fixture(autouse=True)
def isolated_worker(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(rag, "_search_executor", executor)
    monkeypatch.setattr(rag, "_search_slot", threading.BoundedSemaphore(1))
    yield
    executor.shutdown(wait=True, cancel_futures=True)


def _state():
    return TrafficWorkflowState(
        workflow_run_id="test-rag",
        current_event={"eventType": "accident", "roadName": "test road"},
    )


def _config():
    return NodeConfig(node_id="rag", node_type=NodeType.RAG_RETRIEVE)


def _output(query="test"):
    return {"query": query, "results": [{"id": "e1"}], "resultCount": 1,
            "traceId": "trace-test", "degraded": False}


async def _until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        await asyncio.sleep(0.01)


def _slot_available():
    if not rag._search_slot.acquire(blocking=False):
        return False
    rag._search_slot.release()
    return True


def test_v2_retrieval_runs_off_loop_and_preserves_context(monkeypatch):
    import backend.rag.v2.pipeline as pipeline
    main_thread = threading.get_ident()
    worker_threads = []
    calls = []

    def search(**kwargs):
        worker_threads.append(threading.get_ident())
        calls.append(kwargs)
        time.sleep(0.12)
        return {"results": [{"id": "e1"}], "trace": {"trace_id": "real-trace"}}

    monkeypatch.setattr(pipeline, "get_pipeline", lambda: SimpleNamespace(search=search))

    async def scenario():
        state = _state()
        original_event = copy.deepcopy(state.current_event)
        task = asyncio.create_task(rag.execute_rag_retrieve(state, _config()))
        ticks = 0
        while not task.done():
            await asyncio.sleep(0.01)
            ticks += 1
        result = await task
        assert ticks >= 3
        assert state.current_event == original_event
        assert state.rag_context == result["rag_context"]
        assert state.rag_context["resultCount"] == 1
        assert state.rag_trace_ids == ["real-trace"]
    asyncio.run(scenario())
    assert worker_threads == [worker_threads[0]] and worker_threads[0] != main_thread
    assert calls[0]["filters"] == {"event_type": "accident"}


@pytest.mark.parametrize("termination", ["timeout", "cancel"])
def test_late_worker_cannot_mutate_state_or_admit_duplicate(monkeypatch, termination):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def search(*args):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        return _output()

    monkeypatch.setattr(rag, "_search_sync", search)

    async def scenario():
        state = _state()
        initial = copy.deepcopy(state.to_dict())
        task = asyncio.create_task(rag.execute_rag_retrieve(state, _config()))
        try:
            await _until(entered.is_set)
            if termination == "timeout":
                start = time.monotonic()
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(task, timeout=0.04)
                assert time.monotonic() - start < 1
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert state.to_dict() == initial
            with pytest.raises(rag.WorkflowRagBusyError):
                await rag.execute_rag_retrieve(_state(), _config())
            assert len(calls) == 1
        finally:
            release.set()
            await _until(_slot_available)
        # A completed background calculation must still not write late output.
        assert state.to_dict() == initial
        following = _state()
        await rag.execute_rag_retrieve(following, _config())
        assert len(calls) == 2 and following.rag_trace_ids == ["trace-test"]
    asyncio.run(scenario())


def test_v1_fallback_is_also_off_loop(monkeypatch):
    import backend.rag.v2.pipeline as pipeline
    import backend.rag.semantic_retriever as legacy
    worker_threads = []
    main_thread = threading.get_ident()

    def unavailable(**kwargs):
        raise RuntimeError("V2 unavailable")

    def fallback(**kwargs):
        worker_threads.append(threading.get_ident())
        time.sleep(0.08)
        return {"results": [{"id": "legacy"}]}

    monkeypatch.setattr(pipeline, "get_pipeline", lambda: SimpleNamespace(search=unavailable))
    monkeypatch.setattr(legacy, "semantic_search", fallback)
    state = _state()
    asyncio.run(rag.execute_rag_retrieve(state, _config()))
    assert worker_threads[0] != main_thread
    assert state.rag_context["degraded"] is True
    assert state.rag_context["results"] == [{"id": "legacy"}]


def test_planned_timeout_survives_definition_materialization(monkeypatch):
    from backend.planning import planner
    from backend.planning.context import build_planning_context
    from backend.planning.adapter import plan_to_definition
    monkeypatch.setattr(planner, "WORKFLOW_RAG_NODE_TIMEOUT_SECONDS", 210)
    plan = planner.build_plan(build_planning_context(_state().current_event))
    definition = plan_to_definition(plan)
    frozen = next(node for node in definition.to_dict()["nodes"] if node["nodeId"] == "rag_retrieve")
    assert frozen["timeoutSeconds"] == 210


@pytest.mark.parametrize("cancel_run", [False, True])
def test_real_driver_heartbeats_while_blocking_rag_outlives_initial_lease(tmp_path, monkeypatch, cancel_run):
    import backend.config as config
    from backend.workflow.repository import SQLiteWorkflowRepository
    from backend.workflow.models import WorkflowDefinition, WorkflowDefinitionVersion, DefinitionStatus, WorkflowRun, WorkflowRunStatus
    from backend.workflow.executor import WorkflowExecutor
    import backend.workflow.executor as executor_module
    from backend.workflow.run_driver import RunDriver
    from backend.planning.budget import new_lineage, set_lineage

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "workflow.db"))
    def lease_snapshot():
        with sqlite3.connect(config.DB_PATH) as connection:
            connection.row_factory = sqlite3.Row
            return dict(connection.execute(
                "SELECT driver_owner, driver_generation, driver_lease_until FROM workflow_runs WHERE run_id='rag-run'"
            ).fetchone())
    repo = SQLiteWorkflowRepository()
    definition = WorkflowDefinition(
        id="rag-def", name="rag scheduling", status=DefinitionStatus.ACTIVE,
        entry_node_id="trigger", nodes=[
            NodeConfig(node_id="trigger", node_type=NodeType.TRIGGER, next_nodes=["rag"]),
            NodeConfig(node_id="rag", node_type=NodeType.RAG_RETRIEVE, next_nodes=["close"], timeout_seconds=8),
            NodeConfig(node_id="close", node_type=NodeType.CLOSE),
        ],
    )
    repo.save_definition(definition)
    repo.save_definition_version(WorkflowDefinitionVersion(
        id="v-rag", definition_id=definition.id, version=1, definition_json=definition.to_dict(),
    ))
    state = {"status": "pending", "currentEvent": {"eventType": "accident", "roadName": "test"}}
    set_lineage(state, new_lineage("rag-run"))
    repo.save_run(WorkflowRun(run_id="rag-run", definition_id=definition.id,
                             status=WorkflowRunStatus.PENDING, state=state))
    repo.mark_driver_managed("rag-run")
    monkeypatch.setattr(executor_module, "get_executor", lambda: WorkflowExecutor(repository=repo))
    driver = RunDriver(repository=repo, lease_seconds=2, heartbeat_interval=0.1)
    async def no_assessment(run_id):
        pass
    monkeypatch.setattr(driver, "_assess_if_terminal", no_assessment)
    entered, release = threading.Event(), threading.Event()
    count = []
    def slow_search(*args):
        count.append(1)
        entered.set()
        assert release.wait(8)
        return _output()
    monkeypatch.setattr(rag, "_search_sync", slow_search)

    async def scenario():
        await driver._poll_once()
        try:
            await _until(entered.is_set)
            initial_lease = lease_snapshot()["driver_lease_until"]
            await asyncio.sleep(2.3)
            run = repo.get_run("rag-run")
            assert run.status == WorkflowRunStatus.RUNNING
            assert lease_snapshot()["driver_generation"] == 1
            assert lease_snapshot()["driver_lease_until"] > initial_lease
            assert not repo.claim_driver_run("rag-run", "other-worker", "2099-01-01T00:00:00Z")["claimed"]
            assert repo.is_driver_execution_valid("rag-run", driver.owner, 1)
            if cancel_run:
                cancelled = await WorkflowExecutor(repository=repo).cancel("rag-run", reason="test cancel")
                assert cancelled["status"] == "cancelled"
        finally:
            release.set()
            await _until(_slot_available)
        expected = WorkflowRunStatus.CANCELLED if cancel_run else WorkflowRunStatus.COMPLETED
        await _until(lambda: repo.get_run("rag-run").status == expected)
        await _until(lambda: lease_snapshot()["driver_owner"] is None)
        nodes = [n for n in repo.get_node_runs("rag-run") if n.node_id == "rag"]
        assert len(nodes) == 1
        if cancel_run:
            assert nodes[0].status.value != "succeeded"
            assert not repo.get_run("rag-run").state.get("ragContext")
        else:
            assert nodes[0].status.value == "succeeded"
        assert count == [1]
    asyncio.run(scenario())
