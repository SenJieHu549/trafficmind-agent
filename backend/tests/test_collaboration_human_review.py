"""Durable pause, atomic decisions, stale-page protection and fusion-only resume."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from backend.agent.collaboration import db_repository as db
from backend.agent.collaboration import human_review as reviews
from backend.agent.collaboration import orchestrator as orchestration
from backend.agent.collaboration.protocol import AgentResult


@pytest.fixture
def waiting(tmp_path, monkeypatch):
    from backend import config
    path = str(tmp_path / "review.db")
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(orchestration, "LLM_ENABLED", False)
    db.init_collaboration_tables()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE event_records(eventId TEXT PRIMARY KEY,status TEXT,updatedAt TEXT,avgSpeed REAL)")
        conn.execute("INSERT INTO event_records VALUES('event','待研判','original',10)")
    calls = []
    async def execute(task, state, budget):
        calls.append(task.agent_name)
        if task.agent_name == "SignalAgent":
            findings, suggestion = ["信号配时需要延长机动车绿灯"], "延长机动车绿灯20秒"
        elif task.agent_name == "PublicSafetyAgent":
            findings, suggestion = ["学校学生过街安全需要保障"], "保障学生过街相位，不延长机动车绿灯"
        else:
            findings, suggestion = ["等待协调"], "原始调度参考"
        result = AgentResult(agent_name=task.agent_name, task_id=task.task_id, status="completed",
            findings=findings, suggestion=suggestion, confidence=.8, urgency="high", evidence_refs=["source-1"])
        from backend.agent.collaboration.executor import AgentExecutionResult
        return AgentExecutionResult(True, task.agent_name, task.task_id, result)
    monkeypatch.setattr(orchestration, "execute_single_agent", execute)
    event = {"eventId": "event", "eventType": "congestion", "nearbySchool": True}
    before = deepcopy(event)
    async def run():
        return [frame async for frame in orchestration.CollaborationOrchestrator().execute(
            "run", "session", event, ["SignalAgent", "PublicSafetyAgent"])]
    frames = asyncio.run(run())
    assert event == before
    return {"path": path, "repo": db.SQLiteCollaborationRepository(), "calls": calls,
            "frames": frames, "review": reviews.get_review("run")}


def decision(review, **changes):
    return reviews.ReviewDecision(**{
        "expectedVersion": review["version"], "action": "resolve", "reviewer": "operator",
        "comment": "Protect pedestrian crossing",
        "resolutions": [{"conflictId": c["id"], "choice": "select", "selectedAgent": "PublicSafetyAgent"}
                        for c in review["conflicts"]], **changes})


def event_row(waiting):
    with sqlite3.connect(waiting["path"]) as conn:
        return conn.execute("SELECT status FROM event_records WHERE eventId='event'").fetchone()[0]


def test_pause_is_durable_and_has_no_fusion_or_event_completion(waiting):
    run = waiting["repo"].get_run("run")
    assert run["status"] == "awaiting_review" and not run["completed_at"]
    assert waiting["review"]["status"] == "pending"
    assert len(waiting["review"]["conflicts"]) == 3
    assert all(c["proposals"][0]["resultVersion"] for c in waiting["review"]["conflicts"])
    assert event_row(waiting) == "待研判"
    assert 'event: run_awaiting_review' in ''.join(waiting["frames"])
    assert 'event: fusion_start' not in ''.join(waiting["frames"])
    task = next(t for t in waiting["repo"].list_tasks("run") if t["agent_name"] == "FusionAgent")
    assert task["status"] == "blocked" and task["attempt"] == 0


def test_confirm_after_reopening_storage_resumes_only_fusion(waiting):
    # No live generator or in-memory state is passed into decide().
    reopened = reviews.get_review("run")
    calls = list(waiting["calls"])
    result = reviews.decide("run", reopened["reviewId"], decision(reopened))
    assert result["review"]["status"] == "accepted" and result["review"]["version"] == 2
    final = json.loads(waiting["repo"].get_run("run")["final_decision"])
    assert "保障学生过街相位" in final["fusionSummary"]
    assert "原始调度参考" not in final["fusionSummary"]
    assert "延长机动车绿灯20秒" not in final["fusionSummary"]
    assert final["humanReview"]["externalActionsApproved"] is False
    assert final["actionPlan"] == []
    assert waiting["calls"] == calls
    assert event_row(waiting) == "待派单"
    assert all(c["status"] == "resolved" for c in waiting["repo"].list_conflicts("run"))
    task = next(t for t in waiting["repo"].list_tasks("run") if t["agent_name"] == "FusionAgent")
    assert task["status"] == "succeeded" and task["attempt"] == 1
    assert json.loads(task["input_snapshot"])["humanReview"]["action"] == "resolve"


def test_parallel_duplicate_confirms_materialize_fusion_once(waiting):
    review = waiting["review"]
    body = decision(review)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reviews.decide("run", review["reviewId"], body), range(2)))
    assert sorted(r["duplicate"] for r in results) == [False, True]
    assert sum(e["event_type"] == "run_completed" for e in waiting["repo"].list_events("run")) == 1
    with pytest.raises(reviews.ReviewError, match="裁决已处理"):
        reviews.decide("run", review["reviewId"], decision(review, comment="Different decision"))


def test_edit_changes_final_text_and_reject_needs_a_reason(waiting):
    review = waiting["review"]
    body = decision(review, resolutions=[{"conflictId": c["id"], "choice": "edit",
                                          "resolution": "设置临时护学岗，保持原信号配时"} for c in review["conflicts"]])
    result = reviews.decide("run", review["reviewId"], body)
    assert "设置临时护学岗" in result["review"]["decision"]["finalDecision"]["fusionSummary"]


def test_reject_is_terminal_without_event_advance_or_role_rerun(waiting):
    review = waiting["review"]
    request = decision(review, action="reject", comment="Need more evidence", resolutions=[])
    result = reviews.decide("run", review["reviewId"], request)
    assert result["review"]["status"] == "rejected"
    assert waiting["repo"].get_run("run")["status"] == "failed"
    assert event_row(waiting) == "待研判"
    assert reviews.decide("run", review["reviewId"], request)["duplicate"]


@pytest.mark.parametrize("change,code", [
    ({"expectedVersion": 2}, "review_version_conflict"),
    ({"resolutions": []}, "review_incomplete"),
    ({"action": "reject", "resolutions": [], "comment": ""}, "review_invalid_decision"),
    ({"resolutions": [{"conflictId": "foreign", "choice": "select", "selectedAgent": "SignalAgent"}]}, "review_incomplete"),
])
def test_invalid_decision_does_not_change_waiting_run(waiting, change, code):
    with pytest.raises(reviews.ReviewError) as exc:
        reviews.decide("run", waiting["review"]["reviewId"], decision(waiting["review"], **change))
    assert exc.value.code == code
    assert waiting["repo"].get_run("run")["status"] == "awaiting_review"
    assert event_row(waiting) == "待研判"


@pytest.mark.parametrize("sql", [
    "UPDATE event_records SET avgSpeed=20 WHERE eventId='event'",
    "UPDATE collaboration_tasks SET output_snapshot='{}' WHERE agent_name='SignalAgent'",
    "UPDATE collaboration_runs SET grounding_context='{\"changed\":true}' WHERE run_id='run'",
    "UPDATE collaboration_reviews SET checkpoint_json='{}'",
])
def test_changed_sources_or_checkpoint_reject_old_confirmation(waiting, sql):
    with sqlite3.connect(waiting["path"]) as conn:
        conn.execute(sql)
    with pytest.raises(reviews.ReviewError) as exc:
        reviews.decide("run", waiting["review"]["reviewId"], decision(waiting["review"]))
    assert exc.value.code == "review_source_changed"


def test_newer_analysis_invalidates_old_review_but_replay_does_not(waiting):
    repo = waiting["repo"]
    repo.save_run({"run_id": "new", "session_id": "session", "status": "running",
                   "normalized_event": {"eventId": "event", "runKind": "replay"}})
    repo.save_run({"run_id": "new-live", "session_id": "session", "status": "running",
                   "normalized_event": {"eventId": "event"}})
    with pytest.raises(reviews.ReviewError) as exc:
        reviews.decide("run", waiting["review"]["reviewId"], decision(waiting["review"]))
    assert exc.value.code == "review_superseded"


def test_transaction_failure_rolls_back_decision_and_can_retry_after_restart(waiting, monkeypatch):
    def crash(*args):
        raise RuntimeError("injected crash before commit")
    with monkeypatch.context() as patch:
        patch.setattr(reviews, "_audit", crash)
        with pytest.raises(RuntimeError, match="injected crash"):
            reviews.decide("run", waiting["review"]["reviewId"], decision(waiting["review"]))
    assert reviews.get_review("run")["status"] == "pending"
    assert waiting["repo"].get_run("run")["status"] == "awaiting_review"
    assert event_row(waiting) == "待研判"
    assert not any(e["event_type"] == "human_review_decided" for e in waiting["repo"].list_events("run"))
    assert reviews.decide("run", waiting["review"]["reviewId"], decision(reviews.get_review("run")))["review"]["status"] == "accepted"


def test_http_detail_and_confirm_contract(waiting):
    from backend.app import app
    client = TestClient(app)
    detail = client.get('/collaboration/runs/run').json()
    assert detail["run"]["status"] == "awaiting_review"
    review = detail["humanReview"]
    url = f"/collaboration/runs/run/reviews/{review['reviewId']}"
    assert client.post(url, json={"expectedVersion": True}).status_code == 422
    body = decision(review).model_dump()
    assert client.post(url.replace('/runs/run/', '/runs/foreign/'), json=body).status_code == 404
    assert client.post(url, json=body).status_code == 200
    assert client.post(url, json=body).json()["duplicate"] is True
    assert client.get('/collaboration/runs/run').json()["run"]["status"] == "completed"


def test_fresh_python_process_can_resume_from_database_only(waiting):
    code = (
        "import sys,json; from backend.agent.collaboration import db_repository as db; "
        "db.DB_PATH=sys.argv[1]; from backend.agent.collaboration.human_review import decide,ReviewDecision; "
        "result=decide('run',sys.argv[2],ReviewDecision.model_validate_json(sys.argv[3])); "
        "assert result['review']['status']=='accepted'"
    )
    result = subprocess.run([sys.executable, "-c", code, waiting["path"], waiting["review"]["reviewId"],
        decision(waiting["review"]).model_dump_json()], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert waiting["repo"].get_run("run")["status"] == "completed"


def test_new_replay_does_not_invalidate_live_review(waiting):
    waiting["repo"].save_run({"run_id": "replay", "session_id": "session", "status": "running",
        "normalized_event": {"eventId": "event", "runKind": "replay"}})
    assert reviews.decide("run", waiting["review"]["reviewId"], decision(waiting["review"]))["review"]["status"] == "accepted"


def test_background_transport_error_cannot_relabel_a_durable_pause_as_failed(waiting):
    from backend.agent.collaboration.background_stream import detached_stream
    from backend.agent.streaming import sse_event
    async def source():
        yield sse_event("run_awaiting_review", {"runId": "run", "status": "awaiting_review"})
        raise RuntimeError("post-pause transport error")
    async def consume():
        return ''.join([frame async for frame in detached_stream(source(), timeout=5)])
    text = asyncio.run(consume())
    assert 'event: run_failed' not in text
    assert '"status": "awaiting_review"' in text
    assert waiting["repo"].get_run("run")["status"] == "awaiting_review"
