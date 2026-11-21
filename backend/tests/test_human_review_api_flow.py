"""Actual SSE pause -> decision API -> authoritative detail -> Planning handoff."""
import json
import pytest

from backend.tests.test_phase21_grounded_agent_integration import isolated, app_client, _prepare_grounded_event


@pytest.mark.parametrize("post_pause_error", [False, True])
def test_real_stream_waits_and_handoff_uses_only_reviewed_fusion(isolated, app_client, monkeypatch, post_pause_error):
    from backend.planning.analysis_handoff import build_analysis_snapshot, AnalysisHandoffError
    import backend.app as app
    _prepare_grounded_event(isolated)
    monkeypatch.setattr("backend.agent.router.route_agents", lambda *args, **kwargs: {
        "selectedAgents": ["SignalAgent", "PublicSafetyAgent"], "skippedAgents": [], "routingReasons": []})
    calls, memory_writes = [], []
    async def role(name, context):
        calls.append(name)
        if name == "SignalAgent":
            findings, suggestion = ["信号配时优化延长绿灯"], "延长机动车绿灯20秒"
        elif name == "PublicSafetyAgent":
            findings, suggestion = ["学校行人过街安全"], "保障学生过街相位"
        else:
            findings, suggestion = ["等待协调"], "原调度建议"
        return {"findings": findings, "suggestion": suggestion, "confidence": .8, "urgency": "high"}
    async def memory(*args, **kwargs):
        memory_writes.append(args)
        return None
    monkeypatch.setattr("backend.agent.collaboration.executor._call_agent_function", role)
    monkeypatch.setattr(app, "_run_memory_extraction", memory)
    if post_pause_error:
        from backend.agent.collaboration import human_review
        original_pause = human_review.pause_for_review
        def fail_after_durable_pause(state):
            original_pause(state)
            raise RuntimeError("injected error after durable pause")
        monkeypatch.setattr(human_review, "pause_for_review", fail_after_durable_pause)
    response = app_client.post('/agent/routed_analyze/stream', json={
        "eventId": "E_AGENT_GROUNDED", "content": "分析信号与学生过街冲突", "contextPolicy": "fresh_event"})
    assert response.status_code == 200
    assert 'event: run_awaiting_review' in response.text
    assert 'event: run_failed' not in response.text and 'event: run_completed' not in response.text
    assert 'event: fallback_started' not in response.text
    assert '"status": "awaiting_review"' in response.text
    assert not memory_writes  # No completed decision/experience is written while waiting.
    run = isolated["collabRepo"].list_runs_by_event_id("E_AGENT_GROUNDED")[0]
    assert run["status"] == "awaiting_review"
    with pytest.raises(AnalysisHandoffError, match="analysis_source_not_completed"):
        build_analysis_snapshot("E_AGENT_GROUNDED", run["session_id"], run["run_id"])
    review = app_client.get(f"/collaboration/runs/{run['run_id']}").json()["humanReview"]
    body = {"expectedVersion": 1, "action": "resolve", "reviewer": "tester", "comment": "Reviewed",
            "resolutions": [{"conflictId": c["id"], "choice": "edit", "resolution": "保持行人过街相位，暂缓机动车延长方案"}
                            for c in review["conflicts"]]}
    before_calls = list(calls)
    done = app_client.post(f"/collaboration/runs/{run['run_id']}/reviews/{review['reviewId']}", json=body)
    assert done.status_code == 200, done.text
    assert calls == before_calls
    snapshot = build_analysis_snapshot("E_AGENT_GROUNDED", run["session_id"], run["run_id"])
    assert set(snapshot["agentOutputs"]) == {"SourceFusion"}
    assert "保持行人过街相位" in snapshot["agentOutputs"]["SourceFusion"]["summary"]
    assert snapshot["finalDecision"]["humanReview"]["externalActionsApproved"] is False
    final = app_client.get(f"/collaboration/runs/{run['run_id']}").json()
    assert final["run"]["status"] == "completed" and final["humanReview"]["status"] == "accepted"
