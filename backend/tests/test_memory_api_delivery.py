"""The actual routed API forwards recall output to a persisted role input."""
import json

from backend.tests.test_phase21_grounded_agent_integration import (
    isolated, app_client, _prepare_grounded_event, _run_grounded_stream,
)


def test_routed_api_delivers_only_current_scoped_role_memory(isolated, app_client, monkeypatch):
    _prepare_grounded_event(isolated)

    class Recall:
        def recall_and_inject(self, **kwargs):
            item = {
                "memoryId": "api-memory", "memoryType": "constraint",
                "memoryKey": "constraint.pedestrian", "value": {"text": "Keep crossing available"},
                "status": "active", "sessionId": kwargs["session_id"],
                "eventThreadId": "api-thread", "sourceType": "user_explicit",
                "sourceRunId": "previous",
            }
            return {
                "eventThreadId": "api-thread", "routingContext": {},
                "agentInjectionMap": {
                    name: {"items": [item, {**item, "memoryId": "foreign", "sessionId": "other"}]}
                    for name in kwargs["agent_targets"]
                },
            }

    monkeypatch.setattr("backend.memory.coordinator.MemoryCoordinator", Recall)
    _, run, captured = _run_grounded_stream(app_client, isolated, monkeypatch)
    tasks = isolated["collabRepo"].list_tasks(run["run_id"])
    projected = [json.loads(task["input_snapshot"])["memoryContext"]
                 for task in tasks if "memoryContext" in json.loads(task["input_snapshot"])]
    assert projected
    assert all([item["memoryId"] for item in ctx["items"]] == ["api-memory"] for ctx in projected)
    assert any(ctx.get("memoryContext") for ctx in captured.values())
