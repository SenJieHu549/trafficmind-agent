"""Opt-in in-process HTTP entry coverage. No server, downloads or real actions.

Recall, role input projection, review persistence and Planning use serving code.
Only explicitly declared domain-role fixtures may be substituted in offline runs.
"""
from copy import deepcopy
import json
import sqlite3
from unittest.mock import patch

import httpx


def parse_sse(text):
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        kind, data = "", []
        for line in block.splitlines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if kind and data:
            events.append({"event": kind, "data": json.loads("\n".join(data))})
    return events


def seed_memory(request, event, record):
    from backend import config
    from backend.chat.chat_db import create_session
    from backend.memory.sqlite_repository import SQLiteMemoryRepository

    session_id = record["ids"]["sessionId"]
    create_session(session_id, "collaboration")
    repo = SQLiteMemoryRepository()
    thread_id = "eval-thread-" + request["case"]["caseId"]
    with patch("backend.memory.sqlite_repository.generate_thread_id", lambda: thread_id):
        repo.create_event_thread(session_id, title=event["roadName"], started_run_id="fixture-history")
    items = []
    for item in request["case"].get("memoryItems", []):
        options = deepcopy(item)
        scope = options.pop("fixtureScope", "current")
        owner = session_id if scope != "foreign_session" else session_id + "-foreign"
        thread = thread_id if scope != "foreign_thread" else thread_id + "-foreign"
        memory = repo.create_item(session_id=owner, event_thread_id=thread,
            source_type="user_explicit", source_run_id="fixture-history", **options)
        items.append(memory.id)
    # Stable fixture timestamps keep replayed model input independent of run time.
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE memory_items SET created_at=?,updated_at=?",
                     (request["dataset"]["asOf"], request["dataset"]["asOf"]))
    record["memoryFixture"] = {"memoryIds": items, "eventThreadId": thread_id}
    return repo


def model_memory(client):
    contexts = []
    for entry in client.entries:
        for message in (entry.get("request") or {}).get("messages", []):
            if message.get("role") != "user":
                continue
            try:
                value = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and "memoryContext" in value:
                contexts.append(deepcopy(value["memoryContext"]))
    return contexts


async def run_routed_entry(request, event, record, client, stack):
    import backend.app as serving
    from backend.agent.collaboration import executor
    from backend.agent.collaboration.db_repository import SQLiteCollaborationRepository

    case = request["case"]
    repo = seed_memory(request, event, record)
    record["profile"].update(entryPoint="routed_api", memory="real_SQLite_recall_and_projection",
                             humanReview="real_HTTP_decision_API")
    stack.enter_context(patch.object(serving, "COLLABORATION_ORCHESTRATOR_ENABLED", True))
    fixtures = case.get("roleFixtures", {})
    if fixtures:
        if request["transport"] == "live":
            raise ValueError("scripted_role_fixtures_forbidden_in_live")
        record["profile"]["scriptedDomainRoles"] = sorted(fixtures)
        original = executor._call_agent_function
        async def role(name, context):
            if name in fixtures:
                return deepcopy(fixtures[name])
            return await original(name, context)
        stack.enter_context(patch.object(executor, "_call_agent_function", role))
    if case.get("selectedAgents"):
        if request["transport"] == "live":
            raise ValueError("fixture_routing_forbidden_in_live")
        stack.enter_context(patch("backend.agent.router.route_agents", lambda *a, **k: {
            "selectedAgents": list(case["selectedAgents"]), "skippedAgents": [],
            "routingReasons": ["explicit_offline_evaluation_fixture"]}))
        record["profile"]["routing"] = "explicit_offline_fixture"
    collaboration = SQLiteCollaborationRepository()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serving.app), base_url="http://evaluation") as http:
        response = await http.post("/agent/routed_analyze/stream", json={
            "eventId": event["eventId"], "sessionId": record["ids"]["sessionId"],
            "content": case["question"], "contextPolicy": case.get("contextPolicy", "continue_event")})
        response.raise_for_status()
        events = parse_sse(response.text)
        record["entryEvents"] = events
        # The HTTP wrapper and Orchestrator can both announce the same Run.
        # Require one identity, not one transport frame; never guess "latest".
        created = {e["data"]["runId"] for e in events if e["event"] == "run_created"}
        if len(created) != 1:
            raise RuntimeError("evaluation_missing_single_run_identity")
        run_id = created.pop()
        record["ids"]["collaborationRunId"] = run_id
        record["collaborationFrames"] = response.text.split("\n\n")
        trace = repo.get_trace_by_run(run_id)
        record["memoryRecall"] = trace.to_dict() if trace else None
        tasks = collaboration.list_tasks(run_id)
        record["memoryDelivery"] = {t["agent_name"]: json.loads(t["input_snapshot"]).get("memoryContext", {})
                                    for t in tasks if t["input_snapshot"]}
        record["modelMemoryContexts"] = model_memory(client)
        source = collaboration.get_run(run_id)
        record["entryStatus"] = source["status"]
        if source["status"] == "awaiting_review":
            detail = await http.get(f"/collaboration/runs/{run_id}")
            detail.raise_for_status()
            review = detail.json()["humanReview"]
            before_tasks = deepcopy(tasks)
            review_record = {"before": review, "tasksBefore": before_tasks,
                             "modelRequestsBefore": len(client.entries), "statusBefore": source["status"]}
            record["humanReview"] = review_record
            policy = case.get("reviewDecision", {"action": "wait"})
            if policy["action"] != "wait":
                action = policy["action"]
                body = {"expectedVersion": review["version"], "action": "reject" if action == "reject" else "resolve",
                        "reviewer": "evaluation-fixture", "comment": "Fixture decision; no external action authorization",
                        "resolutions": [] if action == "reject" else [
                            {"conflictId": c["id"], **({"choice": "select", "selectedAgent": policy["selectedAgent"]}
                              if action == "select" else {"choice": "edit", "resolution": policy["resolution"]})}
                            for c in review["conflicts"]]}
                review_record["request"] = deepcopy(body)
                decided = await http.post(f"/collaboration/runs/{run_id}/reviews/{review['reviewId']}", json=body)
                review_record["response"] = {"statusCode": decided.status_code, "body": decided.json()}
                decided.raise_for_status()
                if policy.get("repeat"):
                    repeated = await http.post(f"/collaboration/runs/{run_id}/reviews/{review['reviewId']}", json=body)
                    review_record["duplicateResponse"] = {"statusCode": repeated.status_code, "body": repeated.json()}
                    repeated.raise_for_status()
            detail = await http.get(f"/collaboration/runs/{run_id}")
            detail.raise_for_status()
            review_record.update(after=detail.json()["humanReview"],
                modelRequestsAfter=len(client.entries), tasksAfter=collaboration.list_tasks(run_id))
        record["routing"] = {"selectedAgents": json.loads(collaboration.get_run(run_id)["selected_agents"])}
    return run_id
