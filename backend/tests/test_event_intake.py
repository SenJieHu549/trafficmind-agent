"""Confirmed intake on isolated SQLite, including actual Runtime input delivery."""
import asyncio
from copy import deepcopy
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config
from backend import event_intake
from backend.tools import db_tools
from backend.tools.event_identity import compact_event_context, hydrate_authoritative_event
from backend.regional.repository import SQLiteRegionalRepository


@pytest.fixture
def intake(tmp_path, monkeypatch):
    db = str(tmp_path / "intake.db")
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(db_tools, "DB_PATH", db)
    db_tools.init_db()
    repo = SQLiteRegionalRepository(db)
    repo.import_context_pack({"packageVersion": 1, "sourceType": "synthetic_fixture",
        "sourceReference": "isolated-intake-test", "verificationStatus": "synthetic",
        "region": {"regionId": "region", "name": "测试区", "city": "测试市", "timezone": "Asia/Shanghai"},
        "roads": [{"roadId": "road", "name": "测试学校路", "regionId": "region"}]})
    app = FastAPI()
    app.include_router(event_intake.router)
    return TestClient(app), db, repo


def payload():
    return {"description": "学校附近两车相撞，伤亡人数未知", "requestId": "request-intake-001",
            "confirmed": True, "regionId": "region", "roadId": "road", "dataReality": "synthetic",
            "facts": {"eventType": "accident", "vehicleCount": 2, "nearbySchool": True}}


def count(db):
    with sqlite3.connect(db) as connection:
        return connection.execute("SELECT COUNT(*) FROM event_records").fetchone()[0]


def test_preview_recognizes_user_collision_without_writing_or_inventing_metrics(intake):
    client, db, _ = intake
    result = client.post("/events/intake/preview", json={"description": payload()["description"]})
    assert result.status_code == 200
    facts = result.json()["facts"]
    assert facts["eventType"] == "accident" and facts["vehicleCount"] == 2 and facts["nearbySchool"]
    assert all(facts[key] is None for key in ("injuredCount", "fatalityCount", "avgSpeed", "queueLength", "duration"))
    assert facts["weather"] == facts["timePeriod"] == "unknown"
    assert count(db) == 0


@pytest.mark.parametrize("text", ["学校附近没有碰撞，两车道缓行", "学校附近未发生事故，两车道拥堵"])
def test_negated_accident_and_lane_count_do_not_become_a_collision(intake, text):
    result = intake[0].post("/events/intake/preview", json={"description": text}).json()
    assert result["facts"]["eventType"] != "accident"
    assert result["facts"]["vehicleCount"] is None


def test_unknown_speed_is_not_extracted_from_injury_count(intake):
    facts = intake[0].post("/events/intake/preview", json={"description": "两车相撞，车速未知，2人受伤"}).json()["facts"]
    assert facts["avgSpeed"] is None and facts["injuredCount"] == 2 and facts["fatalityCount"] is None


@pytest.mark.parametrize("description,injured,fatality", [
    ("学校附近两车相撞 伤亡未知", None, None),
    ("两车相撞，暂无人员伤亡信息", None, None),
    ("两车相撞，不能确认无人员伤亡", None, None),
    ("两车相撞，尚未确认无人伤亡", None, None),
    ("两车相撞，无人员伤亡", 0, 0),
    ("两车相撞，受伤人数2人，死亡人数未知", 2, None),
    ("两车相撞，2人受伤，1人死亡", 2, 1),
    ("两车相撞，两人受伤，无人死亡", 2, None),
    ("两车相撞，受伤人数为0，死亡人数为0", 0, 0),
    ("两车相撞，2人受伤情况待核实", None, None),
    ("两车相撞，2人受伤，3人受伤", None, None),
])
def test_casualty_semantics_survive_preview_confirmation_and_storage(intake, description, injured, fatality):
    client, db, _ = intake
    preview = client.post("/events/intake/preview", json={"description": description}).json()
    assert preview["facts"]["injuredCount"] == injured
    assert preview["facts"]["fatalityCount"] == fatality
    body = {**payload(), "description": description, "facts": preview["facts"]}
    result = client.post("/events/intake/confirm", json=body)
    assert result.status_code == 200, result.text
    row = db_tools.get_event_by_id(result.json()["eventId"])
    facts = row["rawEvent"]["reportedFacts"]
    assert facts["injuredCount"] == injured and facts["fatalityCount"] == fatality
    assert row["sourceMetadata"]["intakeAudit"]["corrections"] == {}
    assert count(db) == 1 and row["status"] == "待研判"


@pytest.mark.parametrize("duration_text,seconds,report_text", [
    ("持续5秒", 5, "持续时间：5 秒"),
    ("持续5分钟", 300, "持续时间：5 分钟"),
    ("持续1.5小时", 5400, "持续时间：90 分钟"),
    ("持续时间未知", None, "持续时间：未知"),
])
def test_main_road_and_time_units_survive_the_whole_registration_path(intake, duration_text, seconds, report_text):
    client, _, _ = intake
    description = f"主干道两车相撞，{duration_text}"
    preview = client.post("/events/intake/preview", json={"description": description}).json()
    assert preview["facts"]["isMainRoad"] is True and preview["facts"]["duration"] == seconds
    result = client.post("/events/intake/confirm", json={**payload(), "description": description, "facts": preview["facts"]})
    assert result.status_code == 200, result.text
    row = db_tools.get_event_by_id(result.json()["eventId"])
    assert row["duration"] == seconds and row["rawEvent"]["duration"] == seconds
    assert row["rawEvent"]["isMainRoad"] is True
    assert type(row["rawEvent"]["vehicleCount"]) is int
    assert report_text in row["report"]


def test_manual_correction_is_explicitly_audited_without_overwriting_original_report(intake):
    client, _, _ = intake
    description = "学校附近两车相撞，伤亡未知"
    parsed = client.post("/events/intake/preview", json={"description": description}).json()["facts"]
    changed = {**parsed, "injuredCount": 0, "duration": 5}
    saved = client.post("/events/intake/confirm", json={**payload(), "description": description, "facts": changed}).json()
    row = db_tools.get_event_by_id(saved["eventId"])
    audit = row["sourceMetadata"]["intakeAudit"]
    assert audit["parsedFacts"]["injuredCount"] is None
    assert audit["confirmedFacts"]["injuredCount"] == 0
    assert audit["corrections"] == {"injuredCount": {"parsed": None, "confirmed": 0}, "duration": {"parsed": None, "confirmed": 5.0}}
    assert row["rawEvent"]["reportedFacts"]["description"] == description


def test_server_extraction_audit_changes_do_not_break_confirmation_retries(intake, monkeypatch):
    client, db, _ = intake
    body = payload()
    first = client.post("/events/intake/confirm", json=body).json()
    stored_audit = db_tools.get_event_by_id(first["eventId"])["sourceMetadata"]["intakeAudit"]
    original = event_intake.preview_description
    def revised_preview(description):
        result = original(description)
        result["facts"]["injuredCount"] = 7
        return result
    monkeypatch.setattr(event_intake, "preview_description", revised_preview)
    retry = client.post("/events/intake/confirm", json=body)
    assert retry.status_code == 200 and retry.json()["duplicate"]
    assert retry.json()["eventId"] == first["eventId"] and count(db) == 1
    assert db_tools.get_event_by_id(first["eventId"])["sourceMetadata"]["intakeAudit"] == stored_audit


@pytest.mark.parametrize("change", [{"confirmed": False}, {"confirmed": 1}, {"roadId": "other-region-road"},
                                    {"regionId": "missing"}, {"description": " "}])
def test_invalid_or_unconfirmed_intake_does_not_write(intake, change):
    body = {**payload(), **change}
    result = intake[0].post("/events/intake/confirm", json=body)
    assert result.status_code == 422 and count(intake[1]) == 0


@pytest.mark.parametrize("value", [-1, True, "unknown"])
def test_invalid_measurements_are_rejected_before_storage(intake, value):
    body = payload()
    body["facts"]["avgSpeed"] = value
    assert intake[0].post("/events/intake/confirm", json=body).status_code == 422
    assert count(intake[1]) == 0


def test_confirmed_unknowns_survive_db_grounding_and_actual_runtime_input(intake, tmp_path):
    client, db, repo = intake
    response = client.post("/events/intake/confirm", json=payload())
    assert response.status_code == 200, response.text
    result = response.json()
    event_id = result["eventId"]
    assert not result["analysisStarted"] and result["locationBinding"]["roadId"] == "road"
    assert event_id in result["next"]["analysisUrl"]
    row = db_tools.get_event_by_id(event_id)
    assert row["status"] == "待研判"
    assert all(row[field] is None for field in ("avgSpeed", "queueLength", "duration"))
    snapshot = hydrate_authoritative_event(event_id)
    event = compact_event_context(snapshot)
    assert event["vehicleCount"] == 2
    assert event["reportedFacts"]["injuredCount"] is None
    assert event["reportedFacts"]["description"] == payload()["description"]
    from backend.agent.multi_agent import _get_event_info
    assert _get_event_info(event)["reportedFacts"] == event["reportedFacts"]
    from backend.grounding.assembler import minimal_grounded_context_from_event
    grounding = minimal_grounded_context_from_event(snapshot, reason="isolated_test").to_dict()
    assert grounding["currentEvent"]["reportedFacts"] == event["reportedFacts"]
    from backend.tests.test_llm_accident_agent import Client
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    model = Client()
    report = asyncio.run(AccidentAgentRuntime(client=model).run(
        event_id, question="核实伤亡，不执行动作", mode="llm_context", grounding=grounding,
        output_dir=tmp_path / "runtime"))
    assert report["trace"]["generationMode"] == "llm_context", report["trace"]
    actual = json.loads(model.requests[0]["messages"][1]["content"])
    assert actual["event"]["reportedFacts"] == event["reportedFacts"]
    assert actual["event"]["avgSpeed"] is None
    assert actual["availableEvidence"][0]["data"]["reportedFacts"] == event["reportedFacts"]
    assert repo.get_active_event_location_binding(event_id)["regionId"] == "region"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM event_records").fetchone()[0] == 1
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("collaboration_runs", "workflow_runs"):
            if table in names:
                assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_same_confirmation_reuses_event_and_changed_confirmation_cannot_overwrite(intake):
    client, db, _ = intake
    first = client.post("/events/intake/confirm", json=payload()).json()
    second = client.post("/events/intake/confirm", json=payload()).json()
    assert first["eventId"] == second["eventId"] and second["duplicate"]
    changed = deepcopy(payload())
    changed["facts"]["injuredCount"] = 3
    assert client.post("/events/intake/confirm", json=changed).status_code == 409
    assert count(db) == 1
    assert compact_event_context(hydrate_authoritative_event(first["eventId"]))["reportedFacts"]["injuredCount"] is None


def test_explicit_zero_is_kept_separate_from_unknown_and_report_has_no_fabricated_zero(intake):
    body = payload()
    body["facts"].update(injuredCount=0, fatalityCount=0, avgSpeed=0, queueLength=0, duration=0)
    result = intake[0].post("/events/intake/confirm", json=body)
    assert result.status_code == 200, result.text
    event = compact_event_context(hydrate_authoritative_event(result.json()["eventId"]))
    assert event["reportedFacts"]["injuredCount"] == 0 and event["avgSpeed"] == 0
    unknown = payload()
    unknown["requestId"] = "request-unknown-002"
    saved = intake[0].post("/events/intake/confirm", json=unknown).json()
    report = db_tools.get_event_by_id(saved["eventId"])["report"]
    assert "持续时间：未知" in report and "平均车速：未知" in report


def test_reported_facts_participate_in_runtime_event_change_guard(intake):
    saved = intake[0].post("/events/intake/confirm", json=payload()).json()
    from backend.agent.llm_accident.tools import EventQueries
    from backend.agent.llm_accident.contracts import AgentRuntimeError
    queries = EventQueries(saved["eventId"])
    with sqlite3.connect(intake[1]) as connection:
        raw = json.loads(connection.execute("SELECT rawEvent FROM event_records WHERE eventId=?", (saved["eventId"],)).fetchone()[0])
        raw["reportedFacts"]["injuredCount"] = 1
        connection.execute("UPDATE event_records SET rawEvent=? WHERE eventId=?", (json.dumps(raw), saved["eventId"]))
    with pytest.raises(AgentRuntimeError, match="event_changed_during_analysis"):
        queries.assert_current()


def test_retry_after_binding_failure_reuses_the_already_created_event(intake, monkeypatch):
    original = event_intake.EventLocationBindingService.resolve_and_bind
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic binding interruption")
    monkeypatch.setattr(event_intake.EventLocationBindingService, "resolve_and_bind", failed)
    with pytest.raises(RuntimeError, match="binding interruption"):
        intake[0].post("/events/intake/confirm", json=payload())
    assert count(intake[1]) == 1
    monkeypatch.setattr(event_intake.EventLocationBindingService, "resolve_and_bind", original)
    retry = intake[0].post("/events/intake/confirm", json=payload()).json()
    assert retry["duplicate"] and retry["locationBinding"]["status"] == "resolved"
    assert count(intake[1]) == 1


def test_confirmed_event_reaches_plan_approval_execution_and_case_with_reported_facts(intake, monkeypatch):
    import backend.agent.collaboration.db_repository as collaboration
    import backend.chat.chat_db as chat
    import backend.agent.collaboration.orchestrator as orchestration
    from backend.agent.llm_accident import integration
    from backend.agent.llm_accident.runtime import AccidentAgentRuntime
    from backend.tests.test_llm_accident_agent import Client, final
    from backend.grounding.assembler import minimal_grounded_context_from_event
    from backend.planning import api
    from backend.planning.models import Plan
    from backend.workflow.repository import SQLiteWorkflowRepository
    from backend.workflow.executor import WorkflowExecutor
    from backend.tests.test_analysis_handoff import run_to_terminal
    from backend.case_memory.repository import SQLiteCaseMemoryRepository
    from pathlib import Path
    import backend.rag.v2.document_repository as documents
    from backend.rag.v2.models import RagDocument, RagChunk
    monkeypatch.setattr(documents, "RAG_V2_DB_PATH", str(Path(intake[1]).with_name("intake-rag.db")))
    documents.init_db()
    documents.upsert_document(RagDocument(document_id="intake-guidance", source_id="isolated-test",
        title="现场信息核实", content="伤亡与车道未经核实须保留未知；内部核实任务需要审批。",
        grounding_scope="GLOBAL", event_type="accident"))
    documents.upsert_chunks([RagChunk(chunk_id="intake-guidance-c0", document_id="intake-guidance",
        raw_content="伤亡与车道未经核实须保留未知；内部核实任务需要审批。")])
    for module in (collaboration, chat):
        monkeypatch.setattr(module, "DB_PATH", intake[1])
    collaboration.init_collaboration_tables()
    chat.reset_initialized(); chat.init_chat_tables()
    saved = intake[0].post("/events/intake/confirm", json=payload()).json()
    snapshot = hydrate_authoritative_event(saved["eventId"])
    event = compact_event_context(snapshot)
    event.update(originalInput="核实未知伤亡和车道；仅创建内部任务候选", fieldSources={"eventId": "authoritative_event_record"})
    grounding = minimal_grounded_context_from_event(snapshot, reason="isolated_intake").to_dict()
    grounding["regionalContext"]["location"].update(regionId="region", roadId="road", intersectionId=None)
    grounding["knowledgeContext"]["evidence"] = [{"documentId": "intake-guidance", "chunkId": "intake-guidance-c0",
        "excerpt": "伤亡与车道未经核实须保留未知；内部核实任务需要审批。", "evidenceId": "E1"}]
    model = Client([lambda messages: final(messages, actions=[{"actionType": "create_dispatch_task",
        "params": {"instruction": "核实伤亡与车道；本地隔离测试"}}])])
    monkeypatch.setattr(config, "ACCIDENT_AGENT_MODE", "llm_tools")
    monkeypatch.setattr(orchestration, "LLM_ENABLED", False)
    monkeypatch.setattr(integration, "AccidentAgentRuntime", lambda **kwargs: AccidentAgentRuntime(client=model, **kwargs))
    async def analyze():
        return [frame async for frame in orchestration.CollaborationOrchestrator().execute(
            "intake-source", "session", event, ["AccidentAgent"], grounding_context=grounding)]
    asyncio.run(analyze())
    repo = SQLiteWorkflowRepository()
    monkeypatch.setattr(api, "_repo", repo)
    created = asyncio.run(api.create_plan_from_agent(api.AgentPlanRequest(eventId=saved["eventId"],
        sessionId="session", collaborationRunId="intake-source", reuseAnalysis=True)))
    plan = Plan.from_dict(created["plan"])
    run = run_to_terminal({"api": api, "repo": repo}, plan)
    assert run.status.value == "awaiting_approval"
    assert not repo.list_action_records(run.run_id)
    async def finish():
        executor = WorkflowExecutor(repository=repo)
        await executor.approve(run.run_id, reviewer="isolated test", comment="internal task only")
        async for _ in executor.execute_created_run(run.run_id):
            pass
    asyncio.run(finish())
    assert repo.get_run(run.run_id).status.value == "completed"
    case = SQLiteCaseMemoryRepository().get_case_by_source_workflow_run_id(run.run_id)
    assert case is not None
    assert case.event_snapshot["reportedFacts"]["injuredCount"] is None
    assert case.event_snapshot["reportedFacts"]["description"] == payload()["description"]
    assert case.quality_status.value == "UNVERIFIED"
