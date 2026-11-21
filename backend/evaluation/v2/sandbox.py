"""Process-local fixture and side-effect boundaries; no serving code changes."""
from copy import deepcopy
from pathlib import Path
import sqlite3


def assert_isolated_paths(directory):
    import backend.config as config
    from backend.rag.v2 import config as rag
    root = Path(directory).resolve()
    for raw in (config.DB_PATH, rag.RAG_V2_DB_PATH, rag.RAG_V2_FTS_PATH, rag.RAG_V2_VECTOR_DB_PATH):
        if not Path(raw).resolve().is_relative_to(root):
            raise RuntimeError("evaluation_database_not_isolated")


def seed(dataset, case, directory):
    from backend.tools import db_tools as db
    from backend.agent.collaboration.db_repository import init_collaboration_tables
    from backend.chat.chat_db import init_chat_tables
    from backend.rag.v2 import document_repository as docs
    from backend.rag.v2.models import RagDocument, RagChunk
    from backend.regional.repository import SQLiteRegionalRepository, init_regional_tables
    from backend.tools.event_identity import hydrate_authoritative_event, compact_event_context
    from backend import config

    assert_isolated_paths(directory)
    db.init_db(); init_collaboration_tables(); init_chat_tables(); docs.init_db()
    init_regional_tables()
    event = {**deepcopy(dataset["eventDefaults"]), **deepcopy(case.get("event", {})),
             "eventId": "EVAL_V2_" + case["caseId"]}
    if not db.save_event_analysis({"eventId": event["eventId"], "standardEvent": event,
            "riskScore": case.get("riskScore", 45), "riskLevel": case.get("riskLevel", "中风险"), "status": "待派单", "report": ""}):
        raise RuntimeError("evaluation_seed_failed")
    stamp = dataset["asOf"]
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE event_records SET createdAt=?,updatedAt=? WHERE eventId=?", (stamp, stamp, event["eventId"]))
        if case.get("bindRoad"):
            conn.execute("INSERT INTO regions(region_id,name,city,timezone,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                         ("eval-region", "虚构评测区域", "测试城市", "Asia/Shanghai", stamp, stamp))
            conn.execute("INSERT INTO roads(road_id,region_id,name,road_type,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                         ("eval-road", "eval-region", event["roadName"], "urban", stamp, stamp))
    if case.get("bindRoad"):
        SQLiteRegionalRepository().save_resolved_event_location_binding({"status": "resolved",
            "eventId": event["eventId"], "regionId": "eval-region", "roadId": "eval-road",
            "resolutionMethod": "evaluation_fixture", "confidence": 1.0},
            source_type="evaluation_fixture", source_reference="trafficmind-eval-v2")
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute("UPDATE event_location_bindings SET binding_id=?,created_at=?,updated_at=? WHERE event_id=?",
                         ("eval-binding-" + case["caseId"], stamp, stamp, event["eventId"]))
    for item in case.get("knowledge", dataset["knowledge"]):
        docs.upsert_document(RagDocument(document_id=item["documentId"], source_id="evaluation-fixture",
            title=item["title"], content=item["content"], event_type=item.get("eventType", event["eventType"]), grounding_scope="GLOBAL",
            source_uri="fixture://trafficmind-eval-v2/" + item["documentId"]))
        docs.upsert_chunks([RagChunk(chunk_id=item["chunkId"], document_id=item["documentId"], raw_content=item["content"])])
    # Repository upserts stamp wall-clock time; pin it before evidence hashing.
    with sqlite3.connect(docs.RAG_V2_DB_PATH) as conn:
        for table in ("rag_documents", "rag_chunks"):
            conn.execute(f"UPDATE {table} SET created_at=?,updated_at=?", (stamp, stamp))
    if case.get("historicalCases"):
        from backend.case_memory.models import TrafficCaseMemory
        from backend.case_memory.repository import SQLiteCaseMemoryRepository
        for item in case["historicalCases"]:
            SQLiteCaseMemoryRepository().insert_case(TrafficCaseMemory.from_dict({
                "regionId": "eval-region", "eventType": "accident", "roadId": "eval-road",
                "finalStatus": "completed", "qualityStatus": "UNVERIFIED", **item}))
    return compact_event_context(hydrate_authoritative_event(event["eventId"]))


class FrozenKnowledge:
    """Explicit fixed retrieval-service snapshot, not a dense/rerank benchmark.

    The real assembler, event-bound tool adapter and source revalidation still
    run. Only the expensive knowledge retrieval service result is frozen.
    """
    def __init__(self, dataset, case=None):
        self.dataset, self.case = dataset, case or {}
        self.calls = []

    def get_context_for_event(self, event_id, *, query="", limit=5):
        knowledge = self.case.get("knowledge", self.dataset["knowledge"])
        items = (knowledge[:self.case.get("initialEvidenceCount", 1)] if not self.calls else
                 knowledge[-min(limit, self.case.get("toolEvidenceCount", 1)):])
        if self.calls and self.case.get("knowledgeEmpty"): items = []
        response = {"status": "READY", "regionalGroundingStatus": "UNAVAILABLE",
            "scope": {"asOf": self.dataset["asOf"]},
            "provenance": {"capturedAt": self.dataset["asOf"], "retrievalPipeline": "frozen-fixture-service"},
            "evidence": [{"documentId": item["documentId"], "chunkId": item["chunkId"],
                          "title": item["title"], "content": item["content"], "evidenceId": item["chunkId"]} for item in items]}
        if not items: response["status"] = "EMPTY"
        self.calls.append({"eventId": event_id, "query": query, "limit": limit, "response": deepcopy(response)})
        return response


def action_registry(effects):
    from backend.workflow.action_execution import ActionExecutorRegistry, BaseActionExecutor, ActionExecutorResult
    from backend.workflow.models import ActionStatus
    originals = ActionExecutorRegistry()
    legacy_notifications = frozenset({"notify_wechat", "notify_dingtalk"})

    class IsolatedAction(BaseActionExecutor):
        def __init__(self, name): self.action_type = name

        def validate(self, context):
            if self.action_type in legacy_notifications:
                from backend.planning.param_schema import normalize_parameter_hints
                from backend.planning.proposal import PlannerFailure
                try:
                    params = normalize_parameter_hints(self.action_type, context.params)
                except PlannerFailure:
                    return "invalid isolated legacy notification parameters"
                if params != context.params:
                    return "invalid isolated legacy notification parameters"
                return None
            return originals.get(self.action_type).validate(context)

        async def execute(self, context):
            effects.append({"actionType": context.action_type, "eventId": context.event_id,
                "runId": context.workflow_run_id, "nodeId": context.node_id,
                "idempotencyKey": context.idempotency_key, "params": deepcopy(context.params)})
            return ActionExecutorResult(status=ActionStatus.SUCCEEDED, message="isolated evaluation action",
                                        metadata={"evaluationOnly": True}, external_reference="eval:" + context.idempotency_key)

    class Registry:
        canonical = staticmethod(ActionExecutorRegistry.canonical)
        def __init__(self):
            self.items = {name: IsolatedAction(name) for name in
                          ("create_dispatch_task", "send_notification", "update_event_status",
                           *sorted(legacy_notifications))}
        def get(self, name):
            name = self.canonical(name)
            if name == "save_result": return None  # Real save_result writes only isolated DB.
            if name not in self.items: raise RuntimeError("evaluation_action_not_allowed")
            return self.items[name]
        def has(self, name): return self.get(name) is not None
    return Registry()
