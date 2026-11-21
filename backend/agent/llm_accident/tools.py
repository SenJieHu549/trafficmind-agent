"""Event-bound adapters over existing services, with bounded blocking work.

Read-only means no dispatch/status/notification side effects. Existing query
services may still write their own retrieval bookkeeping.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from backend.agent.llm_accident.contracts import (
    AgentRuntimeError, EmptyArgs, KnowledgeArgs, CaseArgs,
)
from backend.agent.tool_registry import get_tool_registry, ToolMetadata, ToolRisk
from backend.agent.tool_policy import enforce_tool_request
from backend.workflow.action_execution import sanitize_public_value

TOOL_MODELS = {"accident_road_context": EmptyArgs,
               "accident_knowledge_search": KnowledgeArgs,
               "accident_case_search": CaseArgs}
DESCRIPTIONS = {
    "accident_road_context": "查询当前绑定事故的规范道路、路口与周边信息。不是实时传感器数据；未定位时明确返回 unavailable。参数为空。",
    "accident_knowledge_search": "按具体证据缺口检索当前事故适用的法规与处置知识；已有依据足够时无需重复查询。地区、事件类型和时间由后端绑定。",
    "accident_case_search": "查询当前事故发生之前、同一区域适用的结构化处置案例，包含正面、负面和未验证经验；不允许跨事件或未来信息。",
}


def tool_schemas():
    return [{"type": "function", "function": {"name": name, "description": DESCRIPTIONS[name],
             "parameters": model.model_json_schema()}} for name, model in TOOL_MODELS.items()]


def register_accident_tools():
    registry = get_tool_registry()
    for name in TOOL_MODELS:
        if not registry.get(name):
            registry.register(ToolMetadata(name=name, description=DESCRIPTIONS[name],
                                            category="accident_query", riskLevel=ToolRisk.READ_ONLY,
                                            timeoutSeconds=240 if name == "accident_knowledge_search" else 30))


def validate_tool(name, arguments, allowed_tools):
    # Role allowlist AND central policy; neither prompt nor tool metadata alone
    # is sufficient to authorize arbitrary registered tools.
    if name not in allowed_tools or name not in TOOL_MODELS:
        raise AgentRuntimeError("tool_denied")
    metadata = get_tool_registry().get(name)
    gate = enforce_tool_request(name, caller="standalone:AccidentAgent")
    if not metadata or metadata.sideEffect or metadata.riskLevel != ToolRisk.READ_ONLY or not gate["allowed"]:
        raise AgentRuntimeError("tool_denied")
    try:
        return TOOL_MODELS[name].model_validate_json(arguments).model_dump()
    except ValueError as exc:
        raise AgentRuntimeError("invalid_tool_arguments") from exc


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="accident-query")
_slot = threading.BoundedSemaphore(1)


async def in_worker(function, *args):
    if not _slot.acquire(blocking=False):
        raise AgentRuntimeError("tool_worker_busy")
    try:
        future = _executor.submit(function, *args)
    except BaseException:
        _slot.release()
        raise
    future.add_done_callback(lambda _: _slot.release())
    return await asyncio.wrap_future(future)


def bounded(value, depth=0):
    value = sanitize_public_value(value)
    if depth > 7:
        return "[depth limited]"
    if isinstance(value, str):
        return value[:1200]
    if isinstance(value, list):
        return [bounded(v, depth + 1) for v in value[:10]]
    if isinstance(value, dict):
        return {k: bounded(v, depth + 1) for k, v in list(value.items())[:40]}
    return value


class EventQueries:
    def __init__(self, event_id):
        from backend.tools.event_identity import hydrate_authoritative_event, compact_event_context
        from backend.regional.repository import SQLiteRegionalRepository
        from backend.planning.analysis_handoff import event_facts
        self.snapshot = hydrate_authoritative_event(event_id)
        self.event = compact_event_context(self.snapshot)
        if self.event["eventType"] not in ("accident", "事故"):
            raise AgentRuntimeError("not_an_accident")
        self.event_id = self.event["eventId"]
        self.as_of = self.snapshot.get("createdAt")
        if not self.as_of:
            raise AgentRuntimeError("event_time_missing")
        from backend.planning.analysis_handoff import _time
        _time(self.as_of)
        self.facts = event_facts(self.event)
        self.regional = SQLiteRegionalRepository()
        self.binding = self.regional.get_active_event_location_binding(self.event_id) or {}
        self.location = {k: self.binding.get(k) for k in ("regionId", "roadId", "intersectionId")}

    def assert_current(self):
        from backend.tools.event_identity import hydrate_authoritative_event, compact_event_context
        from backend.planning.analysis_handoff import event_facts
        current = compact_event_context(hydrate_authoritative_event(self.event_id))
        binding = self.regional.get_active_event_location_binding(self.event_id) or {}
        if event_facts(current) != self.facts or {k: binding.get(k) for k in self.location} != self.location:
            raise AgentRuntimeError("event_changed_during_analysis")

    def initial_sources(self, grounding):
        if not grounding:
            return []
        from backend.planning.analysis_handoff import event_facts, _resolve_evidence, _time
        current = grounding.get("currentEvent") or {}
        if event_facts(current) != self.facts:
            raise AgentRuntimeError("grounding_event_mismatch")
        if _time(current.get("createdAt")) != _time(self.as_of):
            raise AgentRuntimeError("grounding_time_mismatch")
        evidence = (grounding.get("knowledgeContext") or {}).get("evidence") or []
        if not evidence:
            return []
        resolved = _resolve_evidence(evidence, self.facts, _time(self.as_of))
        return [self._knowledge_source(e) for e in resolved]

    @staticmethod
    def _knowledge_source(item, *, include_parent_context=True):
        from backend.rag.v2.document_repository import get_document, get_chunk
        from backend.planning.analysis_handoff import _document_stamp
        doc, chunk = get_document(item["documentId"]), get_chunk(item["chunkId"])
        if not doc or not chunk or chunk.document_id != doc.document_id or doc.status != "active":
            raise AgentRuntimeError("evidence_source_missing")
        # Normalize both initial and on-demand evidence to the same source chunk
        # projection; query scores/shortened excerpts must not defeat dedup.
        data = {"documentId": doc.document_id, "chunkId": chunk.chunk_id, "title": doc.title,
                "excerpt": chunk.raw_content[:1200], "sourceUri": doc.source_uri,
                "effectiveFrom": doc.effective_from.isoformat() if doc.effective_from else None,
                "effectiveTo": doc.effective_to.isoformat() if doc.effective_to else None,
                "regionId": doc.region_id, "roadId": doc.road_id, "intersectionId": doc.intersection_id,
                "eventType": doc.event_type, "documentVersion": doc.version, "chunkVersion": chunk.version,
                "sourceStamp": _document_stamp(doc, chunk)}
        from backend.rag.v2.context_expansion import resolve_parent_context
        related = resolve_parent_context(
            doc.document_id, chunk.chunk_id, parent_chunk_id=chunk.parent_chunk_id,
            version=chunk.version, max_chars=1000) if include_parent_context else None
        if related:
            data["relatedContext"] = related
        return {"kind": "knowledge", "sourceId": str(item["documentId"]) + ":" + str(item["chunkId"]),
                "data": bounded(data)}

    def execute(self, name, params):
        self.assert_current()
        if name == "accident_road_context":
            if not self.binding.get("regionId"):
                return {"status": "unavailable", "reason": "LOCATION_NOT_RESOLVED", "sources": []}
            data = bounded(self.regional.build_regional_location_context(self.binding))
            return {"status": "ok", "sources": [{"kind": "regional_context",
                "sourceId": self.binding["bindingId"], "data": data}]}
        if name == "accident_knowledge_search":
            from backend.knowledge.regional_context import EventKnowledgeContextService
            result = EventKnowledgeContextService(self.regional).get_context_for_event(self.event_id, **params)
            sources = [self._knowledge_source(e) for e in result.get("evidence", [])]
            return {"status": "ok" if sources else "empty", "reason": result.get("reason"), "sources": sources}
        if name == "accident_case_search":
            from backend.case_memory.service import TrafficCaseMemoryService
            from backend.case_memory.models import CaseMemoryError
            try:
                result = TrafficCaseMemoryService(regional_repo=self.regional).get_case_context_for_event(self.event_id, **params)
            except CaseMemoryError as exc:
                if exc.code == "EVENT_CANONICAL_REGION_MISSING":
                    return {"status": "unavailable", "reason": exc.code, "sources": []}
                raise
            keys = ("caseId", "eventId", "sourceWorkflowRunId", "completedAt", "qualityStatus", "experienceType",
                    "generatedSummary", "lessons", "eventOutcome", "workflowOutcome", "actionFeedback", "feedbackAvailability")
            sources = [{"kind": "case", "sourceId": c["caseId"],
                        "data": bounded({k: c.get(k) for k in keys})} for c in result.get("cases", [])]
            return {"status": "ok" if sources else "empty", "sources": sources}
        raise AgentRuntimeError("tool_denied")
