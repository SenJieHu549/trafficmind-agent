"""
rag_retrieve 节点 — RAG 知识检索。

调用 RAG V2（回退到 V1）检索相关知识：
  - 预案规则
  - 历史相似案例
  - 处置经验

检索结果存入 rag_context，不覆盖 current_event。
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from backend.workflow.models import NodeConfig
from backend.workflow.state import TrafficWorkflowState


# Admission is released by the *worker future*, not by the waiting coroutine.
# A timed-out/cancelled await does not stop CPU inference already in progress.
# Reject overlapping submissions instead of building an unbounded retry queue.
_search_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-rag")
_search_slot = threading.BoundedSemaphore(1)


class WorkflowRagBusyError(RuntimeError):
    """An earlier Workflow retrieval is still computing, possibly after timeout."""


def _search_sync(query: str, top_k: int, event_type: Optional[str]) -> Dict[str, Any]:
    """Run blocking retrieval/fallback without accessing Workflow state."""
    results = []
    trace_id = ""
    degraded = False
    try:
        from backend.rag.v2.pipeline import get_pipeline
        rag_result = get_pipeline().search(
            query=query,
            top_k=top_k,
            filters={"event_type": event_type} if event_type else None,
        )
        results = rag_result.get("results", rag_result.get("candidates", []))
        trace_value = rag_result.get("trace")
        trace_payload = trace_value if isinstance(trace_value, dict) else {}
        trace_id = str(
            rag_result.get("traceId")
            or rag_result.get("trace_id")
            or trace_payload.get("traceId")
            or trace_payload.get("trace_id")
            or ""
        )
    except Exception:
        degraded = True
        try:
            from backend.rag.semantic_retriever import semantic_search
            v1_result = semantic_search(query=query, limit=top_k, event_type=event_type)
            results = v1_result.get("results", [])
        except Exception:
            results = []
    return {
        "query": query, "results": results, "resultCount": len(results),
        "traceId": trace_id, "degraded": degraded,
    }


async def _search_in_worker(query: str, top_k: int, event_type: Optional[str]) -> Dict[str, Any]:
    if not _search_slot.acquire(blocking=False):
        raise WorkflowRagBusyError(
            "Workflow RAG is busy: a previous retrieval is still running; retry after it finishes"
        )
    try:
        future = _search_executor.submit(_search_sync, query, top_k, event_type)
    except BaseException:
        _search_slot.release()
        raise
    future.add_done_callback(lambda _: _search_slot.release())
    return await asyncio.wrap_future(future)


async def execute_rag_retrieve(
    state: TrafficWorkflowState, config: NodeConfig
) -> Dict[str, Any]:
    """执行 RAG 知识检索。

    Args:
        state: 工作流状态
        config: 节点配置（config.query_template 可用于自定义查询）

    Returns:
        检索结果（存入 rag_context）
    """
    event = state.current_event or {}
    event_type = event.get("eventTypeCn", event.get("eventType", ""))
    road_name = event.get("roadName", "")
    risk_level = state.risk_assessment.get("riskLevel", "")

    # 构建查询
    query_template = config.config.get("query_template", "{event_type} {road_name} 处置预案")
    query = query_template.format(
        event_type=event_type,
        road_name=road_name,
        risk_level=risk_level,
    )

    # Only immutable request inputs cross the thread boundary. Cancellation or
    # wait_for timeout exits here; a late worker cannot mutate state or audit.
    rag_ctx = await _search_in_worker(
        query, config.config.get("top_k", 5), event.get("eventType") or None,
    )
    trace_id = rag_ctx["traceId"]
    # 存入 rag_context（不覆盖 current_event）
    state.set_rag_context(rag_ctx)

    if trace_id:
        state.rag_trace_ids.append(trace_id)

    state.add_audit_event("rag_retrieved", config.node_id, {
        "query": query,
        "resultCount": rag_ctx["resultCount"],
        "traceId": trace_id or None,
        "degraded": rag_ctx["degraded"],
    })

    return {"rag_context": rag_ctx}
