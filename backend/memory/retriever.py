"""
Memory Retriever — Phase 10 里程碑三

通过 MemoryStore 查询、过滤、排序候选记忆。
不直接访问 SQLite，所有查询通过 MemoryStore 接口。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.memory.models import MemoryItem
from backend.memory.repository import MemoryStore


@dataclass
class FilteredItem:
    """筛选后的记忆项，包含过滤原因和评分。"""
    item: MemoryItem
    selected: bool = False
    rejection_reason: str = ""
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    selected_reason: str = ""
    rank: int = 0
    token_estimate: int = 0


class MemoryRetriever:
    """记忆检索器。

    通过 MemoryStore 接口查询，不依赖 SQLite。
    """

    def __init__(self, repo: MemoryStore):
        self.repo = repo

    def retrieve(
        self,
        plan,  # MemoryRecallPlan
        current_event: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """根据计划检索并过滤记忆。

        Returns:
            {
                "candidates": List[FilteredItem],
                "selected": List[FilteredItem],
                "rejected": List[FilteredItem],
                "total_candidates": int,
                "selected_count": int,
                "rejected_count": int,
            }
        """
        all_candidates: List[FilteredItem] = []
        seen_ids = set()
        # Candidate search is bounded separately from the final injection size.
        query_limit = max(0, min(plan.candidate_limit_per_query, 1000)) if plan.max_items > 0 else 0

        def add(items):
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    all_candidates.append(FilteredItem(item=item))

        # Query by requested types
        for mt in dict.fromkeys(plan.requested_types) if query_limit else []:
            items = self.repo.list_session_items(
                session_id=session_id,
                memory_type=mt,
                limit=query_limit,
            )
            add(items)

        # Query specific keys if requested
        for key in dict.fromkeys(plan.requested_keys) if query_limit else []:
            items = self.repo.list_session_items(
                session_id=session_id,
                memory_key=key,
                limit=query_limit,
            )
            add(items)

        # Filter
        selected = []
        rejected = []
        for fi in all_candidates:
            reason = ("wrong_session" if fi.item.session_id != session_id
                      else self._filter_item(fi, plan, current_event))
            if reason:
                fi.selected = False
                fi.rejection_reason = reason
                rejected.append(fi)
            else:
                fi.selected = True
                selected.append(fi)

        # Rank the full eligible pool before any final-context truncation.
        from backend.memory.reranker import MemoryReranker
        ranker = MemoryReranker()
        ranked = ranker.rerank(selected, plan, current_event)
        selected, overflow = ranker.select(ranked, plan)
        rejected.extend(overflow)

        return {
            "candidates": all_candidates,
            "selected": selected,
            "ranked": ranked,
            "rejected": rejected,
            "total_candidates": len(all_candidates),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
        }

    def _filter_item(
        self,
        fi: FilteredItem,
        plan,  # MemoryRecallPlan
        current_event: Dict[str, Any],
    ) -> str:  # Returns rejection reason, or "" if accepted
        item = fi.item

        # 1. Status exclusions
        historical_proposal = (item.status == "candidate" and item.memory_type == "proposal"
                               and plan.include_proposals)
        if item.status not in {"active", "confirmed"} and not historical_proposal:
            return item.status if item.status in {"rejected", "superseded", "expired"} else "inactive_status"

        # 2. TTL check
        if item.valid_until and not item.is_valid():
            return "invalid_ttl"

        # 3. Event Thread scope
        event_thread_id = getattr(item, "event_thread_id", "")
        if plan.event_thread_id and event_thread_id and event_thread_id != plan.event_thread_id:
            if not plan.include_historical_threads:
                return "wrong_event_thread"
            # historical threads: mark but don't reject
            fi.score_breakdown["historical_reference"] = 1.0

        # Legacy (no thread) items
        if not event_thread_id and item.memory_type in ("stable_fact", "confirmed_decision"):
            return "legacy_unscoped_memory"

        # 4. Current input override — if current_event explicitly provides a field,
        #    don't recall a memory with a different value for the same field
        if item.memory_type == "stable_fact":
            field_map = {
                "road.name": "roadName",
                "school.nearby": "nearbySchool",
                "hospital.nearby": "nearbyHospital",
                "road.is_main": "isMainRoad",
            }
            field_name = field_map.get(item.memory_key)
            if field_name and field_name in current_event:
                current_val = current_event[field_name]
                mem_val = item.value.get("value")
                # Only override when current input has a MEANINGFUL value
                # (not empty, not None, not default placeholder)
                _is_empty = current_val is None or current_val == "" or current_val is False
                _is_default = isinstance(current_val, str) and current_val in ("未知路段", "未命名路段", "未命名")
                if not _is_empty and not _is_default and mem_val is not None and current_val != mem_val:
                    return "current_input_override"

        # 5. Dynamic field block (shouldn't exist in DB for stable_fact, but as safety)
        from backend.memory.constants import DYNAMIC_FIELD_BLOCKLIST
        for blocked_field in DYNAMIC_FIELD_BLOCKLIST:
            if blocked_field in item.value:
                return "dynamic_field_blocked"

        # 6. Proposal not confirmed
        if item.memory_type == "proposal" and not plan.include_proposals:
            return "proposal_not_confirmed"

        # 7. Run summary limit
        if item.memory_type == "run_summary":
            if not plan.include_run_summaries:
                return "intent_not_allowed"
            # Limit enforced via max_items

        # 8. Check against requested types
        if plan.requested_types and item.memory_type not in plan.requested_types:
            return "intent_not_allowed"

        # Accepted
        return ""
