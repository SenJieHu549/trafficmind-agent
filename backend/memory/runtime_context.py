"""Bounded, role-specific memory input; never event facts or authorization."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, Dict

from backend.memory.constants import DYNAMIC_FIELD_BLOCKLIST
from backend.memory.injector import MemoryInjector
from backend.memory.policy import DEFAULT_POLICY

MAX_MEMORY_CHARS = 6000
MAX_ITEM_CHARS = 1800
MAX_MEMORY_ITEMS = 8


def _time(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _has_dynamic_key(value: Any, depth: int = 0) -> bool:
    if depth > 6:
        return True
    if isinstance(value, dict):
        blocked = {key.lower() for key in DYNAMIC_FIELD_BLOCKLIST}
        return any(str(key).lower() in blocked or _has_dynamic_key(item, depth + 1)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_has_dynamic_key(item, depth + 1) for item in value)
    return False


def project_runtime_memory(
    context: Any, agent_name: str, *, session_id: str = "",
    event_thread_id: str = "", current_event: Dict[str, Any] | None = None,
    decisions: list | None = None, max_items: int = MAX_MEMORY_ITEMS,
    max_chars: int = MAX_MEMORY_CHARS,
) -> Dict[str, Any]:
    """Recheck role, scope, validity and size before a recalled item is delivered.

    This projection contains reference data only. Confirmed decisions are not
    execution approvals, and stable memories cannot replace current event facts.
    """
    if not isinstance(context, dict):
        return {}
    event_thread_id = event_thread_id or context.get("eventThreadId", "")
    session_id = session_id or context.get("sessionId", "")
    raw_items = context.get("items")
    if not isinstance(raw_items, list):
        return {}
    allowed_types = DEFAULT_POLICY.get_allowed_memory_types_for_agent(agent_name)
    allowed_keys = MemoryInjector().field_whitelist.get(agent_name, set())
    from backend.workflow.action_execution import sanitize_public_value

    now = datetime.now(timezone.utc)
    items, seen, seen_values = [], set(), set()
    max_items = max(0, min(max_items, MAX_MEMORY_ITEMS))
    max_chars = max(0, min(max_chars, MAX_MEMORY_CHARS))

    def record(item, reason, selected=False):
        if decisions is not None:
            decisions.append({"memoryId": item.get("memoryId") if isinstance(item, dict) else None,
                              "selected": selected, "reason": reason})
    envelope = {"items": [], "itemCount": 0, "untrustedData": True,
                "notCurrentEventFacts": True, "notAuthorization": True,
                "sessionId": session_id, "eventThreadId": event_thread_id}
    current_event = current_event or {}
    fields = {"road.name": "roadName", "road.direction": "direction",
              "road.is_main": "isMainRoad", "school.nearby": "nearbySchool",
              "hospital.nearby": "nearbyHospital", "route.event_type": "eventType"}
    for item in raw_items:
        if not isinstance(item, dict):
            record(item, "invalid_item")
            continue
        key, kind = item.get("memoryKey", ""), item.get("memoryType")
        if not isinstance(key, str) or not isinstance(kind, str) or kind not in allowed_types:
            record(item, "role_type_not_allowed")
            continue
        if not allowed_keys or not any(key.startswith(prefix) for prefix in allowed_keys):
            record(item, "role_key_not_allowed")
            continue
        if item.get("status") not in {"active", "confirmed"}:
            record(item, "inactive_status")
            continue
        if session_id and item.get("sessionId") != session_id:
            record(item, "wrong_session")
            continue
        if (event_thread_id and item.get("eventThreadId")
                and item["eventThreadId"] != event_thread_id):
            record(item, "wrong_event_thread")
            continue
        if kind in {"stable_fact", "confirmed_decision"} and not item.get("eventThreadId"):
            record(item, "legacy_unscoped_memory")
            continue
        try:
            if item.get("validFrom") and _time(item["validFrom"]) > now:
                record(item, "not_yet_valid")
                continue
            if item.get("validUntil") and _time(item["validUntil"]) <= now:
                record(item, "expired_ttl")
                continue
        except (ValueError, TypeError, OverflowError):
            record(item, "invalid_validity")
            continue
        value = item.get("value")
        if not isinstance(value, dict) or _has_dynamic_key(value):
            record(item, "invalid_or_dynamic_value")
            continue
        field = fields.get(key)
        # False and zero are explicit facts too; never treat them as "missing".
        if (kind == "stable_fact" and field and field in current_event
                and current_event[field] is not None and current_event[field] != ""
                and value.get("value") != current_event[field]):
            record(item, "current_input_override")
            continue
        memory_id = item.get("memoryId")
        if not isinstance(memory_id, str) or not memory_id or memory_id in seen:
            record(item, "invalid_or_duplicate_id")
            continue
        projected = sanitize_public_value({k: deepcopy(item[k]) for k in (
            "memoryId", "memoryType", "memoryKey", "value", "status", "sessionId",
            "sourceType", "sourceId", "sourceRunId", "sourceMessageId", "eventThreadId",
            "validFrom", "validUntil", "authorityLevel", "confidence",
        ) if k in item})
        try:
            identity = (item.get("eventThreadId", ""), kind, key,
                        json.dumps(projected.get("value"), sort_keys=True,
                                   ensure_ascii=False, allow_nan=False))
            encoded = json.dumps(projected, ensure_ascii=False, allow_nan=False)
            size = len(json.dumps(
                {**envelope, "items": items + [projected], "itemCount": len(items) + 1},
                ensure_ascii=False))
        except (TypeError, ValueError):
            record(item, "invalid_serialization")
            continue
        if identity in seen_values:
            record(item, "duplicate_memory")
            continue
        if len(encoded) > MAX_ITEM_CHARS:
            record(item, "item_chars_exceeded")
            continue
        if len(items) >= max_items:
            record(item, "role_max_items_exceeded")
            continue
        if size > max_chars:
            record(item, "role_chars_budget_exceeded")
            continue
        items.append(projected)
        seen.add(memory_id)
        seen_values.add(identity)
        record(item, "selected_by_rank_within_role_budget", True)
    if not items:
        return {}
    return {**envelope, "items": items, "itemCount": len(items)}
