"""Small runtime contracts; identities and evidence IDs are server-owned."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentRuntimeError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RuntimeLimits:
    max_tool_rounds: int = 3
    max_tool_calls: int = 4
    max_llm_calls: int = 4
    llm_timeout: float = 45
    tool_timeout: float = 240
    overall_timeout: float = 360
    max_consecutive_errors: int = 2
    max_output_repairs: int = 0

    def __post_init__(self):
        if type(self.max_output_repairs) is not int or not 0 <= self.max_output_repairs <= 1:
            raise ValueError("invalid limit: max_output_repairs")
        for key in ("max_tool_rounds", "max_tool_calls", "max_llm_calls", "max_consecutive_errors"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 12:
                raise ValueError(f"invalid limit: {key}")
        for key in ("llm_timeout", "tool_timeout", "overall_timeout"):
            value = getattr(self, key)
            if isinstance(value, bool) or not 0 < value <= 900:
                raise ValueError(f"invalid timeout: {key}")


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class KnowledgeArgs(EmptyArgs):
    query: str = Field(min_length=1, max_length=240)
    limit: int = Field(default=3, ge=1, le=5)


class CaseArgs(EmptyArgs):
    limit: int = Field(default=3, ge=1, le=5)


class ActionProposal(EmptyArgs):
    actionType: Literal["create_dispatch_task", "send_notification", "update_event_status"]
    params: dict[str, Any] = Field(default_factory=dict)


class FinalAnalysis(EmptyArgs):
    findings: list[str] = Field(min_length=1, max_length=10)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    suggestion: str = Field(min_length=1, max_length=2000)
    urgency: Literal["low", "medium", "high", "critical"]
    evidence_refs: list[str] = Field(max_length=24)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=4)
    assumptions: list[str] = Field(default_factory=list, max_length=10)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class EvidenceLedger:
    def __init__(self, event_id: str, as_of: str):
        self.event_id, self.as_of = event_id, as_of
        self._items: dict[str, dict] = {}

    def add(self, kind: str, source_id: str, data: dict, origin: str) -> str:
        from backend.workflow.action_execution import sanitize_public_value
        if not kind or not source_id or not isinstance(data, dict) or not data:
            raise AgentRuntimeError("invalid_evidence_source")
        body = {"kind": kind, "sourceId": source_id, "eventId": self.event_id,
                "asOf": self.as_of, "data": sanitize_public_value(deepcopy(data))}
        stamp = digest(body)
        evidence_id = "ev_" + stamp[:24]
        if evidence_id not in self._items:
            if len(self._items) >= 32:
                raise AgentRuntimeError("evidence_budget_exhausted")
            self._items[evidence_id] = {**body, "evidenceId": evidence_id, "contentHash": stamp, "origins": []}
        origins = self._items[evidence_id]["origins"]
        if origin not in origins:
            origins.append(origin)
        return evidence_id

    def items(self):
        return deepcopy(list(self._items.values()))

    def validate_refs(self, refs):
        if any(ref not in self._items for ref in refs):
            raise AgentRuntimeError("unknown_evidence_reference")
        return list(dict.fromkeys(refs))

    def freeze(self, refs):
        refs = self.validate_refs(refs)
        body = {"schemaVersion": "accident-evidence-v1", "eventId": self.event_id,
                "asOf": self.as_of, "evidence": self.items(), "citedEvidenceIds": refs}
        return {**body, "snapshotDigest": digest(body)}
