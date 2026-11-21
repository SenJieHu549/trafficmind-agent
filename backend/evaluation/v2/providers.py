"""Model I/O boundary: live recording, strict replay, and scripted acceptance.

Scripted responses exercise the real runtime but are never model-quality data.
Recordings contain public messages only, never credentials or hidden reasoning.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_value(value):
    # Preserve usage.total_tokens (not a credential), unlike generic DTO
    # sanitization which intentionally removes all token-shaped keys.
    from backend.workflow.action_execution import sanitize_public_text
    if isinstance(value, dict):
        return {k: safe_value(v) for k, v in value.items()
                if k.lower() not in {"api_key", "apikey", "authorization", "password", "secret"}}
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    return sanitize_public_text(value) if isinstance(value, str) else value


class EvaluationClient:
    def __init__(self, transport, case, *, cassette=None, inject_failure=False):
        self.transport, self.case = transport, case
        self.entries = []
        self.inject_failure = inject_failure
        self.replay = json.loads(Path(cassette).read_text(encoding="utf-8")) if cassette else []
        self.live = None

    async def complete(self, messages, tools, **kwargs):
        request = {"messages": deepcopy(messages), "tools": deepcopy(tools), **kwargs}
        fingerprint = content_hash(request)
        if self.transport == "replay":
            if len(self.entries) >= len(self.replay):
                raise RuntimeError("replay_exhausted")
            entry = deepcopy(self.replay[len(self.entries)])
            if entry["requestHash"] != fingerprint:
                raise RuntimeError("replay_request_mismatch")
            self.entries.append(entry)
            if entry.get("error"):
                raise RuntimeError(entry["error"])
            return deepcopy(entry["reply"])
        entry = {"requestHash": fingerprint, "request": safe_value(request)}
        self.entries.append(entry)
        try:
            if self.inject_failure:
                raise RuntimeError("injected_provider_failure")
            if self.transport == "live":
                if self.live is None:
                    from backend.agent.llm_accident.client import AccidentLLMClient
                    self.live = AccidentLLMClient()
                reply = await self.live.complete(messages, tools, **kwargs)
            else:
                reply = self._scripted(messages, tools, kwargs["allow_tools"])
            entry["reply"] = safe_value(reply)
            return deepcopy(reply)
        except Exception as exc:
            entry["error"] = type(exc).__name__  # Never serialize provider exception bodies.
            raise

    def assert_consumed(self):
        if self.transport == "replay" and len(self.entries) != len(self.replay):
            raise RuntimeError("replay_unconsumed_responses")

    def _scripted(self, messages, tools, allow_tools):
        observations = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        sequence = (self.case["expected"].get("requiredTools") or self.case["expected"].get("requiredToolAttempts")
                    or ["accident_road_context", "accident_knowledge_search"])
        if tools and allow_tools and len(observations) < len(sequence) and not any(o.get("status") == "error" for o in observations):
            # The second call is made only after consuming the first Observation.
            if observations:
                assert observations[-1]["status"] in {"ok", "empty", "unavailable"}
            name = sequence[len(observations)]
            args = ({"query": "核实现场伤亡与通行状况"} if name == "accident_knowledge_search" else
                    {"limit": 3} if name == "accident_case_search" else {})
            return {"message": {"role": "assistant", "tool_calls": [{
                "id": f"eval_call_{len(observations) + 1}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
                "usage": {}, "model": "scripted-acceptance"}
        initial = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        evidence = initial["availableEvidence"] + [e for o in observations for e in o.get("evidence", [])]
        result = {"findings": ["两车事故，伤亡情况待核实"], "confidence": .8,
                  "suggestion": "保持应急通道畅通，核实现场伤亡与通行状况；候选动作需审批。",
                  "urgency": "medium", "evidence_refs": list(dict.fromkeys(e["evidenceId"] for e in evidence)),
                  "proposed_actions": ([{"actionType": "create_dispatch_task",
                       "params": {"instruction": "核实现场伤亡与通行状况"}}]
                       if self.case["expected"]["requiresCandidate"] else []),
                  "assumptions": ["伤亡情况未知；未获得的信息不能视为已确认"]}
        return {"message": {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                "usage": {}, "model": "scripted-acceptance"}
