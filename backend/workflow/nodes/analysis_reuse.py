"""Load a frozen source analysis only after validating live execution facts."""
import asyncio
from copy import deepcopy

from backend.planning.analysis_handoff import validate_analysis_snapshot


async def execute_analysis_reuse(state, config):
    snapshot = deepcopy(config.config.get("analysis_snapshot"))
    if state.session_id and isinstance(snapshot, dict) and state.session_id != snapshot.get("sessionId"):
        raise ValueError("analysis_identity_mismatch: 执行会话与来源研判不一致")
    # SQLite/document checks must not monopolize the scheduling loop either.
    check = await asyncio.to_thread(validate_analysis_snapshot, snapshot, dict(state.current_event))
    state.analysis_snapshot = snapshot
    state.analysis_validation = check
    state.agent_outputs = deepcopy(snapshot["agentOutputs"])
    evidence = deepcopy(snapshot["evidence"])
    state.evidence_refs = evidence
    state.set_rag_context({"evidence": evidence, "results": [], "resultCount": 0,
                           "origin": "collaboration", "retrievalPerformed": False})
    state.add_audit_event("analysis_reused", config.node_id, check)
    return {"analysis_validation": check}
