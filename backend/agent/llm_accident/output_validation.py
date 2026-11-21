"""Bounded public validation feedback. Never include Pydantic input/context."""
import json

from backend.agent.llm_accident.contracts import ActionProposal
from backend.workflow.action_execution import contains_sensitive_key

REPAIRABLE_FIELDS = frozenset({"findings", "confidence", "suggestion", "urgency", "assumptions"})


def validation_feedback(error, content, ledger, *, compact, validate_actions, resolve_refs):
    errors = error.errors(include_input=False, include_context=False, include_url=False)
    # Extra-field names and error messages may echo arbitrary model content.
    issues = [{"field":str(e.get("loc", ("",))[0]) if e.get("loc") and e["loc"][0] in REPAIRABLE_FIELDS else "output",
               "type":e["type"]} for e in errors[:8]]
    syntax = any(e["type"] == "json_invalid" for e in errors)
    eligible = syntax or bool(errors) and all(e.get("loc") and e["loc"][0] in REPAIRABLE_FIELDS for e in errors)
    guard = None
    try:
        body = json.loads(content)
    except (ValueError, TypeError):
        body = None
    if isinstance(body, dict):
        try:
            if contains_sensitive_key(body):
                raise ValueError("protected_field")
            refs = body.get("evidence_refs")
            if not isinstance(refs, list) or not all(isinstance(r,str) for r in refs):
                raise ValueError("reference_shape")
            resolve_refs(refs, ledger) if compact else ledger.validate_refs(refs)
            actions = body.get("proposed_actions", [])
            if not isinstance(actions,list): raise ValueError("action_shape")
            validate_actions([ActionProposal.model_validate(a) for a in actions])
        except Exception:
            eligible, guard = False, "protected_contract_invalid"
    if not isinstance(content,str) or len(content)>24000:
        eligible, guard = False, "output_size_invalid"
    return {"category":"json_syntax" if syntax else "schema_fields", "issues":issues,
            "repairEligible":bool(eligible), "guard":guard}
