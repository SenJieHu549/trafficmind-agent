"""Project existing action contracts into the accident Agent's prompt.

This is a description of candidate output, never a set of callable write tools.
Param fields come from the same schema used by the deterministic planner.
"""
from copy import deepcopy
from typing import get_args

from backend.agent.llm_accident.contracts import ActionProposal, AgentRuntimeError, FinalAnalysis


def candidate_action_schema():
    from backend.agent.tool_registry import get_tool_registry
    from backend.planning.capability_snapshot import is_planner_executable_action
    from backend.planning.param_schema import get_param_schema

    types = {
        "str": {"type": "string"}, "int": {"type": "integer"},
        "float": {"type": "number"}, "bool": {"type": "boolean"},
        "list[str]": {"type": "array", "items": {"type": "string"}},
    }
    actions = []
    # Keep the role's existing small whitelist. Do not expose every registry tool.
    for name in sorted(get_args(ActionProposal.model_fields["actionType"].annotation)):
        if not is_planner_executable_action(name):
            continue
        meta, schema = get_tool_registry().get(name), get_param_schema(name)
        properties = {}
        for field, definition in schema["fields"].items():
            token = definition["type"]
            if token not in types:
                raise AgentRuntimeError("action_schema_unavailable")
            # Copy the type projection, including nested array item definitions.
            properties[field] = {**deepcopy(types[token]), "description": definition.get("description", "")}
        actions.append({
            "type": "object", "additionalProperties": False,
            "description": schema.get("description") or meta.description,
            "properties": {
                "actionType": {"const": name},
                "params": {"type": "object", "properties": properties,
                           "required": list(schema["required"]), "additionalProperties": False},
            },
            "required": ["actionType", "params"],
        })
    if not actions:
        raise AgentRuntimeError("action_schema_unavailable")
    return {"type": "array", "maxItems": FinalAnalysis.model_json_schema()["properties"]["proposed_actions"]["maxItems"],
            "items": {"oneOf": actions}}
