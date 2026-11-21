"""Candidate contract projection and rejection remain aligned with planner schema."""
from copy import deepcopy

import pytest

from backend.agent.llm_accident.action_contract import candidate_action_schema
from backend.agent.llm_accident.contracts import AgentRuntimeError, canonical, digest
from backend.planning.param_schema import PLANNER_PARAM_SCHEMAS
from backend.tests.test_llm_accident_agent import Client, final, run, worker


def branches():
    return {b["properties"]["actionType"]["const"]: b for b in candidate_action_schema()["items"]["oneOf"]}


def test_contract_exposes_only_existing_role_actions_and_authoritative_params():
    actions = branches()
    assert set(actions) == {"create_dispatch_task", "send_notification", "update_event_status"}
    for name, branch in actions.items():
        params = branch["properties"]["params"]
        assert params["required"] == PLANNER_PARAM_SCHEMAS[name]["required"]
        assert set(params["properties"]) == set(PLANNER_PARAM_SCHEMAS[name]["fields"])
        assert params["additionalProperties"] is False
        assert "eventId" not in params["properties"]
    assert "notify_wechat" not in actions


def test_central_schema_change_updates_prompt_and_runtime_together(monkeypatch):
    changed = deepcopy(PLANNER_PARAM_SCHEMAS["create_dispatch_task"])
    changed["fields"]["reference"] = {"type": "str", "required": True, "description": "fixture-only new field"}
    changed["required"].append("reference")
    monkeypatch.setitem(PLANNER_PARAM_SCHEMAS, "create_dispatch_task", changed)
    schema = candidate_action_schema()
    action = {"actionType": "create_dispatch_task", "params": {"instruction": "核实现场", "reference": "fixture"}}
    client = Client([lambda messages: final(messages, actions=[action])])
    report, queries = run(client)
    assert canonical(schema) in client.requests[0]["messages"][0]["content"]
    assert report["trace"]["actionContractHash"] == digest(schema)
    assert report["agentResult"]["proposed_actions"] == [action]
    assert not queries.calls


def test_unregistered_action_is_not_advertised(monkeypatch):
    from backend.agent.tool_registry import get_tool_registry
    registry = get_tool_registry()
    monkeypatch.setattr(registry, "_tools", {k:v for k,v in registry._tools.items() if k != "create_dispatch_task"})
    assert "create_dispatch_task" not in branches()


def test_unknown_schema_type_fails_closed(monkeypatch):
    changed = deepcopy(PLANNER_PARAM_SCHEMAS["create_dispatch_task"])
    changed["fields"]["instruction"]["type"] = "unsupported-fixture-type"
    monkeypatch.setitem(PLANNER_PARAM_SCHEMAS, "create_dispatch_task", changed)
    with pytest.raises(AgentRuntimeError, match="action_schema_unavailable"):
        candidate_action_schema()


def test_instruction_proposal_is_accepted_without_running_a_write_tool():
    action = {"actionType": "create_dispatch_task", "params": {"instruction": "核实伤亡与通行状况"}}
    client = Client([lambda messages: final(messages, actions=[action])])
    report, queries = run(client)
    assert report["trace"]["generationMode"] == "llm_tools"
    assert report["agentResult"]["proposed_actions"] == [action]
    assert not queries.calls and not report["trace"]["toolCalls"]
    assert {t["function"]["name"] for t in client.requests[0]["tools"]}.isdisjoint(branches())


@pytest.mark.parametrize("params", [
    {}, {"remark": "核实现场"}, {"instruction": 123},
    {"instruction": "核实现场", "eventId": "another-event"},
    {"instruction": "核实现场", "priority": "high"},
    {"instruction": "核实现场", "approvalRequired": False},
])
def test_invalid_params_are_rejected_not_repaired(params):
    action = {"actionType": "create_dispatch_task", "params": params}
    report, queries = run(Client([lambda messages: final(messages, actions=[action])]))
    assert report["trace"]["fallbackReason"] == "invalid_action_proposal"
    assert report["trace"]["generationMode"] == "rule_fallback"
    assert not report["agentResult"]["proposed_actions"] and not queries.calls


@pytest.mark.parametrize("name", ["notify_wechat", "send_email"])
def test_registered_but_out_of_role_actions_are_rejected(name):
    report, _ = run(Client([lambda messages: final(messages, actions=[{"actionType":name,"params":{}}])]))
    assert report["trace"]["fallbackReason"] == "invalid_final_output"
