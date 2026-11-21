"""Deterministic recovery-path checks; not a model-quality benchmark."""
import json

import pytest

from backend.agent.llm_accident.contracts import RuntimeLimits
from backend.tests.test_llm_accident_agent import Client, Queries, call, final, run, worker


def malformed(_): return {"message":{"content":"{not valid json"}}


def repaired(client, **kwargs):
    return run(client, limits=kwargs.pop("limits",RuntimeLimits(max_output_repairs=1)), **kwargs)


def test_one_repair_uses_same_context_no_tools_and_preserves_first_failure():
    client = Client([malformed,final])
    report,q = repaired(client)
    trace = report["trace"]
    assert trace["status"]=="completed" and not q.calls
    assert trace["firstFinalSchemaValid"] is False and trace["outputRepairValidated"]
    assert trace["outputRepairAttempts"]==1 and len(client.requests)==2
    assert client.requests[1]["allow_tools"] is False
    assert client.requests[0]["messages"][1]==client.requests[1]["messages"][1]
    assert trace["outputValidationErrors"][0]["category"]=="json_syntax"
    assert trace["llmCalls"][1]["purpose"]=="output_repair"


@pytest.mark.parametrize("field,value",[("confidence","SECRET_INPUT"),("findings","wrong type"),("urgency","urgent")])
def test_field_feedback_does_not_echo_invalid_input(field,value):
    def bad(messages):
        reply=final(messages); body=json.loads(reply["message"]["content"])
        body[field]=value; reply["message"]["content"]=json.dumps(body)
        return reply
    client=Client([bad,final]); report,_=repaired(client)
    assert report["trace"]["outputRepairValidated"]
    assert value not in json.dumps(report["trace"]["outputValidationErrors"])
    assert value not in client.requests[1]["messages"][-1]["content"]


def test_disabled_policy_and_exhausted_call_budget_do_not_retry():
    for limits in (RuntimeLimits(),RuntimeLimits(max_output_repairs=1,max_llm_calls=1)):
        client=Client([malformed,final]); report,_=run(client,limits=limits)
        assert len(client.requests)==1 and not report["trace"]["outputRepairAttempts"]
        assert report["trace"]["failureReason"]=="invalid_final_output"


def test_second_invalid_response_stops_and_all_attempts_are_counted():
    client=Client([malformed,malformed,final]); report,_=repaired(client)
    assert len(client.requests)==2 and report["trace"]["outputRepairAttempts"]==1
    assert not report["trace"]["outputRepairValidated"] and report["trace"]["degraded"]
    assert len(report["trace"]["outputValidationErrors"])==2


@pytest.mark.parametrize("violation",["reference","action","permission","root_identity"])
def test_known_protected_contract_error_is_not_hidden_by_another_field_error(violation):
    def bad(messages):
        reply=final(messages); body=json.loads(reply["message"]["content"])
        body["confidence"]="bad"
        if violation=="reference":body["evidence_refs"]=["invented"]
        if violation=="action":body["proposed_actions"]=[{"actionType":"send_email","params":{}}]
        if violation=="permission":body["proposed_actions"]=[{"actionType":"create_dispatch_task","params":{"instruction":"x","eventId":"other"}}]
        if violation=="root_identity":body["eventId"]="other"
        reply["message"]["content"]=json.dumps(body); return reply
    client=Client([bad,final]); report,_=repaired(client)
    assert len(client.requests)==1 and not report["trace"]["outputRepairAttempts"]
    assert not report["trace"]["outputValidationErrors"][0]["repairEligible"]


@pytest.mark.parametrize("violation",["reference","action","tool"])
def test_repaired_response_still_passes_every_original_guard(violation):
    def response(messages):
        if violation=="reference": return final(messages,refs=["invented"])
        if violation=="action": return final(messages,actions=[{"actionType":"create_dispatch_task","params":{"instruction":"x","eventId":"other"}}])
        return {"message":{"tool_calls":[call()]}}
    client=Client([malformed,response,final]); report,q=repaired(client)
    assert not q.calls and len(client.requests)==2
    assert not report["trace"]["outputRepairValidated"] and report["trace"]["degraded"]


def test_provider_failure_is_not_retried_as_format_repair():
    client=Client([RuntimeError("secret provider error"),final]); report,_=repaired(client)
    assert len(client.requests)==1 and not report["trace"]["outputValidationErrors"]
    assert "secret provider error" not in json.dumps(report)


def test_tool_budget_is_not_reset_by_output_repair():
    client=Client([[call()],malformed,final]); report,q=repaired(client)
    assert len(q.calls)==1 and len(client.requests)==3
    assert report["trace"]["toolAttempts"]==1 and report["trace"]["toolRounds"]==1


def test_event_changed_during_repair_blocks_the_result():
    class Changed(Queries):
        def assert_current(self):
            from backend.agent.llm_accident.contracts import AgentRuntimeError
            raise AgentRuntimeError("event_changed_during_analysis")
    report,_=repaired(Client([malformed,final]),queries=Changed("event"))
    assert report["agentResult"] is None and report["trace"]["failureReason"]=="event_changed_during_analysis"


@pytest.mark.parametrize("value",[-1,2,True])
def test_repair_cap_is_zero_or_one(value):
    with pytest.raises(ValueError,match="max_output_repairs"):RuntimeLimits(max_output_repairs=value)


def test_repair_deadline_is_the_original_overall_deadline():
    import asyncio
    class SlowRepair(Client):
        async def complete(self,messages,tools,**kwargs):
            if self.requests:await asyncio.sleep(.2)
            return await super().complete(messages,tools,**kwargs)
    report,_=repaired(SlowRepair([malformed,final]),limits=RuntimeLimits(max_output_repairs=1,overall_timeout=.05))
    assert report["trace"]["failureReason"]=="overall_timeout"
    assert not report["trace"]["outputRepairValidated"]


def test_recovery_metric_requires_full_task_pass_and_retains_eligible_failures(tmp_path):
    from backend.tests.test_evaluation_iteration import execution
    from backend.evaluation.v2.iteration import trial_row, metrics
    trial,path=execution(tmp_path)
    record=json.loads(__import__('pathlib').Path(path).read_text())
    record['runtimeTrace'].update(firstFinalSchemaValid=False,outputRepairAttempts=1,
        outputRepairValidated=True,outputValidationErrors=[{'repairEligible':True,'category':'json_syntax'}])
    __import__('pathlib').Path(path).write_text(json.dumps(record),encoding='utf-8')
    good=trial_row(trial,{'version':'candidate','split':'dev'}, {})
    failed={**good,'caseId':'other','taskPassed':False}
    m=metrics([good,failed])
    assert m['repairTaskRecoveryRate']==.5 and m['repairTaskRecoveredTrials']==1
    assert m['firstFinalSchemaValidRateObserved']==0 and m['repairEligibleTrials']==2
