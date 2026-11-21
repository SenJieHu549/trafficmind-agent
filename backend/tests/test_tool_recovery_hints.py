import sqlite3

import pytest

from backend.agent.llm_accident.tool_recovery import recovery_hint


@pytest.mark.parametrize("sqlite_code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY | (1 << 8)])
def test_only_typed_database_busy_or_locked_error_is_retryable(sqlite_code):
    error = sqlite3.OperationalError("not returned to model")
    error.sqlite_errorcode = sqlite_code
    hint = recovery_hint("accident_case_search", "tool_execution_failed", error, 1)
    assert hint["retryable"] and hint["remainingRetries"] == 1
    assert hint["category"] == "transient_query_failure"
    assert "not returned to model" not in str(hint)
    assert not recovery_hint("accident_case_search", "tool_execution_failed", error, 2)["retryable"]


def test_exception_message_alone_does_not_authorize_retry():
    hint = recovery_hint("accident_case_search", "tool_execution_failed", sqlite3.OperationalError("database is locked"), 1)
    assert not hint["retryable"]


def test_parameter_feedback_contains_only_trusted_tool_schema():
    hint = recovery_hint("accident_case_search", "invalid_tool_arguments", ValueError("SECRET_BODY"), 1)
    assert hint["parameters"]["properties"]["limit"]["maximum"] == 5
    assert "SECRET_BODY" not in str(hint)
    assert hint["retryable"]


@pytest.mark.parametrize("code", ["tool_denied", "tool_timeout", "tool_worker_busy", "invalid_tool_observation"])
def test_permission_timeout_and_unknown_failures_are_not_retryable(code):
    assert not recovery_hint("accident_knowledge_search", code, ValueError(), 1)["retryable"]


def test_unknown_tool_and_exhausted_parameter_repair_are_not_retryable():
    assert not recovery_hint("unregistered", "invalid_tool_arguments", ValueError(), 1)["retryable"]
    assert not recovery_hint("accident_case_search", "invalid_tool_arguments", ValueError(), 2)["retryable"]
