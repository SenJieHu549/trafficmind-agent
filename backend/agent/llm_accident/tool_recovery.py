"""Trusted recovery hints; never infer retryability from exception messages."""
import sqlite3

from backend.agent.llm_accident.tools import TOOL_MODELS


def recovery_hint(name, code, error, failure_count):
    remaining = 1 if failure_count == 1 else 0
    schema = TOOL_MODELS.get(name)
    busy_code = getattr(error, "sqlite_errorcode", None)
    temporary = (type(busy_code) is int and busy_code & 255 in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
    citation_note = "引用必须保留服务端提供的 evidenceId，不得改写成完整 contentHash。"
    if code == "invalid_tool_arguments" and schema and remaining:
        return {"category": "argument_validation", "retryable": True, "remainingRetries": remaining,
            "parameters": schema.model_json_schema(),
            "instruction": "参数未通过校验，本次未执行查询。按参数 Schema 修正后重试一次，保持当前事件绑定，不得新增权限或身份参数。" + citation_note}
    if code == "tool_execution_failed" and temporary and remaining:
        return {"category": "transient_query_failure", "retryable": True, "remainingRetries": remaining,
            "instruction": "只读查询发生可识别的临时数据库忙错误。请在原调用预算内对同一工具用相同参数重试一次；仍失败则停止该查询，说明证据缺失，不得伪造查询结果。" + citation_note}
    return {"category": "retry_exhausted" if failure_count > 1 else "non_retryable_or_unknown",
        "retryable": False, "remainingRetries": 0,
        "instruction": "不要自动重复该查询。保留信息缺失，依据已有有效证据给出有限结论，必要时进入人工复核。" + citation_note}
