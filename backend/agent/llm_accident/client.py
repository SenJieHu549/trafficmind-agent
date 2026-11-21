"""Native tool_calls transport using the project's existing SDK/config.

No changes to the planner provider or the legacy ReAct path. SDK retries are
disabled so one runtime counter represents one attempted HTTP completion.
"""
from copy import deepcopy

from backend.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from backend.agent.llm_accident.contracts import AgentRuntimeError


class AccidentLLMClient:
    def __init__(self, *, api_key=None, base_url=None, model=None):
        self.api_key = DEEPSEEK_API_KEY if api_key is None else api_key
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.model = model or DEEPSEEK_MODEL

    async def complete(self, messages, tools, *, allow_tools, timeout):
        if not self.api_key or self.api_key == "your_api_key":
            raise AgentRuntimeError("llm_unavailable")
        from openai import AsyncOpenAI
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url,
                               max_retries=0, timeout=timeout) as client:
            kwargs = {"model": self.model, "messages": deepcopy(messages),
                      "temperature": .1, "max_tokens": 1800}
            if tools:
                kwargs.update(tools=deepcopy(tools), tool_choice="auto" if allow_tools else "none")
            else:
                kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise AgentRuntimeError("llm_incomplete_response")
        message = choice.message.model_dump(exclude_none=True)
        # No private reasoning or raw provider body is persisted in the trace.
        return {"message": {k: message[k] for k in ("role", "content", "tool_calls") if k in message},
                "usage": response.usage.model_dump() if response.usage else {},
                "model": response.model}
