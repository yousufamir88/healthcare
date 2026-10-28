"""A CrewAI BaseLLM that runs the support demo without an API key.

CrewAI passes the tools that belong to the current agent as JSON schemas.
This model reads those schemas and returns a native tool call. It does not
decide which tool to use by searching the system prompt for a tool name.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM

from app.rag.retrieval import FALLBACK_ANSWER
from app.session.schema import SupportIntent, SupportResponse, response_format

QUESTION_MARKER = "SUPPORT_QUESTION:"
_QUESTION = re.compile(r"SUPPORT_QUESTION:\s*(.+)")
_RECORD_ID = re.compile(r"APPT-\d+")

_CALLS: list[dict[str, Any]] = []
_CALL_NUMBER = 0


def reset_llm_log() -> None:
    """Clear the record of mock model calls. Tests use this between runs."""
    global _CALL_NUMBER
    _CALLS.clear()
    _CALL_NUMBER = 0


def llm_calls() -> list[dict[str, Any]]:
    """Return the calls made since the last reset."""
    return list(_CALLS)


class MockLLM(BaseLLM):
    """Local model used by every graded crew run."""

    def __init__(self) -> None:
        super().__init__(model="mock-llm", provider="mock", temperature=0.0)

    def supports_function_calling(self) -> bool:
        """Ask CrewAI to use native tool calls instead of ReAct text."""
        return True

    def supports_stop_words(self) -> bool:
        """This stand-in does not use provider stop sequences."""
        return False

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type | None = None,
    ) -> str | list[dict[str, Any]]:
        """Return a tool call or a final answer built from tool evidence."""
        del callbacks, available_functions, from_task, from_agent
        text = _message_text(messages)
        tool_output = _latest_tool_output(messages)
        if tool_output is not None:
            _record("final", tool_output)
            return tool_output

        tool_call = _tool_call_from_schema(tools or [], text)
        if tool_call is not None:
            _record("tool_call", tool_call)
            return [tool_call]

        if _schema_has_property(tools or [], "record_id"):
            skipped = json.dumps(
                {
                    "tool": "lookup_skipped",
                    "answer": (
                        "No appointment record id was provided, so the "
                        "appointment lookup tool was not called."
                    ),
                }
            )
            _record("final", skipped)
            return skipped

        structured = compose_support_response(text)
        rendered = structured.model_dump_json()
        _record("final", rendered)
        if response_model is response_format or response_model is SupportResponse:
            return rendered
        return rendered


def compose_from_evidence(text: str) -> str:
    """Build the composer answer from tool JSON already present in the prompt."""
    return compose_support_response(text).answer


def compose_support_response(text: str) -> SupportResponse:
    """Build a ``response_format`` / ``SupportResponse`` object from tool evidence."""
    payloads = _extract_payloads(text)
    parts: list[str] = []
    sources: list[str] = []
    appointment_id: str | None = None
    escalation_score: float | None = None
    requires_human_review = False
    saw_policy = False
    saw_appointment = False
    seen: set[str] = set()
    for payload in payloads:
        key = json.dumps(payload, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        tool_name = payload.get("tool")
        if tool_name == "search_policies":
            saw_policy = True
            for source in payload.get("sources") or []:
                if source not in sources:
                    sources.append(str(source))
            if payload.get("fallback") is True:
                requires_human_review = True
        elif tool_name == "check_appointment_status":
            saw_appointment = True
            if payload.get("found") is True:
                appointment_id = str(payload.get("record_id"))
                escalation_score = float(payload["escalation_score"])
                if payload.get("requires_escalation") is True:
                    requires_human_review = True
            else:
                requires_human_review = True
        rendered = _render_payload(payload)
        if rendered:
            parts.append(rendered)
    if saw_policy and saw_appointment:
        intent = SupportIntent.MIXED
    elif saw_appointment:
        intent = SupportIntent.APPOINTMENT
    elif saw_policy:
        intent = SupportIntent.POLICY
    else:
        intent = SupportIntent.UNKNOWN
        requires_human_review = True
    answer = " ".join(parts) if parts else FALLBACK_ANSWER
    return response_format.model_validate(
        {
            "answer": answer,
            "intent": intent,
            "sources": sources,
            "appointment_id": appointment_id,
            "escalation_score": escalation_score,
            "requires_human_review": requires_human_review,
        }
    )


def _tool_call_from_schema(
    tools: list[dict[str, Any]],
    text: str,
) -> dict[str, Any] | None:
    """Choose a tool from its argument schema and fill those arguments."""
    record_match = _RECORD_ID.search(text)
    question = _extract_question(text)
    query_call: dict[str, Any] | None = None

    for schema in _function_schemas(tools):
        name = schema.get("name")
        properties = _properties(schema)
        if not isinstance(name, str) or not properties:
            continue
        if "record_id" in properties and record_match is not None:
            return _call(name, {"record_id": record_match.group(0)})
        if "query" in properties and query_call is None:
            query_call = _call(name, {"query": question})

    return query_call


def _call(name: str, arguments: dict[str, str]) -> dict[str, Any]:
    global _CALL_NUMBER
    _CALL_NUMBER += 1
    return {
        "id": f"call_{_CALL_NUMBER}",
        "function": {"name": name, "arguments": arguments},
    }


def _schema_has_property(tools: list[dict[str, Any]], property_name: str) -> bool:
    return any(property_name in _properties(schema) for schema in _function_schemas(tools))


def _function_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if isinstance(function, dict):
            schemas.append(function)
    return schemas


def _properties(schema: dict[str, Any]) -> set[str]:
    parameters = schema.get("parameters") or {}
    if not isinstance(parameters, dict):
        return set()
    properties = parameters.get("properties") or {}
    if not isinstance(properties, dict):
        return set()
    return set(properties)


def _extract_question(text: str) -> str:
    match = _QUESTION.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _latest_tool_output(messages: str | list[Any]) -> str | None:
    if isinstance(messages, str):
        return None
    outputs = [
        str(message.get("content", "")).strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    outputs = [output for output in outputs if output]
    if not outputs:
        return None
    return outputs[-1]


def _message_text(messages: str | list[Any]) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for message in messages:
        if isinstance(message, str):
            parts.append(message)
            continue
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
        elif content:
            parts.append(str(content))
    return "\n".join(parts)


def _extract_payloads(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            payloads.append(obj)
        index = max(end, start + 1)
    return payloads


def _render_payload(payload: dict[str, Any]) -> str:
    tool_name = payload.get("tool")
    if tool_name == "search_policies":
        answer = str(payload.get("answer", "")).strip()
        sources = payload.get("sources") or []
        if sources:
            return f"{answer} Sources: {', '.join(str(source) for source in sources)}."
        return answer
    if tool_name == "check_appointment_status":
        if payload.get("found") is True:
            return (
                f"Appointment {payload.get('record_id')} is {payload.get('status')}. "
                f"The consultation fee is {payload.get('consultation_fee_inr')} INR. "
                f"The escalation score is {payload.get('escalation_score')}."
            )
        error = payload.get("error")
        if error:
            return str(error)
        return "No synthetic appointment exists for that record id."
    if tool_name == "lookup_skipped":
        return str(payload.get("answer", "")).strip()
    return ""


def _record(kind: str, payload: Any) -> None:
    _CALLS.append({"kind": kind, "payload": payload})
