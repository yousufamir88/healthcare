"""Session history for the support crew.

Each session id has its own ``InMemoryChatMessageHistory``. A follow-up in
that session can reuse an appointment id from earlier turns. A different
session id starts empty.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.crew.support_crew import run_support_crew
from app.crew.tools import reset_tool_log, tool_calls
from app.governance.controls import (
    assess_budget,
    budget_response,
    classify_risk,
    normalize_query,
    recall,
    remember,
)
from app.guardrails.checks import (
    apply_output_guard,
    injection_response,
    mask_phone_numbers,
    prompt_injection_detected,
    write_request_log,
)
from app.review.stage import review_draft
from app.session.schema import (
    SupportIntent,
    SupportResponse,
    response_format,
    safe_support_response,
)

_RECORD_ID = re.compile(r"APPT-\d+")
_FOLLOW_UP = re.compile(
    r"\b(that|this|same|it|its|the appointment)\b",
    re.IGNORECASE,
)

_SESSIONS: dict[str, InMemoryChatMessageHistory] = {}


def reset_sessions() -> None:
    """Drop every stored conversation. Tests call this before a demo."""
    _SESSIONS.clear()


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the history object for one session, creating it on first use."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    key = session_id.strip()
    history = _SESSIONS.get(key)
    if history is None:
        history = InMemoryChatMessageHistory()
        _SESSIONS[key] = history
    return history


def resolve_question(question: str, history: list[Any]) -> str:
    """Reuse the latest appointment id when this turn refers to an earlier one."""
    if _RECORD_ID.search(question):
        return question
    if _FOLLOW_UP.search(question) is None:
        return question
    remembered = _latest_record_id(history)
    if remembered is None:
        return question
    return f"{question} The earlier record id is {remembered}."


def response_from_tool_log(answer: str) -> SupportResponse:
    """Fill the schema from the tool results of the crew run that just finished."""
    sources: list[str] = []
    appointment_id: str | None = None
    escalation_score: float | None = None
    requires_human_review = False
    saw_policy = False
    saw_appointment = False

    for call in tool_calls():
        result = call.get("result") or {}
        if call.get("tool") == "search_policies":
            saw_policy = True
            for source in result.get("sources") or []:
                if source not in sources:
                    sources.append(str(source))
            if result.get("fallback") is True:
                requires_human_review = True
        elif call.get("tool") == "check_appointment_status":
            saw_appointment = True
            if result.get("found") is True:
                appointment_id = str(result.get("record_id"))
                escalation_score = float(result["escalation_score"])
                if result.get("requires_escalation") is True:
                    requires_human_review = True
            else:
                requires_human_review = True

    if saw_policy and saw_appointment:
        intent = SupportIntent.MIXED
    elif saw_appointment:
        intent = SupportIntent.APPOINTMENT
    elif saw_policy:
        intent = SupportIntent.POLICY
    else:
        intent = SupportIntent.UNKNOWN
        requires_human_review = True

    return response_format.model_validate(
        safe_support_response(
            {
                "answer": answer,
                "intent": intent,
                "sources": sources,
                "appointment_id": appointment_id,
                "escalation_score": escalation_score,
                "requires_human_review": requires_human_review,
            }
        ).model_dump()
    )


def build_session_chain() -> RunnableWithMessageHistory:
    """Wrap one crew turn so LangChain stores it under the session id."""
    return RunnableWithMessageHistory(
        RunnableLambda(_run_turn),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def ask(session_id: str, question: str) -> SupportResponse:
    """Ask one question in a session and validate the crew reply."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    session_key = session_id.strip()
    masked = mask_phone_numbers(question.strip())
    started = time.perf_counter()
    resolved = masked
    if prompt_injection_detected(question) or prompt_injection_detected(masked):
        response = injection_response()
        outcome = "rejected_injection"
        risk = classify_risk(masked, requires_human_review=True)
    else:
        budget = assess_budget(masked)
        if not budget.allowed:
            response = budget_response()
            outcome = "rejected_budget"
            risk = classify_risk(masked, requires_human_review=True)
        else:
            history = list(get_session_history(session_key).messages)
            resolved = mask_phone_numbers(resolve_question(masked, history))
            cached = recall(normalize_query(resolved))
            if cached is not None:
                reset_tool_log()
                get_session_history(session_key).add_user_message(masked)
                get_session_history(session_key).add_ai_message(cached.model_dump_json())
                response = cached
                outcome = "cache_hit"
            else:
                raw = build_session_chain().invoke(
                    {"question": masked},
                    config={"configurable": {"session_id": session_key}},
                )
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("The session chain returned an empty response.")
                response = SupportResponse.model_validate_json(raw)
                remember(normalize_query(resolved), response)
                outcome = "answered"
            risk = classify_risk(
                resolved,
                requires_human_review=response.requires_human_review,
            )
    write_request_log(
        session_id=session_key,
        question=masked,
        outcome=outcome,
        duration_ms=(time.perf_counter() - started) * 1000,
        risk=risk,
    )
    return response


def _run_turn(payload: dict[str, Any]) -> str:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    masked = mask_phone_numbers(question.strip())
    if prompt_injection_detected(masked):
        return injection_response().model_dump_json()
    history = payload.get("history") or []
    resolved = mask_phone_numbers(resolve_question(masked, list(history)))
    answer = run_support_crew(resolved)
    calls = tool_calls()
    response = apply_output_guard(response_from_tool_log(answer), calls)
    verdict = review_draft(response.answer, calls)
    if verdict.final_answer != response.answer:
        response = response.model_copy(
            update={
                "answer": verdict.final_answer,
                "requires_human_review": True,
            }
        )
    return response.model_dump_json()


def _latest_record_id(history: list[Any]) -> str | None:
    found: str | None = None
    for message in history:
        match = None
        for match in _RECORD_ID.finditer(_message_text(message)):
            pass
        if match is not None:
            found = match.group(0)
    return found


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return "" if content is None else str(content)


def main() -> int:
    """Show one remembered follow-up and one fresh session."""
    reset_sessions()
    turns = (
        ("clinic-session-a", "What is the status of appointment APPT-1001?"),
        ("clinic-session-a", "What is the consultation fee for that appointment?"),
        ("clinic-session-b", "What is the consultation fee for that appointment?"),
    )
    lines = [
        "Session memory demonstration",
        "Mode: MOCK_LLM",
        "History: InMemoryChatMessageHistory",
    ]
    for session_id, question in turns:
        response = ask(session_id, question)
        history = get_session_history(session_id)
        lines.extend(
            [
                "",
                f"Session: {session_id}",
                f"Question: {question}",
                f"Stored messages: {len(history.messages)}",
                f"Intent: {response.intent.value}",
                f"Appointment id: {response.appointment_id}",
                f"Escalation score: {response.escalation_score}",
                f"Requires human review: {response.requires_human_review}",
                f"Sources: {', '.join(response.sources) if response.sources else 'none'}",
                f"Answer: {response.answer}",
            ]
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
