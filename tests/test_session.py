"""Tests for session isolation and the support response schema."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import ValidationError

from app.rag.retrieval import FALLBACK_ANSWER
from app.session.memory import (
    ask,
    get_session_history,
    reset_sessions,
    resolve_question,
)
from app.session.schema import SupportIntent, SupportResponse, safe_support_response


def setup_function():
    reset_sessions()


def test_schema_accepts_a_complete_appointment_reply():
    response = SupportResponse(
        answer="Appointment APPT-1001 is Completed.",
        intent=SupportIntent.APPOINTMENT,
        sources=["appointment-booking.md"],
        appointment_id="APPT-1001",
        escalation_score=0.0,
        requires_human_review=False,
    )

    checked = SupportResponse.model_validate_json(response.model_dump_json())

    assert checked.appointment_id == "APPT-1001"
    assert checked.escalation_score == 0.0


def test_schema_rejects_an_unbounded_score_and_a_bad_id():
    with pytest.raises(ValidationError, match="escalation_score"):
        SupportResponse(
            answer="Too high.",
            intent=SupportIntent.APPOINTMENT,
            escalation_score=1.5,
        )

    with pytest.raises(ValidationError, match="appointment_id"):
        SupportResponse(
            answer="Bad id.",
            intent=SupportIntent.APPOINTMENT,
            appointment_id="patient-1",
        )


def test_invalid_payload_becomes_a_review_fallback():
    response = safe_support_response(
        {
            "answer": "",
            "intent": SupportIntent.POLICY,
            "escalation_score": 4,
        }
    )

    assert response.answer == FALLBACK_ANSWER
    assert response.intent is SupportIntent.UNKNOWN
    assert response.requires_human_review is True
    assert response.appointment_id is None


def test_follow_up_reuses_the_id_stored_in_history():
    history = [
        HumanMessage(content="What is the status of appointment APPT-1001?"),
        AIMessage(content='{"appointment_id": "APPT-1001", "answer": "Completed"}'),
    ]

    resolved = resolve_question(
        "What is the consultation fee for that appointment?",
        history,
    )

    assert "APPT-1001" in resolved


def test_fresh_history_does_not_invent_an_appointment_id():
    resolved = resolve_question(
        "What is the consultation fee for that appointment?",
        [],
    )

    assert "APPT-" not in resolved


def test_blank_session_id_is_rejected():
    with pytest.raises(ValueError, match="session_id"):
        get_session_history("   ")


def test_runnable_history_is_isolated_between_session_ids():
    seen: list[tuple[str, int]] = []

    def remember(payload: dict) -> str:
        seen.append((payload["question"], len(payload.get("history") or [])))
        return "stored"

    chain = RunnableWithMessageHistory(
        RunnableLambda(remember),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    config_a = {"configurable": {"session_id": "session-a"}}
    config_b = {"configurable": {"session_id": "session-b"}}

    chain.invoke({"question": "first"}, config=config_a)
    chain.invoke({"question": "second"}, config=config_a)
    chain.invoke({"question": "other"}, config=config_b)

    assert seen == [("first", 0), ("second", 2), ("other", 0)]
    assert get_session_history("session-a") is get_session_history("session-a")
    assert get_session_history("session-a") is not get_session_history("session-b")
    assert len(get_session_history("session-a").messages) == 4
    assert len(get_session_history("session-b").messages) == 2


@pytest.mark.integration
def test_same_session_reuses_the_appointment_and_a_new_session_does_not():
    first = ask("clinic-session-a", "What is the status of appointment APPT-1001?")
    second = ask(
        "clinic-session-a",
        "What is the consultation fee for that appointment?",
    )
    fresh = ask(
        "clinic-session-b",
        "What is the consultation fee for that appointment?",
    )

    assert first.appointment_id == "APPT-1001"
    assert first.escalation_score == 0.0
    assert "Completed" in first.answer
    assert "1883" in first.answer

    assert second.appointment_id == "APPT-1001"
    assert second.escalation_score == 0.0
    assert "1883" in second.answer
    assert len(get_session_history("clinic-session-a").messages) == 4

    assert fresh.appointment_id is None
    assert fresh.escalation_score is None
    assert "APPT-1001" not in fresh.answer
    assert "1883" not in fresh.answer
    assert len(get_session_history("clinic-session-b").messages) == 2
    session_b_text = " ".join(
        str(message.content) for message in get_session_history("clinic-session-b").messages
    )
    assert "APPT-1001" not in session_b_text
