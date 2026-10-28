"""AutoGen review: an approved draft stays, and a flawed draft is replaced."""

import pytest
from pydantic import ValidationError

from app.rag.retrieval import FALLBACK_ANSWER
from autogen_agentchat.messages import StructuredMessage

from app.review.stage import (
    EDITOR_NAME,
    REVIEWER_NAME,
    ReviewVerdict,
    review_transcript,
)

EVIDENCE = [
    {
        "tool": "search_policies",
        "result": {
            "answer": (
                "A booked appointment may be cancelled or rescheduled "
                "until six hours before the scheduled start time."
            ),
            "sources": ["cancellation-rescheduling.md"],
            "fallback": False,
        },
    }
]

GROUNDED = (
    "A booked appointment may be cancelled or rescheduled until "
    "six hours before the scheduled start time. "
    "Sources: cancellation-rescheduling.md."
)


def test_grounded_draft_is_approved_and_the_answer_stays():
    verdict, speakers = review_transcript(GROUNDED, EVIDENCE)

    assert speakers == [REVIEWER_NAME, EDITOR_NAME]
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is True
    assert verdict.final_answer == GROUNDED


def test_final_editor_uses_structured_verdict_wiring():
    import inspect

    from app.review.stage import _run_team

    source = inspect.getsource(_run_team)
    assert "output_content_type=ReviewVerdict" in source
    assert "custom_message_types=[StructuredMessage[ReviewVerdict]]" in source
    assert StructuredMessage[ReviewVerdict] is not None
    verdict, speakers = review_transcript(GROUNDED, EVIDENCE)
    assert speakers == [REVIEWER_NAME, EDITOR_NAME]
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is True


def test_medical_draft_is_replaced_with_the_fallback():
    draft = "You should take 500mg of ibuprofen for the pain."
    verdict, speakers = review_transcript(draft, EVIDENCE)

    assert speakers == [REVIEWER_NAME, EDITOR_NAME]
    assert verdict.approved is False
    assert verdict.final_answer == FALLBACK_ANSWER
    assert verdict.final_answer != draft


def test_invented_policy_sentence_is_replaced():
    draft = "A booked appointment may be cancelled until two hours before the start."
    verdict, _speakers = review_transcript(draft, EVIDENCE)

    assert verdict.approved is False
    assert verdict.final_answer == FALLBACK_ANSWER


def test_phone_number_in_the_draft_is_replaced():
    draft = "Call +91 9876543210 to cancel six hours before the visit."
    verdict, _speakers = review_transcript(draft, EVIDENCE)

    assert verdict.approved is False
    assert "9876543210" not in verdict.final_answer


def test_verdict_requires_a_final_answer_and_a_reason():
    with pytest.raises(ValidationError):
        ReviewVerdict(approved=True, final_answer="", reason="missing answer")
    with pytest.raises(ValidationError):
        ReviewVerdict(approved=False, final_answer="kept", reason="")
