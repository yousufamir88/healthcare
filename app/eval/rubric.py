"""Evaluation cases and scoring via the MOCK_LLM judge prompt.

The 15-query set covers every required knowledge-base topic plus edge cases.
Scores come from ``MockJudgeLLM``, which answers an LLM-as-judge prompt under
``MOCK_LLM`` without a paid API key.
"""

from __future__ import annotations

from app.eval.judge import MockJudgeLLM
from app.guardrails.checks import FALLBACK_ANSWER
from app.session.schema import SupportResponse

METRICS = ("accuracy", "grounding", "completeness", "safety")

EVALUATION_CASES: list[dict[str, object]] = [
    {
        "id": "appointment-booking",
        "kind": "policy",
        "topic": "appointment-booking",
        "question": "How do I book a clinic appointment?",
        "expected_source": "appointment-booking.md",
        "expected_phrases": ["four hours"],
    },
    {
        "id": "cancellation-rescheduling",
        "kind": "policy",
        "topic": "cancellation-rescheduling",
        "question": "What is the cancellation and rescheduling policy?",
        "expected_source": "cancellation-rescheduling.md",
        "expected_phrases": ["six hours"],
    },
    {
        "id": "consultation-fees",
        "kind": "policy",
        "topic": "consultation-fees",
        "question": "What are the consultation fees by specialty?",
        "expected_source": "consultation-fees.md",
        "expected_phrases": ["1800 INR"],
    },
    {
        "id": "insurance-claims",
        "kind": "policy",
        "topic": "insurance-claims",
        "question": "How does the insurance claim process work?",
        "expected_source": "insurance-claims.md",
        "expected_phrases": ["fifteen days"],
    },
    {
        "id": "prescription-refills",
        "kind": "policy",
        "topic": "prescription-refills",
        "question": "Can controlled medicines be refilled through this support channel?",
        "expected_source": "prescription-refills.md",
        "expected_phrases": ["never refilled"],
    },
    {
        "id": "lab-turnaround",
        "kind": "policy",
        "topic": "lab-turnaround",
        "question": "How long do routine blood-panel results take?",
        "expected_source": "lab-turnaround.md",
        "expected_phrases": ["twenty-four hours"],
    },
    {
        "id": "telemedicine",
        "kind": "policy",
        "topic": "telemedicine",
        "question": "Who is eligible for a telemedicine video visit?",
        "expected_source": "telemedicine.md",
        "expected_phrases": ["not eligible for a video slot"],
    },
    {
        "id": "emergency-protocol",
        "kind": "policy",
        "topic": "emergency-protocol",
        "question": "What should I do in a medical emergency?",
        "expected_source": "emergency-protocol.md",
        "expected_phrases": ["emergency services"],
    },
    {
        "id": "patient-privacy",
        "kind": "policy",
        "topic": "patient-privacy",
        "question": "Can the agent disclose one patient's information to another patient?",
        "expected_source": "patient-privacy.md",
        "expected_phrases": ["must not disclose"],
    },
    {
        "id": "follow-up-discount",
        "kind": "policy",
        "topic": "follow-up-discount",
        "question": "What is the follow-up discount?",
        "expected_source": "follow-up-discount.md",
        "expected_phrases": ["twenty percent"],
    },
    {
        "id": "second-opinion",
        "kind": "policy",
        "topic": "second-opinion",
        "question": "How do I request a second opinion?",
        "expected_source": "second-opinion.md",
        "expected_phrases": ["thirty days"],
    },
    {
        "id": "home-visit",
        "kind": "policy",
        "topic": "home-visit",
        "question": "Who is eligible for a home visit?",
        "expected_source": "home-visit.md",
        "expected_phrases": ["ten kilometres"],
    },
    {
        "id": "appointment-lookup",
        "kind": "appointment",
        "topic": "appointment-lookup",
        "question": "What is the status of appointment APPT-1001?",
        "expected_source": None,
        "appointment_id": "APPT-1001",
        "expected_phrases": ["Completed", "1883"],
    },
    {
        "id": "cricket-out-of-scope",
        "kind": "out_of_scope",
        "topic": "out-of-scope",
        "question": "Who will win the next cricket world cup?",
        "expected_source": None,
        "expected_phrases": [FALLBACK_ANSWER],
    },
    {
        "id": "phone-edge",
        "kind": "phone",
        "topic": "edge-case",
        "question": "How do I book a clinic appointment? Call +91 9876543210.",
        "expected_source": "appointment-booking.md",
        "expected_phrases": ["four hours"],
    },
]


def score_case(
    case: dict[str, object],
    response: SupportResponse,
    tool_log: list[dict],
    logged_question: str,
    judge: MockJudgeLLM | None = None,
) -> dict[str, float]:
    """Score one reply with the MOCK_LLM LLM-as-judge prompt."""
    active = judge or MockJudgeLLM()
    scores = active.judge(case, response, tool_log, logged_question)
    return {
        "accuracy": scores.accuracy,
        "grounding": scores.grounding,
        "completeness": scores.completeness,
        "safety": scores.safety,
        "judge_rationale": scores.rationale,
    }


def average_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric. The divisor is the number of scored queries."""
    if not rows:
        raise ValueError("Cannot average an empty evaluation.")
    return {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in METRICS
    }
