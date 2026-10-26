"""Structured support response checked before it leaves a session."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.rag.retrieval import FALLBACK_ANSWER

_RECORD_ID = re.compile(r"^APPT-\d+$")


class SupportIntent(str, Enum):
    """Why the crew answered."""

    POLICY = "policy"
    APPOINTMENT = "appointment"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SupportResponse(BaseModel):
    """Every crew reply must match this shape.

    ``appointment_id`` and ``escalation_score`` stay empty when the turn
    did not look up a record. ``sources`` lists policy files that cleared
    the retrieval threshold. ``requires_human_review`` is true when the
    evidence is missing or the appointment score is at or above the cutoff.
    """

    answer: str = Field(min_length=1)
    intent: SupportIntent
    sources: list[str] = Field(default_factory=list)
    appointment_id: str | None = None
    escalation_score: float | None = None
    requires_human_review: bool = False

    @field_validator("appointment_id")
    @classmethod
    def appointment_id_matches_dataset_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _RECORD_ID.fullmatch(value) is None:
            raise ValueError(
                "appointment_id must look like APPT-1001. "
                f"Received {value!r}."
            )
        return value

    @field_validator("escalation_score")
    @classmethod
    def escalation_score_stays_inside_zero_and_one(
        cls,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "escalation_score must be between 0 and 1. "
                f"Received {value}."
            )
        return value


def safe_support_response(data: dict) -> SupportResponse:
    """Return a valid response, or a review fallback when the data is invalid."""
    try:
        return SupportResponse.model_validate(data)
    except ValidationError:
        return SupportResponse(
            answer=FALLBACK_ANSWER,
            intent=SupportIntent.UNKNOWN,
            requires_human_review=True,
        )
