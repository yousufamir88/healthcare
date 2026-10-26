"""Look up a synthetic appointment and score whether it needs escalation.

The records come from the seeded dataset. There are no real patients.
An unknown id returns a dictionary with found=False instead of raising
KeyError.
"""

from __future__ import annotations

from app.data.dataset import APPOINTMENTS, DAYS_SINCE_CREATED_RANGE, Appointment

MAX_DAYS = DAYS_SINCE_CREATED_RANGE[1]


class AppointmentLookupError(ValueError):
    """Raised when a lookup cannot be attempted because the id is unusable."""


def normalized_recency(days_since_created: int) -> float:
    """Map age onto 0–1, where 1 means the appointment was created today."""
    if MAX_DAYS <= 0:
        raise AppointmentLookupError(
            "days_since_created has no positive upper bound, so recency "
            "cannot be normalized."
        )
    if not 0 <= days_since_created <= MAX_DAYS:
        raise AppointmentLookupError(
            f"days_since_created must be between 0 and {MAX_DAYS}. "
            f"Received {days_since_created}."
        )
    return (MAX_DAYS - days_since_created) / MAX_DAYS


def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    """Combine the follow-up flag with how far normalized recency has fallen.

    normalized_recency is 1 on day 0 and 0 on day 30.
    escalation_score = follow_up_signal * (1 - normalized_recency)
                     = follow_up_signal * days_since_created / 30

    A record with no follow-up scores 0. A follow-up scores higher as it
    gets older. The result is always between 0 and 1.
    """
    follow_up_signal = 1.0 if follow_up_required else 0.0
    score = follow_up_signal * (1.0 - normalized_recency(days_since_created))
    if not 0.0 <= score <= 1.0:
        raise AppointmentLookupError(
            f"Escalation score {score} is outside 0 to 1."
        )
    return score


def median(values: list[float]) -> float:
    """Return the middle value, or the mean of the two middle values."""
    if not values:
        raise AppointmentLookupError(
            "Cannot calculate an escalation threshold from an empty score list."
        )
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def escalation_threshold(appointments: list[Appointment] | None = None) -> float:
    """Use the median follow-up score in this dataset as the cutoff.

    Records with no follow-up score 0, so they are left out of the median.
    The cutoff is therefore the middle age, as a fraction of 30 days, among
    the appointments that actually require follow-up.
    """
    records = APPOINTMENTS if appointments is None else appointments
    follow_up_scores = [
        escalation_score(item.follow_up_required, item.days_since_created)
        for item in records
        if item.follow_up_required
    ]
    return median(follow_up_scores)


_INDEX = {item.record_id: item for item in APPOINTMENTS}
ESCALATION_THRESHOLD = escalation_threshold(APPOINTMENTS)


def check_appointment_status(record_id: str) -> dict:
    """Return status, fee, and escalation score for one synthetic record."""
    if not isinstance(record_id, str) or not record_id.strip():
        raise AppointmentLookupError(
            "record_id must be a non-empty string such as 'APPT-1001'."
        )

    cleaned = record_id.strip()
    appointment = _INDEX.get(cleaned)
    if appointment is None:
        return {
            "record_id": cleaned,
            "found": False,
            "status": None,
            "consultation_fee_inr": None,
            "escalation_score": None,
            "error": (
                "No synthetic appointment exists for record_id "
                f"{cleaned!r}."
            ),
        }

    score = escalation_score(
        appointment.follow_up_required,
        appointment.days_since_created,
    )
    return {
        "record_id": appointment.record_id,
        "found": True,
        "status": appointment.status.value,
        "consultation_fee_inr": appointment.consultation_fee_inr,
        "escalation_score": score,
        "follow_up_required": appointment.follow_up_required,
        "days_since_created": appointment.days_since_created,
        "normalized_recency": normalized_recency(appointment.days_since_created),
        "escalation_threshold": ESCALATION_THRESHOLD,
        "requires_escalation": score >= ESCALATION_THRESHOLD,
    }


def escalation_summary(appointments: list[Appointment] | None = None) -> dict:
    """Summarize the scores used to justify the threshold."""
    records = APPOINTMENTS if appointments is None else appointments
    scores = [
        escalation_score(item.follow_up_required, item.days_since_created)
        for item in records
    ]
    follow_up_scores = [
        escalation_score(item.follow_up_required, item.days_since_created)
        for item in records
        if item.follow_up_required
    ]
    threshold = median(follow_up_scores)
    return {
        "record_count": len(records),
        "follow_up_count": len(follow_up_scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "follow_up_score_min": min(follow_up_scores),
        "follow_up_score_max": max(follow_up_scores),
        "threshold": threshold,
        "escalation_count": sum(score >= threshold for score in scores),
    }


def format_lookup_report(example_id: str = "APPT-1001") -> str:
    """Text printed by ``python -m app.tools.appointment_tools``."""
    summary = escalation_summary()
    known = check_appointment_status(example_id)
    unknown = check_appointment_status("APPT-9999")
    lines = [
        "Appointment lookup",
        "Formula: escalation_score = follow_up_signal * (1 - normalized_recency)",
        "normalized_recency = (30 - days_since_created) / 30",
        f"Records: {summary['record_count']}",
        f"Follow-up records: {summary['follow_up_count']}",
        (
            "Follow-up scores: "
            f"{summary['follow_up_score_min']:.6f} to "
            f"{summary['follow_up_score_max']:.6f}"
        ),
        (
            "Threshold: median follow-up score "
            f"{summary['threshold']:.6f}"
        ),
        (
            "Records at or above the threshold: "
            f"{summary['escalation_count']}"
        ),
        f"Known {example_id}: {known}",
        f"Unknown APPT-9999: {unknown}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_lookup_report())
