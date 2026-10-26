"""Tests for synthetic appointment lookup and escalation scores."""

import pytest

from app.data.dataset import APPOINTMENTS
from app.tools.appointment_tools import (
    ESCALATION_THRESHOLD,
    AppointmentLookupError,
    check_appointment_status,
    escalation_score,
    escalation_summary,
    escalation_threshold,
    normalized_recency,
)


def test_known_lookup_matches_the_dataset():
    appointment = APPOINTMENTS[0]
    result = check_appointment_status(appointment.record_id)

    assert result["found"] is True
    assert result["status"] == appointment.status.value
    assert result["consultation_fee_inr"] == appointment.consultation_fee_inr
    assert result["escalation_score"] == pytest.approx(
        escalation_score(
            appointment.follow_up_required,
            appointment.days_since_created,
        )
    )


def test_every_record_lookup_matches_the_dataset():
    for appointment in APPOINTMENTS:
        result = check_appointment_status(appointment.record_id)
        assert result["status"] == appointment.status.value
        assert result["consultation_fee_inr"] == appointment.consultation_fee_inr
        assert 0.0 <= result["escalation_score"] <= 1.0


def test_lookup_is_reproducible():
    first = check_appointment_status("APPT-1001")
    second = check_appointment_status(" APPT-1001 ")

    assert first == second


def test_unknown_record_returns_a_safe_result():
    result = check_appointment_status("APPT-9999")

    assert result["found"] is False
    assert result["status"] is None
    assert result["consultation_fee_inr"] is None
    assert result["escalation_score"] is None
    assert "APPT-9999" in result["error"]


def test_blank_record_id_raises_a_clear_error():
    with pytest.raises(AppointmentLookupError, match="record_id"):
        check_appointment_status("   ")


def test_score_uses_follow_up_and_recency():
    assert normalized_recency(0) == pytest.approx(1.0)
    assert normalized_recency(30) == pytest.approx(0.0)
    assert escalation_score(False, 30) == pytest.approx(0.0)
    assert escalation_score(True, 0) == pytest.approx(0.0)
    assert escalation_score(True, 30) == pytest.approx(1.0)
    assert escalation_score(True, 15) == pytest.approx(0.5)


def test_threshold_is_the_median_follow_up_score():
    follow_up_scores = [
        escalation_score(item.follow_up_required, item.days_since_created)
        for item in APPOINTMENTS
        if item.follow_up_required
    ]

    assert escalation_threshold() == pytest.approx(sorted(follow_up_scores)[10])
    assert ESCALATION_THRESHOLD == pytest.approx(14 / 30)


def test_escalation_flag_follows_the_measured_threshold():
    summary = escalation_summary()
    flagged = [
        check_appointment_status(item.record_id)
        for item in APPOINTMENTS
        if check_appointment_status(item.record_id)["requires_escalation"]
    ]

    assert summary["escalation_count"] == len(flagged)
    assert flagged
    assert all(item["follow_up_required"] for item in flagged)
    assert all(
        item["days_since_created"] >= 14
        for item in flagged
    )
    assert all(
        not check_appointment_status(item.record_id)["requires_escalation"]
        for item in APPOINTMENTS
        if not item.follow_up_required
    )
