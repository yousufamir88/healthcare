"""Tests for the synthetic appointment dataset."""

import pytest
from pydantic import ValidationError

from app.data.dataset import (
    APPOINTMENTS,
    CATEGORY_WEIGHTS,
    CONSULTATION_FEE_RANGE,
    DAYS_SINCE_CREATED_RANGE,
    SEED,
    STATUS_WEIGHTS,
    Appointment,
    Category,
    DatasetValidationError,
    Status,
    format_dataset_report,
    generate_appointments,
    validate_dataset,
)


def test_dataset_is_valid():
    report = validate_dataset(APPOINTMENTS)

    assert report["record_count"] >= 40
    assert report["seed"] == SEED


def test_all_categories_are_present():
    actual = {item.category for item in APPOINTMENTS}

    assert set(Category).issubset(actual)


def test_all_statuses_are_present():
    actual = {item.status for item in APPOINTMENTS}

    assert set(Status).issubset(actual)


def test_record_ids_are_unique():
    ids = [item.record_id for item in APPOINTMENTS]

    assert len(ids) == len(set(ids))
    assert ids[0] == "APPT-1001"


def test_generation_is_deterministic():
    first = [item.model_dump() for item in generate_appointments()]
    second = [item.model_dump() for item in generate_appointments()]

    assert first == second
    assert [item.model_dump() for item in APPOINTMENTS] == first


def test_each_category_has_at_least_three_records():
    report = validate_dataset(APPOINTMENTS)

    assert all(count >= 3 for count in report["category_counts"].values())


def test_days_and_fees_stay_inside_documented_ranges():
    for item in APPOINTMENTS:
        assert DAYS_SINCE_CREATED_RANGE[0] <= item.days_since_created <= DAYS_SINCE_CREATED_RANGE[1]
        assert CONSULTATION_FEE_RANGE[0] <= item.consultation_fee_inr <= CONSULTATION_FEE_RANGE[1]


def test_follow_up_percentage_is_inside_the_required_band():
    report = validate_dataset(APPOINTMENTS)

    assert 10 <= report["follow_up_percentage"] <= 30


def test_report_prints_counts_and_follow_up_percentage():
    report = validate_dataset(APPOINTMENTS)
    text = format_dataset_report(report)

    assert "Category counts:" in text
    assert "Status counts:" in text
    for category in Category:
        assert category.value in text
    for status in Status:
        assert status.value in text
    assert f"({report['follow_up_percentage']:.2f}%)" in text


def test_seed_42_measured_distribution_stays_stable():
    report = validate_dataset(APPOINTMENTS)

    assert report["category_counts"] == {
        "General Medicine": 20,
        "Cardiology": 24,
        "Dermatology": 15,
        "Pediatrics": 15,
        "Orthopedics": 26,
    }
    assert report["status_counts"] == {
        "Scheduled": 38,
        "Completed": 35,
        "Cancelled": 10,
        "No-Show": 5,
        "Rescheduled": 12,
    }
    assert report["follow_up_count"] == 21
    assert report["follow_up_percentage"] == 21.0


def test_too_few_records_raise_a_clear_error():
    with pytest.raises(DatasetValidationError, match="at least 40"):
        validate_dataset(APPOINTMENTS[:10])


def test_duplicate_record_ids_raise_a_clear_error():
    records = [item.model_copy() for item in APPOINTMENTS]
    records[1] = records[1].model_copy(update={"record_id": records[0].record_id})

    with pytest.raises(DatasetValidationError, match="record_id"):
        validate_dataset(records)


def test_missing_category_raises_a_clear_error():
    records = [
        item.model_copy(update={"category": Category.CARDIOLOGY})
        if item.category is Category.ORTHOPEDICS
        else item.model_copy()
        for item in APPOINTMENTS
    ]

    with pytest.raises(DatasetValidationError, match="Orthopedics"):
        validate_dataset(records)


def test_follow_up_share_outside_the_band_raises_a_clear_error():
    records = [
        item.model_copy(update={"follow_up_required": False})
        for item in APPOINTMENTS
    ]

    with pytest.raises(DatasetValidationError, match="10% to 30%"):
        validate_dataset(records)


def test_invalid_record_count_is_rejected_before_generation():
    with pytest.raises(DatasetValidationError, match="RECORD_COUNT"):
        generate_appointments(record_count=10)


def test_invalid_weights_are_rejected():
    with pytest.raises(DatasetValidationError, match="CATEGORY_WEIGHTS"):
        generate_appointments(category_weights=[1, 1])

    with pytest.raises(DatasetValidationError, match="STATUS_WEIGHTS"):
        generate_appointments(status_weights=[0, 1, 1, 1, 1])


def test_invalid_ranges_and_follow_up_probability_are_rejected():
    with pytest.raises(DatasetValidationError, match="days_since_created"):
        generate_appointments(days_range=(-1, 40))

    with pytest.raises(DatasetValidationError, match="consultation_fee_inr"):
        generate_appointments(fee_range=(100, 5000))

    with pytest.raises(DatasetValidationError, match="FOLLOW_UP_PROBABILITY"):
        generate_appointments(follow_up_probability=0.8)


def test_appointment_model_rejects_an_impossible_day_count():
    source = APPOINTMENTS[0].model_dump()
    source["days_since_created"] = 31

    with pytest.raises(ValidationError):
        Appointment.model_validate(source)


def test_documented_weights_match_the_generator_defaults():
    assert CATEGORY_WEIGHTS == [20, 20, 20, 20, 20]
    assert STATUS_WEIGHTS == [35, 30, 10, 10, 15]
    assert CONSULTATION_FEE_RANGE == (300, 2000)
