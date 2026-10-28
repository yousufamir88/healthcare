"""Root entry point for the capstone ``dataset.py`` deliverable.

Generation and validation live in ``app.data.dataset``. This module keeps the
repository-root name the brief asks for and re-exports the same API.
"""

from __future__ import annotations

from app.data.dataset import (
    APPOINTMENTS,
    CATEGORY_WEIGHTS,
    CONSULTATION_FEE_RANGE,
    DAYS_SINCE_CREATED_RANGE,
    FOLLOW_UP_PROBABILITY,
    RECORD_COUNT,
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

__all__ = [
    "APPOINTMENTS",
    "CATEGORY_WEIGHTS",
    "CONSULTATION_FEE_RANGE",
    "DAYS_SINCE_CREATED_RANGE",
    "FOLLOW_UP_PROBABILITY",
    "RECORD_COUNT",
    "SEED",
    "STATUS_WEIGHTS",
    "Appointment",
    "Category",
    "DatasetValidationError",
    "Status",
    "format_dataset_report",
    "generate_appointments",
    "validate_dataset",
]


if __name__ == "__main__":
    print(format_dataset_report(validate_dataset(APPOINTMENTS)))
