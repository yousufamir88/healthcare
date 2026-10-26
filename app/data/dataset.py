"""Synthetic clinic appointments for the Practo support-agent capstone.

Every record is fictional. The generator uses a fixed seed so the same file
always produces the same appointments. Validation checks the capstone rules
and raises DatasetValidationError with a specific message when a rule fails.
"""

from __future__ import annotations

import random
from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field, ValidationError

SEED = 42
RECORD_COUNT = 100

# Equal weights give every specialty the same chance after the initial
# coverage record. The seed then decides the exact counts.
CATEGORY_WEIGHTS = [20, 20, 20, 20, 20]

# Scheduled and completed visits are common. Cancellations and no-shows are
# less common. Rescheduled sits between those groups.
STATUS_WEIGHTS = [35, 30, 10, 10, 15]

# Fictional INR fees for this demo, not live Practo prices.
CONSULTATION_FEE_RANGE = (300, 2000)

DAYS_SINCE_CREATED_RANGE = (0, 30)

# Target rate inside the required 10%–30% band. The actual share is checked
# after generation because each record draws its own follow-up flag.
FOLLOW_UP_PROBABILITY = 0.20

MINIMUM_RECORD_COUNT = 40
MINIMUM_PER_CATEGORY = 3
FOLLOW_UP_PERCENT_RANGE = (10, 30)


class DatasetValidationError(ValueError):
    """Raised when the dataset or its settings break a capstone rule."""


class Category(str, Enum):
    GENERAL_MEDICINE = "General Medicine"
    CARDIOLOGY = "Cardiology"
    DERMATOLOGY = "Dermatology"
    PEDIATRICS = "Pediatrics"
    ORTHOPEDICS = "Orthopedics"


class Status(str, Enum):
    SCHEDULED = "Scheduled"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"
    NO_SHOW = "No-Show"
    RESCHEDULED = "Rescheduled"


class Appointment(BaseModel):
    """One fictional appointment. Pydantic rejects values outside the ranges."""

    record_id: str = Field(pattern=r"^APPT-\d+$")
    category: Category
    status: Status
    consultation_fee_inr: int = Field(
        ge=CONSULTATION_FEE_RANGE[0],
        le=CONSULTATION_FEE_RANGE[1],
    )
    days_since_created: int = Field(
        ge=DAYS_SINCE_CREATED_RANGE[0],
        le=DAYS_SINCE_CREATED_RANGE[1],
    )
    follow_up_required: bool


def validate_configuration(
    *,
    record_count: int = RECORD_COUNT,
    category_weights: list[int] | None = None,
    status_weights: list[int] | None = None,
    fee_range: tuple[int, int] = CONSULTATION_FEE_RANGE,
    days_range: tuple[int, int] = DAYS_SINCE_CREATED_RANGE,
    follow_up_probability: float = FOLLOW_UP_PROBABILITY,
) -> None:
    """Reject generation settings that cannot satisfy the dataset rules."""
    category_weights = CATEGORY_WEIGHTS if category_weights is None else category_weights
    status_weights = STATUS_WEIGHTS if status_weights is None else status_weights
    errors: list[str] = []

    if record_count < MINIMUM_RECORD_COUNT:
        errors.append(
            f"RECORD_COUNT must be at least {MINIMUM_RECORD_COUNT}. "
            f"Received {record_count}."
        )
    if len(category_weights) != len(Category):
        errors.append(
            f"CATEGORY_WEIGHTS needs {len(Category)} positive numbers, "
            f"one per category. Received {list(category_weights)}."
        )
    elif any(weight <= 0 for weight in category_weights):
        errors.append(
            "CATEGORY_WEIGHTS must all be positive. "
            f"Received {list(category_weights)}."
        )
    if len(status_weights) != len(Status):
        errors.append(
            f"STATUS_WEIGHTS needs {len(Status)} positive numbers, "
            f"one per status. Received {list(status_weights)}."
        )
    elif any(weight <= 0 for weight in status_weights):
        errors.append(
            "STATUS_WEIGHTS must all be positive. "
            f"Received {list(status_weights)}."
        )
    if fee_range[0] > fee_range[1] or fee_range != CONSULTATION_FEE_RANGE:
        errors.append(
            "consultation_fee_inr must use the documented fictional range "
            f"{CONSULTATION_FEE_RANGE}. Received {fee_range}."
        )
    if (
        days_range[0] > days_range[1]
        or days_range[0] < DAYS_SINCE_CREATED_RANGE[0]
        or days_range[1] > DAYS_SINCE_CREATED_RANGE[1]
    ):
        errors.append(
            "days_since_created must stay inside "
            f"{DAYS_SINCE_CREATED_RANGE}. Received {days_range}."
        )
    low, high = FOLLOW_UP_PERCENT_RANGE
    if not (low / 100) <= follow_up_probability <= (high / 100):
        errors.append(
            "FOLLOW_UP_PROBABILITY must be between "
            f"{low / 100:.2f} and {high / 100:.2f} so follow-ups can land "
            f"in the {low}%–{high}% band. Received {follow_up_probability}."
        )

    if errors:
        raise DatasetValidationError(
            "Invalid dataset configuration:\n- " + "\n- ".join(errors)
        )


def generate_appointments(
    seed: int = SEED,
    record_count: int = RECORD_COUNT,
    category_weights: list[int] | None = None,
    status_weights: list[int] | None = None,
    fee_range: tuple[int, int] = CONSULTATION_FEE_RANGE,
    days_range: tuple[int, int] = DAYS_SINCE_CREATED_RANGE,
    follow_up_probability: float = FOLLOW_UP_PROBABILITY,
) -> list[Appointment]:
    """Build a deterministic list of fictional appointments.

    The first pass places one record in every category and every status.
    The remaining records are drawn with the configured weights. A shuffle
    then mixes those coverage records through the list.
    """
    category_weights = list(
        CATEGORY_WEIGHTS if category_weights is None else category_weights
    )
    status_weights = list(
        STATUS_WEIGHTS if status_weights is None else status_weights
    )
    validate_configuration(
        record_count=record_count,
        category_weights=category_weights,
        status_weights=status_weights,
        fee_range=fee_range,
        days_range=days_range,
        follow_up_probability=follow_up_probability,
    )

    rng = random.Random(seed)
    category_pool = list(Category)
    status_pool = list(Status)
    categories = category_pool.copy()
    statuses = status_pool.copy()

    while len(categories) < record_count:
        categories.append(
            rng.choices(category_pool, weights=category_weights, k=1)[0]
        )
        statuses.append(
            rng.choices(status_pool, weights=status_weights, k=1)[0]
        )

    rng.shuffle(categories)
    rng.shuffle(statuses)

    appointments: list[Appointment] = []
    for index in range(record_count):
        try:
            appointments.append(
                Appointment(
                    record_id=f"APPT-{1001 + index}",
                    category=categories[index],
                    status=statuses[index],
                    consultation_fee_inr=rng.randint(*fee_range),
                    days_since_created=rng.randint(*days_range),
                    follow_up_required=rng.random() < follow_up_probability,
                )
            )
        except ValidationError as exc:
            raise DatasetValidationError(
                "Generated appointment failed field validation: "
                f"{exc}"
            ) from exc

    return appointments


APPOINTMENTS = generate_appointments()


def validate_dataset(appointments: list[Appointment]) -> dict:
    """Check the capstone rules and return the measured summary."""
    try:
        records = [
            item if isinstance(item, Appointment) else Appointment.model_validate(item)
            for item in appointments
        ]
    except ValidationError as exc:
        raise DatasetValidationError(
            "An appointment record has an invalid field: "
            f"{exc}"
        ) from exc

    errors: list[str] = []
    if len(records) < MINIMUM_RECORD_COUNT:
        errors.append(
            f"The dataset needs at least {MINIMUM_RECORD_COUNT} records. "
            f"Received {len(records)}."
        )

    id_counts = Counter(item.record_id for item in records)
    duplicates = sorted(record_id for record_id, count in id_counts.items() if count > 1)
    if duplicates:
        errors.append(
            "record_id values must be unique. "
            f"Duplicates: {duplicates}."
        )

    category_counts = Counter(item.category.value for item in records)
    for category in Category:
        count = category_counts[category.value]
        if count < MINIMUM_PER_CATEGORY:
            errors.append(
                f"{category.value} has {count} records. "
                f"Every category needs at least {MINIMUM_PER_CATEGORY}."
            )

    status_counts = Counter(item.status.value for item in records)
    for status in Status:
        if status_counts[status.value] < 1:
            errors.append(
                f"{status.value} does not appear. "
                "Every status needs at least 1 record."
            )

    for item in records:
        if not DAYS_SINCE_CREATED_RANGE[0] <= item.days_since_created <= DAYS_SINCE_CREATED_RANGE[1]:
            errors.append(
                f"{item.record_id} has days_since_created="
                f"{item.days_since_created}. Allowed range is "
                f"{DAYS_SINCE_CREATED_RANGE[0]} to {DAYS_SINCE_CREATED_RANGE[1]}."
            )
        if not CONSULTATION_FEE_RANGE[0] <= item.consultation_fee_inr <= CONSULTATION_FEE_RANGE[1]:
            errors.append(
                f"{item.record_id} has consultation_fee_inr="
                f"{item.consultation_fee_inr}. Allowed range is "
                f"{CONSULTATION_FEE_RANGE[0]} to {CONSULTATION_FEE_RANGE[1]} INR."
            )

    follow_up_count = sum(item.follow_up_required for item in records)
    follow_up_percentage = (
        100 * follow_up_count / len(records) if records else 0
    )
    low, high = FOLLOW_UP_PERCENT_RANGE
    if records and not low <= follow_up_percentage <= high:
        errors.append(
            "follow_up_required=True must be "
            f"{low}% to {high}% of records. Received {follow_up_count} of "
            f"{len(records)} ({follow_up_percentage:.2f}%). "
            "Change FOLLOW_UP_PROBABILITY or SEED and regenerate."
        )

    if errors:
        raise DatasetValidationError(
            "Dataset validation failed:\n- " + "\n- ".join(errors)
        )

    return {
        "record_count": len(records),
        "seed": SEED,
        "category_counts": {
            category.value: category_counts[category.value]
            for category in Category
        },
        "status_counts": {
            status.value: status_counts[status.value]
            for status in Status
        },
        "follow_up_count": follow_up_count,
        "follow_up_percentage": round(follow_up_percentage, 2),
    }


def format_dataset_report(report: dict) -> str:
    """Turn a validation summary into the text printed by this module."""
    lines = [
        "Dataset validation passed.",
        f"Records: {report['record_count']}",
        f"Seed: {report['seed']}",
        "Category counts:",
    ]
    lines.extend(
        f"  {name}: {count}"
        for name, count in report["category_counts"].items()
    )
    lines.append("Status counts:")
    lines.extend(
        f"  {name}: {count}"
        for name, count in report["status_counts"].items()
    )
    lines.append(
        "Follow-up required: "
        f"{report['follow_up_count']}/{report['record_count']} "
        f"({report['follow_up_percentage']:.2f}%)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_dataset_report(validate_dataset(APPOINTMENTS)))
