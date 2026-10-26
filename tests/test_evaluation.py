"""Tests for the local evaluation rubric and the measured report."""

import json

import pytest

from app.eval.rubric import (
    EVALUATION_CASES,
    average_scores,
    score_case,
)
from app.guardrails.checks import FALLBACK_ANSWER, write_request_log
from app.rag.knowledge_base import REQUIRED_POLICIES
from app.session.schema import SupportIntent, SupportResponse


def test_evaluation_set_covers_every_policy_and_two_edges():
    topics = {str(case["topic"]) for case in EVALUATION_CASES}
    kinds = [str(case["kind"]) for case in EVALUATION_CASES]

    assert len(EVALUATION_CASES) == 15
    assert set(REQUIRED_POLICIES).issubset(topics)
    assert kinds.count("out_of_scope") + kinds.count("phone") >= 2


def test_average_is_the_sum_divided_by_the_row_count():
    rows = [
        {"accuracy": 1.0, "grounding": 0.0, "completeness": 0.5, "safety": 1.0},
        {"accuracy": 0.0, "grounding": 1.0, "completeness": 0.5, "safety": 1.0},
    ]

    averages = average_scores(rows)

    assert averages["accuracy"] == pytest.approx(0.5)
    assert averages["grounding"] == pytest.approx(0.5)
    assert averages["completeness"] == pytest.approx(0.5)
    assert averages["safety"] == pytest.approx(1.0)


def test_out_of_scope_fallback_scores_accurately_and_a_dose_fails_safety():
    case = next(item for item in EVALUATION_CASES if item["kind"] == "out_of_scope")
    response = SupportResponse(
        answer=FALLBACK_ANSWER,
        intent=SupportIntent.POLICY,
        requires_human_review=True,
    )
    tool_log = [
        {
            "tool": "search_policies",
            "result": {"answer": FALLBACK_ANSWER, "sources": [], "fallback": True},
        }
    ]

    scores = score_case(case, response, tool_log, "Who will win the next cricket world cup?")

    assert scores["accuracy"] == 1.0
    assert scores["grounding"] == 1.0
    assert scores["safety"] == 1.0

    unsafe = score_case(
        case,
        response.model_copy(update={"answer": "You should take 500mg of ibuprofen."}),
        tool_log,
        case["question"],
    )
    assert unsafe["safety"] == 0.0


def test_request_log_records_duration_and_masks_the_phone(tmp_path):
    path = tmp_path / "requests.jsonl"
    record = write_request_log(
        session_id="eval-demo",
        question="Call +91 9876543210",
        outcome="answered",
        duration_ms=12.3456,
        path=path,
    )
    logged = path.read_text(encoding="utf-8")

    assert record["duration_ms"] == 12.346
    assert "trace_id" in record
    assert "timestamp" in record
    assert "9876543210" not in logged


def test_saved_report_has_fifteen_rows_and_matching_averages():
    path = (
        __import__("app.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "evaluation"
        / "phase9_scores.json"
    )
    if not path.is_file():
        pytest.skip("Run python -m app.eval.harness before checking the saved report.")

    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["rows"]
    averages = average_scores(rows)

    assert len(rows) == 15
    assert all(metric in row for row in rows for metric in averages)
    for metric, value in averages.items():
        assert report["averages"][metric] == pytest.approx(value)
