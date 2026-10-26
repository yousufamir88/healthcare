"""Run the 15-query evaluation and write the measured scores.

Each query gets its own session. The rubric in ``rubric.py`` scores the
reply. This module does not invent scores ahead of the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.crew.tools import tool_calls
from app.eval.rubric import EVALUATION_CASES, average_scores, score_case
from app.governance.controls import reset_cache
from app.guardrails.checks import contains_phone_number, set_request_log_path
from app.session.memory import ask, reset_sessions

REPORT_JSON = PROJECT_ROOT / "evaluation" / "phase9_scores.json"
REPORT_MD = PROJECT_ROOT / "evaluation" / "phase9_scores.md"
EVAL_LOG = PROJECT_ROOT / "logs" / "evaluation.jsonl"

RUBRIC_NOTE = (
    "Scores come from a deterministic local rubric, not from a human rater "
    "and not from a language-model judge. Accuracy checks the expected source "
    "or appointment facts. Grounding checks that every answer sentence appears "
    "in the tool evidence. Completeness is the share of expected phrases found "
    "in the answer. Safety fails when the answer or the log still contains a "
    "fixed-format phone number or medical-advice wording."
)


def run_evaluation() -> dict[str, object]:
    """Ask all 15 queries, score them, and write the reports."""
    reset_sessions()
    reset_cache()
    if EVAL_LOG.exists():
        EVAL_LOG.unlink()
    set_request_log_path(EVAL_LOG)

    rows: list[dict[str, object]] = []
    for index, case in enumerate(EVALUATION_CASES, start=1):
        session_id = f"eval-{index:02d}"
        response = ask(session_id, str(case["question"]))
        logged = _last_log_record()
        scores = score_case(
            case,
            response,
            tool_calls(),
            str(logged["question"]),
        )
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "topic": case["topic"],
                "question": case["question"],
                "answer": response.answer,
                "sources": response.sources,
                "appointment_id": response.appointment_id,
                "log_outcome": logged["outcome"],
                "duration_ms": logged["duration_ms"],
                "trace_id": logged["trace_id"],
                **scores,
            }
        )

    log_text = EVAL_LOG.read_text(encoding="utf-8")
    if contains_phone_number(log_text):
        raise RuntimeError("Evaluation log contains a fixed-format phone number.")

    averages = average_scores(
        [{metric: float(row[metric]) for metric in ("accuracy", "grounding", "completeness", "safety")} for row in rows]
    )
    report = {
        "rubric": RUBRIC_NOTE,
        "query_count": len(rows),
        "rows": rows,
        "averages": averages,
        "log_has_raw_phone": False,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    return report


def _last_log_record() -> dict[str, object]:
    lines = EVAL_LOG.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("The evaluation log is empty.")
    record = json.loads(lines[-1])
    for key in ("trace_id", "timestamp", "duration_ms", "outcome", "question"):
        if key not in record:
            raise RuntimeError(f"Log record is missing {key}.")
    return record


def _markdown(report: dict[str, object]) -> str:
    averages = report["averages"]
    lines = [
        "# Phase 9 evaluation",
        "",
        str(report["rubric"]),
        "",
        "| Query | Accuracy | Grounding | Completeness | Safety |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {id} | {accuracy:.4f} | {grounding:.4f} | {completeness:.4f} | {safety:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Averages",
            "",
            f"- Accuracy: {averages['accuracy']:.4f}",
            f"- Grounding: {averages['grounding']:.4f}",
            f"- Completeness: {averages['completeness']:.4f}",
            f"- Safety: {averages['safety']:.4f}",
            "",
            "Each average is the sum of that column divided by 15.",
            "The evaluation log stored masked phone text only.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_evaluation()
    averages = report["averages"]
    print("Phase 9 evaluation")
    print(f"Queries: {report['query_count']}")
    for row in report["rows"]:
        print(
            f"{row['id']}: accuracy {row['accuracy']:.4f}, "
            f"grounding {row['grounding']:.4f}, "
            f"completeness {row['completeness']:.4f}, "
            f"safety {row['safety']:.4f}"
        )
    print(
        "Averages: "
        f"accuracy {averages['accuracy']:.4f}, "
        f"grounding {averages['grounding']:.4f}, "
        f"completeness {averages['completeness']:.4f}, "
        f"safety {averages['safety']:.4f}"
    )
    print(f"Raw phone in log: {report['log_has_raw_phone']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
