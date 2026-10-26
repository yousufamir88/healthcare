"""Crew tools. Only the lookup agent is given the appointment tool."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import tool

from app.rag.chunking import SENTENCE_STRATEGY
from app.rag.retrieval import EVALUATION_DIR, PolicyRetriever
from app.tools.appointment_tools import check_appointment_status

_CALLS: list[dict[str, Any]] = []
_RETRIEVER: PolicyRetriever | None = None


def reset_tool_log() -> None:
    """Clear the record of tool executions."""
    _CALLS.clear()


def tool_calls() -> list[dict[str, Any]]:
    """Return the tool executions since the last reset."""
    return list(_CALLS)


def load_measured_threshold() -> float:
    """Read the threshold written by the Phase 3 measurement."""
    path = EVALUATION_DIR / "retrieval_threshold_report.json"
    if not path.is_file():
        raise RuntimeError(
            "The measured retrieval threshold file is missing. "
            "Run python -m app.rag.retrieval before starting the crew."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(report["threshold"]["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "The retrieval threshold report does not contain "
            "threshold.threshold as a number."
        ) from exc


def reset_retriever() -> None:
    """Drop the cached search client so the next search reads a fresh index."""
    global _RETRIEVER
    _RETRIEVER = None


def _retriever() -> PolicyRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        retriever = PolicyRetriever()
        names = {collection.name for collection in retriever.client().list_collections()}
        if SENTENCE_STRATEGY not in names:
            retriever.index()
        _RETRIEVER = retriever
    return _RETRIEVER


@tool("search_policies")
def search_policies(query: str) -> str:
    """Search the fictional clinic policies and return retrieved text only."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string.")

    threshold = load_measured_threshold()
    grounded = _retriever().answer(cleaned, SENTENCE_STRATEGY, threshold)
    payload = {
        "tool": "search_policies",
        "query": cleaned,
        "fallback": grounded.fallback,
        "answer": grounded.answer,
        "sources": [chunk.source for chunk in grounded.chunks if chunk.similarity >= threshold],
        "threshold": grounded.threshold,
    }
    rendered = json.dumps(payload)
    _CALLS.append({"tool": "search_policies", "arguments": {"query": cleaned}, "result": payload})
    return rendered


@tool("check_appointment_status")
def lookup_appointment(record_id: str) -> str:
    """Look up one synthetic appointment and return its status, fee, and score."""
    result = check_appointment_status(record_id)
    payload = {"tool": "check_appointment_status", **result}
    rendered = json.dumps(payload)
    _CALLS.append(
        {
            "tool": "check_appointment_status",
            "arguments": {"record_id": record_id.strip()},
            "result": payload,
        }
    )
    return rendered
