"""Retrieval tests.

The marked integration test loads the local embedding model and builds a
temporary Chroma index. The other tests check the scoring rules only.
"""

from pathlib import Path

import pytest

from app.rag.chunking import FIXED_SIZE_STRATEGY, SENTENCE_STRATEGY
from app.rag.retrieval import (
    FALLBACK_ANSWER,
    RetrievedChunk,
    render_grounded_answer,
    run_evaluation,
    score_documents,
    select_threshold,
)


def _hit(text: str, similarity: float, doc_id: str = "appointment-booking") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:sentence:0",
        doc_id=doc_id,
        title="Title",
        source=f"{doc_id}.md",
        strategy=SENTENCE_STRATEGY,
        text=text,
        similarity=similarity,
        distance=1 - similarity,
    )


def test_document_scores_count_each_source_once():
    scored = score_documents(
        ["appointment-booking", "appointment-booking", "insurance-claims"],
        ["appointment-booking"],
    )

    assert scored["retrieved_doc_ids"] == ["appointment-booking", "insurance-claims"]
    assert scored["true_positives"] == 1
    assert scored["precision_arithmetic"] == "1 / 2 = 0.5"
    assert scored["recall_arithmetic"] == "1 / 1 = 1.0"


def test_threshold_midpoint_comes_from_the_measured_gap():
    decision = select_threshold([0.80, 0.70, 0.60], [0.20, 0.10])

    assert decision["rule"] == "midpoint"
    assert decision["threshold"] == pytest.approx((0.60 + 0.20) / 2)


def test_threshold_requires_enough_calibration_queries():
    with pytest.raises(ValueError, match="at least 3"):
        select_threshold([0.8, 0.7], [0.1, 0.2])


def test_overlapping_scores_use_measured_separation_not_a_preset():
    decision = select_threshold([0.90, 0.40, 0.30], [0.50, 0.10])

    assert decision["rule"] == "best-separation"
    assert decision["threshold"] in {0.90, 0.50, 0.40, 0.30, 0.10}
    assert decision["threshold"] not in {0.5, 0.6, 0.7}


def test_answer_quotes_only_chunks_at_or_above_the_threshold():
    answer, fallback = render_grounded_answer(
        [
            _hit("Supported policy sentence.", 0.80),
            _hit("This weaker chunk stays out.", 0.20, doc_id="insurance-claims"),
        ],
        threshold=0.55,
    )

    assert fallback is False
    assert answer == "Supported policy sentence."


def test_low_similarity_triggers_the_fallback():
    answer, fallback = render_grounded_answer(
        [_hit("Unrelated policy sentence.", 0.10)],
        threshold=0.55,
    )

    assert fallback is True
    assert answer == FALLBACK_ANSWER


@pytest.mark.integration
def test_indexed_retrieval_answers_from_the_policy_index(tmp_path: Path):
    report = run_evaluation(
        persist_directory=tmp_path / "chroma",
        output_directory=tmp_path / "evaluation",
    )

    counts = report["collection_counts"]
    assert counts[FIXED_SIZE_STRATEGY] > 0
    assert counts[SENTENCE_STRATEGY] > 0

    single_topic_rows = [
        row
        for row in report["comparison"]
        if row["strategy"] == report["recommended_strategy"]
        and len(row["expected_doc_ids"]) == 1
    ]
    relevant = [
        row
        for row in single_topic_rows
        if set(row["expected_doc_ids"]).issubset(set(row["retrieved_doc_ids"]))
    ]
    assert len(relevant) >= 5

    fallbacks = [
        item for item in report["demonstrations"] if item["fallback"] is True
    ]
    supported = [
        item for item in report["demonstrations"] if item["fallback"] is False
    ]
    assert supported
    assert fallbacks
    assert fallbacks[0]["answer"] == FALLBACK_ANSWER
    for item in supported:
        for chunk in item["chunks"]:
            if chunk["similarity"] >= item["threshold"]:
                assert chunk["text"] in item["answer"]

    probe = report["cosine_distance_probe"]
    assert probe["identical"] == pytest.approx(0.0, abs=1e-6)
    assert probe["orthogonal"] == pytest.approx(1.0, abs=1e-6)
