"""Tests for the fictional clinic knowledge base."""

from pathlib import Path

import pytest

from app.rag.knowledge_base import (
    REQUIRED_POLICIES,
    KnowledgeBase,
    KnowledgeBaseValidationError,
    count_sentences,
    format_knowledge_base_report,
)

MEASURED_SENTENCE_COUNTS = {
    "appointment-booking": 4,
    "cancellation-rescheduling": 4,
    "consultation-fees": 4,
    "insurance-claims": 4,
    "prescription-refills": 4,
    "lab-turnaround": 4,
    "telemedicine": 4,
    "emergency-protocol": 4,
    "patient-privacy": 4,
    "follow-up-discount": 4,
    "second-opinion": 4,
    "home-visit": 4,
}


def _policy(
    doc_id: str,
    text: str,
    *,
    topic: str | None = None,
    title: str | None = None,
    source: str | None = None,
) -> str:
    file_name = source or f"{doc_id}.md"
    return (
        "---\n"
        f"doc_id: {doc_id}\n"
        f"topic: {topic or doc_id}\n"
        f"title: {title or doc_id}\n"
        f"source: {file_name}\n"
        "---\n\n"
        f"{text}\n"
    )


def _write_valid_knowledge_base(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for doc_id, phrases in REQUIRED_POLICIES.items():
        sentences = [
            f"This fictional {doc_id} policy explains the clinic rule.",
            "It applies only to this demonstration.",
        ]
        for phrase in phrases:
            sentences.append(f"The policy states {phrase} for this demo.")
        text = " ".join(sentences)
        file_name = f"{doc_id}.md"
        (directory / file_name).write_text(
            _policy(doc_id, text, source=file_name),
            encoding="utf-8",
        )


def test_real_knowledge_base_loads_every_required_topic():
    documents = KnowledgeBase().load()

    assert [document.doc_id for document in documents] == list(REQUIRED_POLICIES)
    assert {document.source for document in documents} == {
        f"{doc_id}.md" for doc_id in REQUIRED_POLICIES
    }


def test_each_real_policy_has_two_to_five_sentences():
    documents = KnowledgeBase().load()
    counts = {
        document.doc_id: count_sentences(document.text)
        for document in documents
    }

    assert counts == MEASURED_SENTENCE_COUNTS
    assert all(2 <= count <= 5 for count in counts.values())


def test_report_lists_sources_and_sentence_counts():
    documents = KnowledgeBase().load()
    text = format_knowledge_base_report(documents)

    assert "Documents: 12" in text
    assert "appointment-booking: 4 sentences, source=appointment-booking.md" in text
    assert "patient-privacy: 4 sentences, source=patient-privacy.md" in text


def test_safety_and_privacy_statements_are_present():
    documents = {document.doc_id: document for document in KnowledgeBase().load()}

    emergency = documents["emergency-protocol"].text.casefold()
    privacy = documents["patient-privacy"].text.casefold()
    assert "emergency services" in emergency
    assert "cannot diagnose" in emergency
    assert "must not disclose" in privacy
    assert "fabricated" in privacy
    assert "not a real medical record" in privacy


def test_missing_directory_raises_a_clear_error(tmp_path):
    missing = tmp_path / "absent"

    with pytest.raises(KnowledgeBaseValidationError, match="does not exist"):
        KnowledgeBase(missing).load()


def test_empty_directory_raises_a_clear_error(tmp_path):
    with pytest.raises(KnowledgeBaseValidationError, match="No markdown"):
        KnowledgeBase(tmp_path).load()


def test_missing_required_topic_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    (tmp_path / "home-visit.md").unlink()

    with pytest.raises(KnowledgeBaseValidationError, match="home-visit"):
        KnowledgeBase(tmp_path).load()


def test_sentence_count_outside_the_range_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    (tmp_path / "home-visit.md").write_text(
        _policy("home-visit", "Only one sentence here."),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="1 sentences"):
        KnowledgeBase(tmp_path).load()


def test_duplicate_doc_id_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    (tmp_path / "extra-booking.md").write_text(
        _policy(
            "appointment-booking",
            "This is a second copy. It should be rejected.",
            source="extra-booking.md",
        ),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="Duplicates"):
        KnowledgeBase(tmp_path).load()


def test_missing_source_metadata_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    text = (tmp_path / "home-visit.md").read_text(encoding="utf-8")
    (tmp_path / "home-visit.md").write_text(
        text.replace("source: home-visit.md\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="missing source"):
        KnowledgeBase(tmp_path).load()


def test_source_must_match_the_file_name(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    text = (tmp_path / "home-visit.md").read_text(encoding="utf-8")
    (tmp_path / "home-visit.md").write_text(
        text.replace("source: home-visit.md", "source: other.md"),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="source id"):
        KnowledgeBase(tmp_path).load()


def test_removed_privacy_statement_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    (tmp_path / "patient-privacy.md").write_text(
        _policy(
            "patient-privacy",
            "Staff may view a record. This sentence omits the required privacy rule.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="must not disclose"):
        KnowledgeBase(tmp_path).load()


def test_file_without_frontmatter_raises_a_clear_error(tmp_path):
    _write_valid_knowledge_base(tmp_path)
    (tmp_path / "notes.md").write_text(
        "This file has two sentences. It still has no metadata block.",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseValidationError, match="frontmatter"):
        KnowledgeBase(tmp_path).load()
