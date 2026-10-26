"""Unit tests for policy chunking. These do not load the embedding model."""

import pytest

from app.rag.chunking import (
    FIXED_SIZE_STRATEGY,
    SENTENCE_STRATEGY,
    chunk_documents,
    fixed_size_chunks,
    sentence_chunks,
)
from app.rag.knowledge_base import PolicyDocument


def test_sentence_chunks_split_on_punctuation():
    chunks = sentence_chunks("One sentence here. Second sentence here.")

    assert chunks == ["One sentence here.", "Second sentence here."]


def test_fixed_size_chunks_overlap_shared_words():
    text = " ".join(f"w{index:02d}" for index in range(30))
    chunks = fixed_size_chunks(text, chunk_size=40, overlap=10)

    assert len(chunks) > 1
    assert chunks[0].split()[-1] in chunks[1].split()


def test_invalid_chunk_settings_raise_clear_errors():
    with pytest.raises(ValueError, match="chunk_size"):
        fixed_size_chunks("hello policy", chunk_size=0)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        fixed_size_chunks("hello policy", chunk_size=10, overlap=10)


def test_chunks_keep_source_metadata():
    document = PolicyDocument(
        doc_id="appointment-booking",
        topic="appointment-booking",
        title="Appointment Booking Policy",
        text="Alpha sentence here. Beta sentence here.",
        source="appointment-booking.md",
    )

    grouped = chunk_documents([document], chunk_size=30, overlap=5)

    for strategy in (FIXED_SIZE_STRATEGY, SENTENCE_STRATEGY):
        assert grouped[strategy]
        assert all(chunk.source == "appointment-booking.md" for chunk in grouped[strategy])
        assert all(chunk.doc_id == "appointment-booking" for chunk in grouped[strategy])
