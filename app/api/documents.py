"""Add one fictional policy and index it before it can be searched."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.crew.tools import reset_retriever
from app.rag.chunking import DEFAULT_CHROMA_PATH, index_knowledge_base
from app.rag.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE,
    MAX_SENTENCES,
    MIN_SENTENCES,
    KnowledgeBase,
    KnowledgeBaseValidationError,
    count_sentences,
)


class DocumentRejected(ValueError):
    """Raised when a new policy fails validation before it is indexed."""


class AddDocumentRequest(BaseModel):
    """A new fictional policy. The topic must match the document id."""

    doc_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    topic: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str | None = None

    @model_validator(mode="after")
    def source_matches_the_file_name(self) -> "AddDocumentRequest":
        expected = f"{self.doc_id}.md"
        if self.topic != self.doc_id:
            raise ValueError("topic must match doc_id.")
        if self.source is None:
            self.source = expected
        elif self.source != expected:
            raise ValueError(
                f"source must be {expected!r}. Received {self.source!r}."
            )
        return self


def add_policy_document(
    request: AddDocumentRequest,
    directory: Path | None = None,
    persist_directory: Path | None = None,
) -> dict[str, object]:
    """Write the policy, validate the folder, then rebuild the search index."""
    folder = Path(directory or DEFAULT_KNOWLEDGE_BASE)
    chroma_directory = Path(persist_directory or DEFAULT_CHROMA_PATH)
    sentence_count = count_sentences(request.text)
    if not MIN_SENTENCES <= sentence_count <= MAX_SENTENCES:
        raise DocumentRejected(
            f"The policy has {sentence_count} sentences. "
            f"Each policy needs {MIN_SENTENCES} to {MAX_SENTENCES}."
        )

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / str(request.source)
    if path.exists():
        raise DocumentRejected(f"{path.name} already exists.")

    path.write_text(_markdown(request), encoding="utf-8")
    try:
        KnowledgeBase(folder).load()
    except KnowledgeBaseValidationError as exc:
        path.unlink(missing_ok=True)
        raise DocumentRejected(str(exc)) from exc

    reset_retriever()
    counts = index_knowledge_base(
        directory=folder,
        persist_directory=chroma_directory,
    )
    return {
        "doc_id": request.doc_id,
        "source": request.source,
        "sentences": sentence_count,
        "indexed": True,
        "collection_counts": counts,
    }


def _markdown(request: AddDocumentRequest) -> str:
    return (
        "---\n"
        f"doc_id: {request.doc_id}\n"
        f"topic: {request.topic}\n"
        f"title: {request.title}\n"
        f"source: {request.source}\n"
        "---\n\n"
        f"{request.text.strip()}\n"
    )
