"""Load and validate the fictional clinic policy documents.

Each markdown file is one policy. The frontmatter identifies the document,
and the body is the policy text. Loading the folder checks that every
required topic is present, that each policy has two to five sentences, and
that safety and privacy statements stay in the documents that need them.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

MIN_SENTENCES = 2
MAX_SENTENCES = 5

# A sentence ends with . ! or ? and is followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# doc_id values required by the capstone, in specification order.
# Phrases are checked case-insensitively so safety and privacy text cannot
# be removed without a validation error.
REQUIRED_POLICIES: dict[str, tuple[str, ...]] = {
    "appointment-booking": (),
    "cancellation-rescheduling": (),
    "consultation-fees": (),
    "insurance-claims": (),
    "prescription-refills": ("never refilled",),
    "lab-turnaround": ("interpreting",),
    "telemedicine": ("emergency services",),
    "emergency-protocol": ("emergency services", "cannot diagnose"),
    "patient-privacy": ("must not disclose", "fabricated"),
    "follow-up-discount": (),
    "second-opinion": ("not a substitute for emergency",),
    "home-visit": (),
}

REQUIRED_METADATA = ("doc_id", "topic", "title", "source")


class KnowledgeBaseValidationError(ValueError):
    """Raised when policy files or their metadata break a knowledge-base rule."""


class PolicyDocument(BaseModel):
    """One loaded policy, including the file name used as its source id."""

    doc_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)


class KnowledgeBase:
    """Read markdown policies from a directory and validate the set."""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or DEFAULT_KNOWLEDGE_BASE)

    def load(self) -> list[PolicyDocument]:
        """Return every valid policy, or raise one error listing every problem."""
        if not self.directory.is_dir():
            raise KnowledgeBaseValidationError(
                "Knowledge base directory does not exist: "
                f"{self.directory}"
            )

        paths = sorted(self.directory.glob("*.md"))
        if not paths:
            raise KnowledgeBaseValidationError(
                "No markdown policy documents found in "
                f"{self.directory}."
            )

        documents: list[PolicyDocument] = []
        errors: list[str] = []
        for path in paths:
            document, file_errors = self._load_file(path)
            errors.extend(file_errors)
            if document is not None:
                documents.append(document)

        errors.extend(_validate_collection(documents))
        if errors:
            raise KnowledgeBaseValidationError(
                "Knowledge base validation failed:\n- " + "\n- ".join(errors)
            )

        order = {doc_id: index for index, doc_id in enumerate(REQUIRED_POLICIES)}
        documents.sort(key=lambda document: order.get(document.doc_id, len(order)))
        return documents

    def _load_file(self, path: Path) -> tuple[PolicyDocument | None, list[str]]:
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---"):
            return None, [
                f"{path.name} is missing frontmatter. "
                "Start the file with a --- metadata block."
            ]

        metadata, body = _split_frontmatter(raw)
        errors: list[str] = []
        for key in REQUIRED_METADATA:
            if not metadata.get(key):
                errors.append(f"{path.name} is missing {key} metadata.")

        source = metadata.get("source", "")
        if source and source != path.name:
            errors.append(
                f"{path.name} declares source {source!r}. "
                f"The source id must be the file name {path.name!r}."
            )

        text = body.strip()
        if not text:
            errors.append(f"{path.name} has no policy text.")

        if errors:
            return None, errors

        try:
            document = PolicyDocument(
                doc_id=metadata["doc_id"],
                topic=metadata["topic"],
                title=metadata["title"],
                text=text,
                source=source,
            )
        except ValueError as exc:
            return None, [f"{path.name} has invalid metadata: {exc}"]
        return document, []


def count_sentences(text: str) -> int:
    """Count sentences by splitting on . ! or ? followed by whitespace."""
    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return len(
        [
            sentence
            for sentence in _SENTENCE_BOUNDARY.split(normalized)
            if sentence.strip()
        ]
    )


def _validate_collection(documents: list[PolicyDocument]) -> list[str]:
    errors: list[str] = []
    id_counts: dict[str, int] = {}
    for document in documents:
        id_counts[document.doc_id] = id_counts.get(document.doc_id, 0) + 1
        if document.topic != document.doc_id:
            errors.append(
                f"{document.source} topic is {document.topic!r}. "
                f"It must match doc_id {document.doc_id!r}."
            )
        sentence_count = count_sentences(document.text)
        if not MIN_SENTENCES <= sentence_count <= MAX_SENTENCES:
            errors.append(
                f"{document.source} has {sentence_count} sentences. "
                f"Each policy needs {MIN_SENTENCES} to {MAX_SENTENCES}."
            )
        for phrase in REQUIRED_POLICIES.get(document.doc_id, ()):
            if phrase.casefold() not in document.text.casefold():
                errors.append(
                    f"{document.source} is missing the required safety or "
                    f"privacy statement {phrase!r}."
                )

    duplicates = sorted(doc_id for doc_id, count in id_counts.items() if count > 1)
    if duplicates:
        errors.append(
            "doc_id values must be unique. "
            f"Duplicates: {duplicates}."
        )

    missing = [
        doc_id for doc_id in REQUIRED_POLICIES if doc_id not in id_counts
    ]
    if missing:
        errors.append(
            "The knowledge base is missing required topics: "
            + ", ".join(missing)
            + "."
        )
    return errors


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw

    _, frontmatter, body = raw.split("---", 2)
    metadata: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body


def format_knowledge_base_report(documents: list[PolicyDocument]) -> str:
    """Turn a loaded knowledge base into the text printed by this module."""
    lines = [
        "Knowledge base validation passed.",
        f"Documents: {len(documents)}",
    ]
    for document in documents:
        lines.append(
            f"  {document.doc_id}: "
            f"{count_sentences(document.text)} sentences, "
            f"source={document.source}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    loaded = KnowledgeBase().load()
    print(format_knowledge_base_report(loaded))
