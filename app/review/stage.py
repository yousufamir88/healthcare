"""Review a support draft with a two-agent AutoGen team.

The team is a ``RoundRobinGroupChat`` limited to two turns. The first
speaker is Policy-Compliance-Reviewer. The second speaker is Final-Editor.
Both use a local mock model, so this stage does not call a paid API.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from pydantic import BaseModel, Field

from app.guardrails.checks import (
    _sentences,
    contains_phone_number,
    evidence_text,
    medical_advice_detected,
)
from app.rag.retrieval import FALLBACK_ANSWER

# AutoGen 0.7 requires each agent name to be a Python identifier, so the
# hyphenated role titles cannot be used as the name argument.
REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"
REVIEWER_ROLE = "Policy-Compliance-Reviewer"
EDITOR_ROLE = "Final-Editor"
MAX_TURNS = 2
DRAFT_MARKER = "SUPPORT_DRAFT:"
EVIDENCE_MARKER = "SUPPORT_EVIDENCE:"


class ReviewVerdict(BaseModel):
    """Decision returned by the Final-Editor.

    ``approved`` is true only when the draft stays inside the evidence.
    ``final_answer`` is the text the caller returns. On approval it is the
    original draft. On revision it is the verified fallback.
    """

    approved: bool
    final_answer: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MockReviewClient(ChatCompletionClient):
    """Local model client. It reads the draft in the prompt and does not use an API key."""

    def __init__(self) -> None:
        self._last_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ) -> CreateResult:
        reply = _reply_for(messages)
        usage = RequestUsage(
            prompt_tokens=max(1, len(_conversation(messages)) // 4),
            completion_tokens=max(1, len(reply) // 4),
        )
        self._last_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
        )
        return CreateResult(
            finish_reason="stop",
            content=reply,
            usage=usage,
            cached=False,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        async def _stream() -> AsyncGenerator[str | CreateResult, None]:
            result = await self.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            if isinstance(result.content, str):
                yield result.content
            yield result

        return _stream()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
    ) -> int:
        return max(1, len(_conversation(messages)) // 4)

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
    ) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))

    @property
    def capabilities(self) -> ModelCapabilities:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
        }

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }


def draft_is_compliant(draft: str, evidence: str) -> bool:
    """Return True when every draft sentence is in the evidence and no dose is present."""
    if contains_phone_number(draft) or medical_advice_detected(draft):
        return False
    supported = " ".join(evidence.split())
    normalized_draft = " ".join(draft.split())
    if not supported:
        return normalized_draft == " ".join(FALLBACK_ANSWER.split())
    for sentence in _sentences(draft):
        if " ".join(sentence.split()) not in supported:
            return False
    return True


def verdict_for(draft: str, evidence: str) -> ReviewVerdict:
    """Approve the draft unchanged, or replace it with the verified fallback."""
    if draft_is_compliant(draft, evidence):
        return ReviewVerdict(
            approved=True,
            final_answer=draft,
            reason=(
                "The draft matches the policy evidence and contains no "
                "medical advice."
            ),
        )
    return ReviewVerdict(
        approved=False,
        final_answer=FALLBACK_ANSWER,
        reason=(
            "The draft is outside the policy evidence, so the answer was "
            "replaced with the verified fallback."
        ),
    )


def review_draft(draft: str, tool_log: list[dict] | None = None) -> ReviewVerdict:
    """Run the two-turn review and return the editor's verdict."""
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError("draft must be a non-empty string.")
    evidence = evidence_text(tool_log or [], draft)
    verdict, _speakers = asyncio.run(_run_team(draft, evidence))
    return verdict


def review_transcript(
    draft: str,
    tool_log: list[dict] | None = None,
) -> tuple[ReviewVerdict, list[str]]:
    """Return the verdict and the speaker order from one review."""
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError("draft must be a non-empty string.")
    evidence = evidence_text(tool_log or [], draft)
    return asyncio.run(_run_team(draft, evidence))


async def _run_team(draft: str, evidence: str) -> tuple[ReviewVerdict, list[str]]:
    client = MockReviewClient()
    reviewer = AssistantAgent(
        REVIEWER_NAME,
        model_client=client,
        description=f"{REVIEWER_ROLE}. Checks a draft against the policy evidence.",
        system_message=(
            f"You are {REVIEWER_ROLE}. Your agent name is {REVIEWER_NAME}. "
            "Read the draft and the evidence. Say whether the draft is "
            "approved. Do not rewrite the answer."
        ),
    )
    editor = AssistantAgent(
        EDITOR_NAME,
        model_client=client,
        description=f"{EDITOR_ROLE}. Publishes the reviewed answer.",
        system_message=(
            f"You are {EDITOR_ROLE}. Your agent name is {EDITOR_NAME}. "
            "If the draft is approved, copy it unchanged into final_answer. "
            "If it is not approved, set final_answer to the verified fallback. "
            "Reply with the verdict JSON."
        ),
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=MAX_TURNS,
    )
    result = await team.run(
        task=(
            f"{DRAFT_MARKER}\n{draft}\n"
            f"{EVIDENCE_MARKER}\n{evidence}"
        )
    )
    speakers = [
        message.source
        for message in result.messages
        if getattr(message, "source", None) in {REVIEWER_NAME, EDITOR_NAME}
    ]
    if speakers != [REVIEWER_NAME, EDITOR_NAME]:
        raise RuntimeError(
            "The review team did not stop after the reviewer and the editor. "
            f"Speakers: {speakers}."
        )
    editor_message = result.messages[-1]
    content = getattr(editor_message, "content", "")
    if not isinstance(content, str):
        raise RuntimeError("The Final-Editor did not return text.")
    verdict = ReviewVerdict.model_validate_json(content)
    return verdict, speakers


def _reply_for(messages: Sequence[LLMMessage]) -> str:
    text = _conversation(messages)
    draft = _section(text, DRAFT_MARKER, EVIDENCE_MARKER)
    evidence = _section(text, EVIDENCE_MARKER, None)
    verdict = verdict_for(draft, evidence)
    if EDITOR_NAME in text:
        return verdict.model_dump_json()
    return f"approved={str(verdict.approved).lower()}\nreason={verdict.reason}"


def _conversation(messages: Sequence[LLMMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _section(text: str, start: str, end: str | None) -> str:
    begin = text.find(start)
    if begin < 0:
        return ""
    begin += len(start)
    if end is None:
        return text[begin:].strip()
    finish = text.find(end, begin)
    if finish < 0:
        return text[begin:].strip()
    return text[begin:finish].strip()


def main() -> int:
    """Show one unchanged approval and one revised flawed draft."""
    evidence = [
        {
            "tool": "search_policies",
            "result": {
                "answer": (
                    "A booked appointment may be cancelled or rescheduled "
                    "until six hours before the scheduled start time."
                ),
                "sources": ["cancellation-rescheduling.md"],
                "fallback": False,
            },
        }
    ]
    grounded = (
        "A booked appointment may be cancelled or rescheduled until "
        "six hours before the scheduled start time. "
        "Sources: cancellation-rescheduling.md."
    )
    flawed = "You should take 500mg of ibuprofen for the pain."
    grounded_verdict, grounded_speakers = review_transcript(grounded, evidence)
    flawed_verdict, flawed_speakers = review_transcript(flawed, evidence)
    lines = [
        "AutoGen review demonstration",
        "Mode: MOCK_LLM",
        "Team: RoundRobinGroupChat",
        f"max_turns: {MAX_TURNS}",
        f"Roles: {REVIEWER_ROLE}, {EDITOR_ROLE}",
        f"Agent names: {REVIEWER_NAME}, {EDITOR_NAME}",
        "",
        "Case: grounded draft",
        f"Speakers: {', '.join(grounded_speakers)}",
        f"Approved: {grounded_verdict.approved}",
        f"Answer unchanged: {grounded_verdict.final_answer == grounded}",
        f"Reason: {grounded_verdict.reason}",
        f"Final answer: {grounded_verdict.final_answer}",
        "",
        "Case: flawed draft",
        f"Speakers: {', '.join(flawed_speakers)}",
        f"Approved: {flawed_verdict.approved}",
        f"Answer unchanged: {flawed_verdict.final_answer == flawed}",
        f"Reason: {flawed_verdict.reason}",
        f"Final answer: {flawed_verdict.final_answer}",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
