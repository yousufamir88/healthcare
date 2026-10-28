"""Three-agent Practo support crew.

The retrieval agent can search policies. The lookup agent is the only agent
that can read an appointment. The composer turns those tool results into the
final answer. Every agent uses MockLLM, so this module does not call a paid
model.
"""

from __future__ import annotations

import os

from app.config import get_settings

get_settings()
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, Process, Task

from app.crew.tools import (
    lookup_appointment,
    reset_tool_log,
    search_policies,
    tool_calls,
)
from app.llm.mock_llm import QUESTION_MARKER, MockLLM, reset_llm_log
from app.session.schema import SupportResponse, response_format, safe_support_response

RETRIEVAL_ROLE = "Retrieval Agent"
LOOKUP_ROLE = "Lookup Agent"
COMPOSER_ROLE = "Response Composer"


def build_support_crew() -> Crew:
    """Create a new sequential crew. Each kickoff needs its own crew."""
    llm = MockLLM()
    retrieval_agent = Agent(
        role=RETRIEVAL_ROLE,
        goal="Find clinic policy text that answers the support question.",
        backstory=(
            "You search the fictional clinic knowledge base. You answer only "
            "with text the search tool returns."
        ),
        tools=[search_policies],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        cache=False,
        max_iter=4,
    )
    lookup_agent = Agent(
        role=LOOKUP_ROLE,
        goal="Look up a synthetic appointment when the question contains a record id.",
        backstory=(
            "You are the only agent allowed to read appointment records. "
            "You use the appointment tool only when a record id is present."
        ),
        tools=[lookup_appointment],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        cache=False,
        max_iter=4,
    )
    composer = Agent(
        role=COMPOSER_ROLE,
        goal="Combine the policy evidence and the appointment evidence.",
        backstory=(
            "You write the final support answer from the evidence the other "
            "agents already collected. You do not add facts of your own."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        cache=False,
        max_iter=4,
    )

    retrieval_task = Task(
        description=(
            "Search the clinic policies for this question.\n"
            f"{QUESTION_MARKER} {{question}}"
        ),
        expected_output="The policy search result as JSON.",
        agent=retrieval_agent,
    )
    lookup_task = Task(
        description=(
            "Look up the appointment only when the question contains a record id.\n"
            f"{QUESTION_MARKER} {{question}}"
        ),
        expected_output="The appointment lookup result as JSON.",
        agent=lookup_agent,
    )
    compose_task = Task(
        description=(
            "Write the final answer from the policy result and the appointment "
            "result. Do not add facts that are not in those results.\n"
            f"{QUESTION_MARKER} {{question}}"
        ),
        expected_output=(
            "A SupportResponse JSON object with answer, intent, sources, "
            "appointment_id, escalation_score, and requires_human_review."
        ),
        agent=composer,
        context=[retrieval_task, lookup_task],
        output_pydantic=response_format,
    )
    return Crew(
        agents=[retrieval_agent, lookup_agent, composer],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        planning=False,
        cache=False,
        tracing=False,
    )


def run_support_crew(question: str) -> str:
    """Run one question through the crew and return the final answer text.

    The Response Composer task uses ``response_format`` / ``output_pydantic``
    so the crew output is validated as ``SupportResponse`` before the answer
    text is returned.
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string.")
    reset_llm_log()
    reset_tool_log()
    result = build_support_crew().kickoff(inputs={"question": cleaned})
    validated = validate_crew_response(result)
    return validated.answer


def validate_crew_response(result: object) -> SupportResponse:
    """Validate a crew kickoff result against ``response_format``."""
    pydantic_output = getattr(result, "pydantic", None)
    if isinstance(pydantic_output, SupportResponse):
        return response_format.model_validate(pydantic_output.model_dump())
    if isinstance(pydantic_output, dict):
        return response_format.model_validate(pydantic_output)
    raw = getattr(result, "raw", None)
    text = raw if isinstance(raw, str) and raw.strip() else str(result)
    try:
        return response_format.model_validate_json(text)
    except Exception:
        return safe_support_response({"answer": text, "intent": "unknown"})


def main() -> int:
    """Run one policy question and one appointment question."""
    questions = (
        "How do I book a clinic appointment?",
        "What is the status of appointment APPT-1001?",
    )
    lines = [
        "Practo support crew",
        "Mode: MOCK_LLM",
        "Agents: Retrieval Agent, Lookup Agent, Response Composer",
    ]
    for question in questions:
        answer = run_support_crew(question)
        lines.append("")
        lines.append(f"Question: {question}")
        lines.append("Tool calls:")
        if not tool_calls():
            lines.append("  none")
        for call in tool_calls():
            lines.append(f"  {call['tool']} {call['arguments']}")
        lines.append(f"Final answer: {answer}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
