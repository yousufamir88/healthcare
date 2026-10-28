"""Tests for the three-agent support crew."""

import pytest

from app.crew.support_crew import (
    COMPOSER_ROLE,
    LOOKUP_ROLE,
    RETRIEVAL_ROLE,
    build_support_crew,
    run_support_crew,
)
from app.crew.tools import tool_calls
from app.session.schema import SupportResponse, response_format


def test_only_the_lookup_agent_receives_the_appointment_tool():
    crew = build_support_crew()
    tools_by_role = {
        agent.role: [tool.name for tool in (agent.tools or [])]
        for agent in crew.agents
    }

    assert list(tools_by_role) == [RETRIEVAL_ROLE, LOOKUP_ROLE, COMPOSER_ROLE]
    assert tools_by_role[RETRIEVAL_ROLE] == ["search_policies"]
    assert tools_by_role[LOOKUP_ROLE] == ["check_appointment_status"]
    assert tools_by_role[COMPOSER_ROLE] == []


def test_composer_task_declares_response_format():
    crew = build_support_crew()
    compose_task = crew.tasks[-1]

    assert response_format is SupportResponse
    assert compose_task.output_pydantic is response_format


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="question"):
        run_support_crew("   ")


@pytest.mark.integration
def test_policy_query_invokes_retrieval_and_appointment_query_invokes_lookup():
    policy_answer = run_support_crew("How do I book a clinic appointment?")
    policy_tools = [call["tool"] for call in tool_calls()]

    assert policy_tools == ["search_policies"]
    assert "four hours" in policy_answer
    assert "APPT-1001" not in policy_answer

    appointment_answer = run_support_crew(
        "What is the status of appointment APPT-1001?"
    )
    appointment_tools = [call["tool"] for call in tool_calls()]

    assert "search_policies" in appointment_tools
    assert "check_appointment_status" in appointment_tools
    lookup = next(
        call for call in tool_calls() if call["tool"] == "check_appointment_status"
    )
    assert lookup["arguments"] == {"record_id": "APPT-1001"}
    assert lookup["result"]["status"] == "Completed"
    assert lookup["result"]["consultation_fee_inr"] == 1883
    assert "Completed" in appointment_answer
    assert "1883" in appointment_answer
