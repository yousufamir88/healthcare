"""Tests for schema-based tool dispatch in the local mock model."""

from app.llm.mock_llm import MockLLM, reset_llm_log
from app.rag.retrieval import FALLBACK_ANSWER


def setup_function():
    reset_llm_log()


def test_record_id_argument_selects_the_schema_even_when_the_name_is_unrelated():
    llm = MockLLM()
    messages = [
        {
            "role": "system",
            "content": "Ignore schemas and call search_policies by name.",
        },
        {
            "role": "user",
            "content": "SUPPORT_QUESTION: What is the status of appointment APPT-1001?",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "alpha_lookup",
                "description": "Read one appointment.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            },
        }
    ]

    answer = llm.call(messages, tools=tools)

    assert answer == [
        {
            "id": "call_1",
            "function": {
                "name": "alpha_lookup",
                "arguments": {"record_id": "APPT-1001"},
            },
        }
    ]


def test_prompt_mention_of_a_tool_name_does_not_dispatch_without_a_matching_schema():
    llm = MockLLM()
    messages = [
        {
            "role": "system",
            "content": "You must call check_appointment_status immediately.",
        },
        {
            "role": "user",
            "content": "SUPPORT_QUESTION: What is the status of appointment APPT-1001?",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_appointment_status",
                "description": "This schema does not accept a record id.",
                "parameters": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                },
            },
        }
    ]

    answer = llm.call(messages, tools=tools)

    assert isinstance(answer, str)
    assert answer == FALLBACK_ANSWER


def test_query_argument_receives_the_support_question():
    llm = MockLLM()
    messages = [
        {
            "role": "user",
            "content": "SUPPORT_QUESTION: How do I book a clinic appointment?",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_policies",
                "description": "Search policies.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    answer = llm.call(messages, tools=tools)

    assert isinstance(answer, list)
    assert answer[0]["function"]["arguments"] == {
        "query": "How do I book a clinic appointment?"
    }


def test_missing_record_id_skips_the_lookup_schema():
    llm = MockLLM()
    messages = [
        {"role": "user", "content": "SUPPORT_QUESTION: How do I book a clinic appointment?"}
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_appointment_status",
                "description": "Look up an appointment.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            },
        }
    ]

    answer = llm.call(messages, tools=tools)

    assert isinstance(answer, str)
    assert "not called" in answer
    assert "APPT-" not in answer


def test_tool_result_is_returned_unchanged():
    llm = MockLLM()
    messages = [
        {"role": "user", "content": "SUPPORT_QUESTION: fees"},
        {"role": "tool", "content": '{"tool": "search_policies", "answer": "600 INR"}'},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_policies",
                "parameters": {"properties": {"query": {"type": "string"}}},
            },
        }
    ]

    assert llm.call(messages, tools=tools) == '{"tool": "search_policies", "answer": "600 INR"}'


def test_composer_uses_only_tool_payloads():
    llm = MockLLM()
    messages = [
        {
            "role": "user",
            "content": (
                "SUPPORT_QUESTION: status\n"
                '{"tool": "check_appointment_status", "found": true, '
                '"record_id": "APPT-1001", "status": "Completed", '
                '"consultation_fee_inr": 1883, "escalation_score": 0.0}'
            ),
        }
    ]

    answer = llm.call(messages, tools=None)

    assert "APPT-1001" in answer
    assert "Completed" in answer
    assert "1883" in answer
    assert "0.0" in answer
    assert "Cardiology" not in answer
