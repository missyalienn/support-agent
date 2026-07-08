from langchain_core.messages import AIMessage, ToolMessage

from app.graph.build import route_after_guardrail
from app.graph.nodes import ticket_guardrail
from app.graph.state import EscalationReason


def _ticket_call_message(
    issue_summary: str, category: str = "shipping", tool_call_id: str = "call_1"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_support_ticket",
                "args": {"category": category, "issue_summary": issue_summary},
                "id": tool_call_id,
            }
        ],
    )


def test_ticket_guardrail_passes_specific_summary() -> None:
    state = {
        "messages": [_ticket_call_message("Package marked delivered but never arrived")],
        "retry_count": 0,
    }

    result = ticket_guardrail(state)

    assert result == {"retry_count": 0, "escalation_reason": None}


def test_ticket_guardrail_rejects_vague_summary() -> None:
    state = {"messages": [_ticket_call_message("it's broken")], "retry_count": 0}

    result = ticket_guardrail(state)

    assert result["retry_count"] == 1
    assert result["escalation_reason"] == EscalationReason.VALIDATION_FAILURE
    assert result["messages"][0].tool_call_id == "call_1"


def test_ticket_guardrail_flow_vague_then_specific_creates_single_ticket() -> None:
    state = {"messages": [_ticket_call_message("it's broken")], "retry_count": 0}
    first = ticket_guardrail(state)
    assert route_after_guardrail(first) == "agent"

    state = {
        "messages": [_ticket_call_message("Package marked delivered but never arrived")],
        "retry_count": first["retry_count"],
    }
    second = ticket_guardrail(state)

    assert route_after_guardrail(second) == "tools"
    assert second["retry_count"] == 0


def test_ticket_guardrail_blocks_duplicate_ticket_same_conversation() -> None:
    prior_call = _ticket_call_message(
        "Package marked delivered but never arrived", category="shipping", tool_call_id="call_1"
    )
    prior_result = ToolMessage(content="created=True ticket_id='tick_1_abc'", tool_call_id="call_1")
    new_call = _ticket_call_message(
        "Package marked delivered but never arrived", category="shipping", tool_call_id="call_2"
    )
    state = {"messages": [prior_call, prior_result, new_call], "retry_count": 0}

    result = ticket_guardrail(state)

    assert result["retry_count"] == 1
    assert result["escalation_reason"] == EscalationReason.VALIDATION_FAILURE
    assert route_after_guardrail(result) == "agent"


def test_ticket_guardrail_allows_different_category_after_prior_ticket() -> None:
    prior_call = _ticket_call_message(
        "Package marked delivered but never arrived", category="shipping", tool_call_id="call_1"
    )
    prior_result = ToolMessage(content="created=True ticket_id='tick_1_abc'", tool_call_id="call_1")
    new_call = _ticket_call_message(
        "Charged twice for the same order", category="billing", tool_call_id="call_2"
    )
    state = {"messages": [prior_call, prior_result, new_call], "retry_count": 0}

    result = ticket_guardrail(state)

    assert result["escalation_reason"] is None
