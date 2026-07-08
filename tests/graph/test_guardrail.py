from langchain_core.messages import AIMessage

from app.graph.build import route_after_guardrail
from app.graph.nodes import guardrail, human_handoff
from app.graph.state import EscalationReason


def _tool_call_message(order_id: str | None, tool_call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": order_id}, "id": tool_call_id}],
    )


def test_guardrail_passes_valid_order_id() -> None:
    state = {"messages": [_tool_call_message("ord_1")], "retry_count": 0}

    result = guardrail(state)

    assert result == {"retry_count": 0, "escalation_reason": None}


def test_guardrail_fails_missing_order_id() -> None:
    state = {"messages": [_tool_call_message(None)], "retry_count": 0}

    result = guardrail(state)

    assert result["retry_count"] == 1
    assert result["escalation_reason"] == EscalationReason.VALIDATION_FAILURE
    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "call_1"


def test_guardrail_fails_malformed_order_id() -> None:
    state = {"messages": [_tool_call_message("not-an-order")], "retry_count": 0}

    result = guardrail(state)

    assert result["retry_count"] == 1


def test_route_after_guardrail_pass_goes_to_tools() -> None:
    assert route_after_guardrail({"retry_count": 0}) == "tools"


def test_route_after_guardrail_first_failure_goes_to_agent() -> None:
    assert route_after_guardrail({"retry_count": 1}) == "agent"


def test_route_after_guardrail_second_failure_goes_to_handoff() -> None:
    assert route_after_guardrail({"retry_count": 2}) == "human_handoff"


def test_human_handoff_is_passthrough() -> None:
    state = {"escalation_reason": EscalationReason.VALIDATION_FAILURE}

    result = human_handoff(state)

    assert result == {}


def test_guardrail_flow_retry_then_success() -> None:
    state = {"messages": [_tool_call_message("bad-id")], "retry_count": 0}
    first = guardrail(state)
    assert route_after_guardrail(first) == "agent"

    state = {"messages": [_tool_call_message("ord_2")], "retry_count": first["retry_count"]}
    second = guardrail(state)

    assert route_after_guardrail(second) == "tools"
    assert second["retry_count"] == 0


def test_guardrail_flow_two_failures_then_handoff() -> None:
    state = {"messages": [_tool_call_message("bad-id")], "retry_count": 0}
    first = guardrail(state)
    assert route_after_guardrail(first) == "agent"

    state = {"messages": [_tool_call_message(None)], "retry_count": first["retry_count"]}
    second = guardrail(state)

    assert route_after_guardrail(second) == "human_handoff"
    assert second["retry_count"] == 2
