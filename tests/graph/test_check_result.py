from langchain_core.messages import AIMessage, ToolMessage

from app.graph.build import route_after_check_result
from app.graph.nodes import check_result
from app.graph.state import EscalationReason


def _tool_call_state(order_id: str, customer_id: str) -> dict:
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": order_id}, "id": "call_1"}],
    )
    tool_message = ToolMessage(content="found=True status='shipped'", tool_call_id="call_1")
    return {"messages": [ai_message, tool_message], "customer_id": customer_id}


def test_check_result_flags_authorization_mismatch() -> None:
    # ord_5 is owned by cust_2; cust_1 requesting it is a mismatch.
    state = _tool_call_state(order_id="ord_5", customer_id="cust_1")

    result = check_result(state)

    assert result["escalation_reason"] == EscalationReason.AUTHORIZATION_MISMATCH


def test_check_result_passes_matching_customer() -> None:
    state = _tool_call_state(order_id="ord_5", customer_id="cust_2")

    result = check_result(state)

    assert result["escalation_reason"] is None


def test_route_after_check_result_mismatch_goes_to_handoff() -> None:
    assert route_after_check_result({"escalation_reason": EscalationReason.AUTHORIZATION_MISMATCH}) == "human_handoff"


def test_route_after_check_result_pass_goes_to_agent() -> None:
    assert route_after_check_result({"escalation_reason": None}) == "agent"
