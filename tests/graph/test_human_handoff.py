import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graph import nodes
from app.graph.build import route_after_guardrail, should_continue
from app.graph.nodes import agent, check_result, guardrail, handle_escalation, human_handoff
from app.graph.state import EscalationReason


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "customer_id": "cust_1",
        "order_id": None,
        "order": None,
        "order_status": None,
        "escalation_reason": None,
        "retry_count": 0,
        "turn_count": 1,
    }
    state.update(overrides)
    return state


def _escalation_tool_call_message(reason: str, tool_call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "escalate_to_human",
                "args": {"reason": reason, "summary": "user needs a human"},
                "id": tool_call_id,
            }
        ],
    )


def _capture_interrupt(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured = {}

    def fake_interrupt(payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt)
    return captured


def test_out_of_scope_trigger_reaches_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_interrupt(monkeypatch)
    state = _base_state(
        messages=[_escalation_tool_call_message("out_of_scope")],
        order_id="ord_9",  # pre-existing context that must survive the bypass
    )

    escalation_result = handle_escalation(state)
    assert escalation_result["escalation_reason"] == EscalationReason.OUT_OF_SCOPE
    assert "order_id" not in escalation_result

    merged_state = {**state, **escalation_result, "messages": state["messages"] + escalation_result["messages"]}
    human_handoff(merged_state)

    assert captured["payload"]["reason"] == EscalationReason.OUT_OF_SCOPE
    assert captured["payload"]["partial_state"]["order_id"] == "ord_9"


def test_user_requested_trigger_reaches_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_interrupt(monkeypatch)
    state = _base_state(messages=[_escalation_tool_call_message("user_requested")])

    escalation_result = handle_escalation(state)
    assert escalation_result["escalation_reason"] == EscalationReason.USER_REQUESTED

    merged_state = {**state, **escalation_result, "messages": state["messages"] + escalation_result["messages"]}
    human_handoff(merged_state)

    assert captured["payload"]["reason"] == EscalationReason.USER_REQUESTED


def test_two_guardrail_failures_trigger_reaches_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_interrupt(monkeypatch)

    def _tool_call_message(order_id: str | None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_order_status", "args": {"order_id": order_id}, "id": "call_1"}],
        )

    state = _base_state(messages=[_tool_call_message("bad-id")])
    first = guardrail(state)
    assert route_after_guardrail(first) == "agent"

    state = _base_state(messages=[_tool_call_message(None)], retry_count=first["retry_count"])
    second = guardrail(state)
    assert route_after_guardrail(second) == "human_handoff"

    merged_state = {**state, **second}
    human_handoff(merged_state)

    assert captured["payload"]["reason"] == EscalationReason.VALIDATION_FAILURE


def test_authorization_mismatch_trigger_reaches_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_interrupt(monkeypatch)
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": "ord_5"}, "id": "call_1"}],
    )
    tool_message = ToolMessage(content="found=True status='shipped'", tool_call_id="call_1")
    # ord_5 is owned by cust_2; cust_1 requesting it is a mismatch.
    state = _base_state(messages=[ai_message, tool_message], customer_id="cust_1")

    result = check_result(state)
    assert result["escalation_reason"] == EscalationReason.AUTHORIZATION_MISMATCH

    merged_state = {**state, **result}
    human_handoff(merged_state)

    assert captured["payload"]["reason"] == EscalationReason.AUTHORIZATION_MISMATCH


def test_loop_guard_trigger_reaches_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_interrupt(monkeypatch)

    class _FailingLLM:
        def invoke(self, *args, **kwargs):
            raise AssertionError("LLM must not be called once the loop guard has tripped")

    monkeypatch.setattr(nodes, "_llm", _FailingLLM())

    state = _base_state(messages=[], turn_count=nodes.MAX_AGENT_TURNS)
    result = agent(state)

    assert result["escalation_reason"] == EscalationReason.MAX_ITERATIONS_EXCEEDED
    assert should_continue({**state, **result}) == "human_handoff"

    merged_state = {**state, **result}
    human_handoff(merged_state)

    assert captured["payload"]["reason"] == EscalationReason.MAX_ITERATIONS_EXCEEDED
