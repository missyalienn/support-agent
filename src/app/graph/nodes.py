import re

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.config import settings
from app.data.store import get_order_by_id
from app.graph.state import AgentState, EscalationReason
from app.tools.escalation_tools import escalate_to_human
from app.tools.order_tools import get_order, get_order_status

logger = structlog.get_logger(__name__)

_ORDER_ID_PATTERN = re.compile(r"^ord_\d+$")
MAX_AGENT_TURNS = 6

_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a customer support agent for an order-fulfillment company. "
        "Use get_order or get_order_status to answer questions about a customer's own orders. "
        "If the user asks about anything unrelated to their orders, call escalate_to_human "
        "with reason='out_of_scope'. If the user explicitly asks to speak to a person or a "
        "human agent, call escalate_to_human with reason='user_requested'."
    )
)

_llm = ChatAnthropic(
    model=settings.anthropic_model,
    api_key=settings.anthropic_api_key,
).bind_tools([get_order, get_order_status, escalate_to_human])


def intake(state: AgentState) -> dict:
    """Initialize counters and defaults for a new turn; deterministic, no LLM."""
    return {
        "order_id": state.get("order_id"),
        "order": state.get("order"),
        "order_status": state.get("order_status"),
        "escalation_reason": state.get("escalation_reason"),
        "retry_count": state.get("retry_count", 0),
        "turn_count": state.get("turn_count", 0) + 1,
    }


def agent(state: AgentState) -> dict:
    """LLM node bound to order tools; selects a tool or responds directly."""
    turn_count = state.get("turn_count", 0) + 1
    if turn_count > MAX_AGENT_TURNS:
        logger.warning("agent_loop_guard_tripped", turn_count=turn_count)
        return {
            "turn_count": turn_count,
            "escalation_reason": EscalationReason.MAX_ITERATIONS_EXCEEDED,
        }

    response = _llm.invoke([_SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response], "turn_count": turn_count}


def guardrail(state: AgentState) -> dict:
    """Validate the proposed tool call's order_id before allowing execution."""
    tool_call = state["messages"][-1].tool_calls[0]
    order_id = tool_call.get("args", {}).get("order_id")

    if order_id and _ORDER_ID_PATTERN.match(order_id):
        logger.info("guardrail_passed", order_id=order_id)
        return {"retry_count": 0, "escalation_reason": None}

    retry_count = state.get("retry_count", 0) + 1
    logger.warning("guardrail_failed", order_id=order_id, retry_count=retry_count)
    error_message = ToolMessage(
        content=f"Invalid order_id {order_id!r}: must match 'ord_<number>'.",
        tool_call_id=tool_call["id"],
    )
    return {
        "messages": [error_message],
        "retry_count": retry_count,
        "escalation_reason": EscalationReason.VALIDATION_FAILURE,
    }


def check_result(state: AgentState) -> dict:
    """Verify the fetched order belongs to the requesting customer."""
    tool_call = state["messages"][-2].tool_calls[0]
    order_id = tool_call.get("args", {}).get("order_id")
    order = get_order_by_id(order_id)

    if order is not None and order.customer_id != state["customer_id"]:
        logger.warning(
            "check_result_authorization_mismatch",
            order_id=order_id,
            requesting_customer_id=state["customer_id"],
            order_customer_id=order.customer_id,
        )
        return {"escalation_reason": EscalationReason.AUTHORIZATION_MISMATCH}

    logger.info("check_result_passed", order_id=order_id)
    return {"escalation_reason": None}


_ESCALATION_REASON_MAP = {
    "out_of_scope": EscalationReason.OUT_OF_SCOPE,
    "user_requested": EscalationReason.USER_REQUESTED,
}


def handle_escalation(state: AgentState) -> dict:
    """Translate an escalate_to_human tool call into an escalation_reason.

    Bypasses the order guardrail's format check (irrelevant to this tool) without
    touching order/order_id/escalation_reason context already in state -- only the
    escalation_reason key is returned here.
    """
    tool_call = state["messages"][-1].tool_calls[0]
    reason = _ESCALATION_REASON_MAP[tool_call["args"]["reason"]]
    logger.info("handle_escalation", reason=reason, summary=tool_call["args"].get("summary"))

    ack_message = ToolMessage(content="Escalating to a human.", tool_call_id=tool_call["id"])
    return {"messages": [ack_message], "escalation_reason": reason}


def human_handoff(state: AgentState) -> dict:
    """Build the handoff payload (transcript, partial state, reason) and pause for a human."""
    transcript = [
        {"role": message.type, "content": message.content} for message in state["messages"]
    ]
    order = state.get("order")
    payload = {
        "reason": state.get("escalation_reason"),
        "transcript": transcript,
        "partial_state": {
            "customer_id": state.get("customer_id"),
            "order_id": state.get("order_id"),
            "order": order.model_dump() if order is not None else None,
            "order_status": state.get("order_status"),
        },
    }
    logger.info("human_handoff", reason=payload["reason"])
    interrupt(payload)
    return {}
