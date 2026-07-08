import re

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage

from app.config import settings
from app.graph.state import AgentState, EscalationReason
from app.tools.order_tools import get_order, get_order_status

logger = structlog.get_logger(__name__)

_ORDER_ID_PATTERN = re.compile(r"^ord_\d+$")

_llm = ChatAnthropic(
    model=settings.anthropic_model,
    api_key=settings.anthropic_api_key,
).bind_tools([get_order, get_order_status])


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
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


def guardrail(state: AgentState) -> dict:
    """Validate the proposed tool call's order_id before allowing execution."""
    tool_call = state["messages"][-1].tool_calls[0]
    order_id = tool_call.get("args", {}).get("order_id")

    if order_id and _ORDER_ID_PATTERN.match(order_id):
        logger.info("guardrail_passed", order_id=order_id)
        return {"retry_count": 0}

    retry_count = state.get("retry_count", 0) + 1
    logger.warning("guardrail_failed", order_id=order_id, retry_count=retry_count)
    error_message = ToolMessage(
        content=f"Invalid order_id {order_id!r}: must match 'ord_<number>'.",
        tool_call_id=tool_call["id"],
    )
    return {"messages": [error_message], "retry_count": retry_count}


def check_result(state: AgentState) -> dict:
    """Pass-through stub; real result validation lands in a later phase."""
    logger.info("check_result_passthrough")
    return {}


def human_handoff(state: AgentState) -> dict:
    """Stub handoff node; real escalation logic lands in a later phase."""
    logger.info("human_handoff", reason=EscalationReason.GUARDRAIL_FAILURE)
    return {"escalation_reason": EscalationReason.GUARDRAIL_FAILURE}
