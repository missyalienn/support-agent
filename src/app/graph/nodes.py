import structlog
from langchain_anthropic import ChatAnthropic

from app.config import settings
from app.graph.state import AgentState
from app.tools.order_tools import get_order

logger = structlog.get_logger(__name__)

_llm = ChatAnthropic(
    model=settings.anthropic_model,
    api_key=settings.anthropic_api_key,
).bind_tools([get_order])


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
    """LLM node bound to get_order; selects the tool or responds directly."""
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


def guardrail(state: AgentState) -> dict:
    """Pass-through stub; real pre-execution validation lands in a later phase."""
    logger.info("guardrail_passthrough")
    return {}


def check_result(state: AgentState) -> dict:
    """Pass-through stub; real result validation lands in a later phase."""
    logger.info("check_result_passthrough")
    return {}
