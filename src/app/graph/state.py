from enum import StrEnum
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from app.data.models import Order, OrderStatus


class EscalationReason(StrEnum):
    """Distinct, taggable reasons for handing off to a human."""

    OUT_OF_SCOPE = "out_of_scope"
    GUARDRAIL_FAILURE = "guardrail_failure"
    DATA_INCONSISTENCY = "data_inconsistency"
    LOOP_LIMIT = "loop_limit"
    USER_REQUESTED = "user_requested"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: str | None
    order: Order | None
    order_status: OrderStatus | None
    escalation_reason: EscalationReason | None
    retry_count: int
    turn_count: int
