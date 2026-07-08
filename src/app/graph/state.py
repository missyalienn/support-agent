from enum import StrEnum
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from app.data.models import Order, OrderStatus


class EscalationReason(StrEnum):
    """Distinct, taggable reasons for handing off to a human."""

    OUT_OF_SCOPE = "out_of_scope"
    VALIDATION_FAILURE = "validation_failure"
    AUTHORIZATION_MISMATCH = "authorization_mismatch"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    USER_REQUESTED = "user_requested"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str
    order_id: str | None
    order: Order | None
    order_status: OrderStatus | None
    escalation_reason: EscalationReason | None
    retry_count: int
    turn_count: int
