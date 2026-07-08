from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.data.models import Order, OrderStatus


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(description="The order ID to look up, e.g. 'ord_1'.")


class GetOrderOutput(BaseModel):
    found: bool = Field(description="Whether an order with this ID exists.")
    order: Order | None = Field(default=None, description="The order details, if found.")


class GetOrderStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(description="The order ID to look up, e.g. 'ord_1'.")


class GetOrderStatusOutput(BaseModel):
    found: bool = Field(description="Whether an order with this ID exists.")
    status: OrderStatus | None = Field(default=None, description="The order's status, if found.")


class EscalateToHumanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Literal["out_of_scope", "user_requested"] = Field(
        description="Why this conversation needs a human: 'out_of_scope' for requests "
        "unrelated to order support, 'user_requested' if the user explicitly asked for a person."
    )
    summary: str = Field(description="A short summary of the user's request for the human agent.")


class EscalateToHumanOutput(BaseModel):
    acknowledged: bool = Field(description="Whether the escalation was acknowledged.")


class CreateSupportTicketOutput(BaseModel):
    created: bool = Field(description="Whether the ticket was created.")
    ticket_id: str | None = Field(default=None, description="The created ticket's ID, if created.")
