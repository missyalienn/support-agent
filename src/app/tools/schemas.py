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
