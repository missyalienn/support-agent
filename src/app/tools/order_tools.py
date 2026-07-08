from langchain_core.tools import tool

from app.data.store import get_order_by_id, get_status_by_id
from app.tools.schemas import (
    GetOrderInput,
    GetOrderOutput,
    GetOrderStatusInput,
    GetOrderStatusOutput,
)


@tool(args_schema=GetOrderInput)
def get_order(order_id: str) -> GetOrderOutput:
    """Look up full order details by order ID."""
    order = get_order_by_id(order_id)
    return GetOrderOutput(found=order is not None, order=order)


@tool(args_schema=GetOrderStatusInput)
def get_order_status(order_id: str) -> GetOrderStatusOutput:
    """Look up an order's current status by order ID."""
    status = get_status_by_id(order_id)
    return GetOrderStatusOutput(found=status is not None, status=status)
