from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class OrderStatus(StrEnum):
    """Lifecycle state of an order."""

    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str


class Order(BaseModel):
    order_id: str
    customer_id: str
    items: list[str]
    status: OrderStatus
    created_at: datetime


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    category: str
    issue_summary: str
    created_at: datetime
