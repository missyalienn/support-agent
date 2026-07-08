from datetime import datetime
from itertools import count
from uuid import uuid4

from app.data.models import Customer, Order, OrderStatus, Ticket

_CUSTOMERS: list[Customer] = [
    Customer(customer_id="cust_1", name="Ada Lovelace", email="ada@example.com"),
    Customer(customer_id="cust_2", name="Grace Hopper", email="grace@example.com"),
    Customer(customer_id="cust_3", name="Alan Turing", email="alan@example.com"),
]

_ORDERS: list[Order] = [
    Order(
        order_id="ord_1",
        customer_id="cust_1",
        items=["Widget"],
        status=OrderStatus.DELIVERED,
        created_at=datetime(2026, 1, 5),
    ),
    Order(
        order_id="ord_2",
        customer_id="cust_1",
        items=["Gadget", "Gizmo"],
        status=OrderStatus.SHIPPED,
        created_at=datetime(2026, 2, 10),
    ),
    Order(
        order_id="ord_3",
        customer_id="cust_2",
        items=["Widget"],
        status=OrderStatus.PENDING,
        created_at=datetime(2026, 3, 1),
    ),
    Order(
        order_id="ord_4",
        customer_id="cust_3",
        items=["Doohickey"],
        status=OrderStatus.CANCELLED,
        created_at=datetime(2026, 3, 15),
    ),
    # Deliberately owned by cust_2 so a caller claiming cust_1 (or cust_3) for
    # this order exercises the order/customer mismatch path used by later
    # guardrail and check_result tests.
    Order(
        order_id="ord_5",
        customer_id="cust_2",
        items=["Widget"],
        status=OrderStatus.SHIPPED,
        created_at=datetime(2026, 4, 20),
    ),
]

_TICKETS: list[Ticket] = []

_ticket_ids = count(1)


def get_order_by_id(order_id: str) -> Order | None:
    """Look up an order by id; None if it doesn't exist."""
    return next((order for order in _ORDERS if order.order_id == order_id), None)


def get_status_by_id(order_id: str) -> OrderStatus | None:
    """Look up an order's status by id; None if the order doesn't exist."""
    order = get_order_by_id(order_id)
    return order.status if order is not None else None


def save_ticket(customer_id: str, category: str, issue_summary: str) -> Ticket:
    """Persist a new support ticket and return it."""
    ticket = Ticket(
        ticket_id=f"tick_{next(_ticket_ids)}_{uuid4().hex[:8]}",
        customer_id=customer_id,
        category=category,
        issue_summary=issue_summary,
        created_at=datetime.now(),
    )
    _TICKETS.append(ticket)
    return ticket
