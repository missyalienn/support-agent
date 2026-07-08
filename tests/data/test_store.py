from app.data.store import get_order_by_id, get_status_by_id, save_ticket


def test_get_order_by_id_found() -> None:
    order = get_order_by_id("ord_1")

    assert order is not None
    assert order.customer_id == "cust_1"


def test_get_order_by_id_not_found() -> None:
    assert get_order_by_id("does_not_exist") is None


def test_get_status_by_id_found() -> None:
    assert get_status_by_id("ord_3") == "pending"


def test_get_status_by_id_processing() -> None:
    assert get_status_by_id("ord_6") == "processing"


def test_get_status_by_id_returned() -> None:
    assert get_status_by_id("ord_7") == "returned"


def test_get_status_by_id_not_found() -> None:
    assert get_status_by_id("does_not_exist") is None


def test_mismatched_owner_is_detectable() -> None:
    order = get_order_by_id("ord_5")

    assert order is not None
    assert order.customer_id == "cust_2"
    assert order.customer_id != "cust_1"


def test_save_ticket_persists_and_returns_ticket() -> None:
    ticket = save_ticket(
        customer_id="cust_1",
        category="shipping",
        issue_summary="Package marked delivered but never arrived",
    )

    assert ticket.customer_id == "cust_1"
    assert ticket.category == "shipping"
    assert ticket.ticket_id
