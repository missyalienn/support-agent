from app.tools.order_tools import get_order, get_order_status


def test_get_order_found() -> None:
    result = get_order.invoke({"order_id": "ord_1"})

    assert result.found is True
    assert result.order is not None
    assert result.order.customer_id == "cust_1"


def test_get_order_not_found() -> None:
    result = get_order.invoke({"order_id": "does_not_exist"})

    assert result.found is False
    assert result.order is None


def test_get_order_status_found() -> None:
    result = get_order_status.invoke({"order_id": "ord_3"})

    assert result.found is True
    assert result.status == "pending"


def test_get_order_status_not_found() -> None:
    result = get_order_status.invoke({"order_id": "does_not_exist"})

    assert result.found is False
    assert result.status is None
