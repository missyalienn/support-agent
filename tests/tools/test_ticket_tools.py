from app.tools.ticket_tools import create_support_ticket


def test_create_support_ticket_persists_and_returns_ticket() -> None:
    result = create_support_ticket.func(
        category="shipping",
        issue_summary="Package marked delivered but customer says it never arrived",
        state={"customer_id": "cust_1"},
    )

    assert result.created is True
    assert result.ticket_id
