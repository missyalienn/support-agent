from langchain.tools import ToolRuntime

from app.tools.ticket_tools import create_support_ticket


def _runtime(customer_id: str) -> ToolRuntime:
    return ToolRuntime(
        state={"customer_id": customer_id},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id=None,
        store=None,
    )


def test_create_support_ticket_persists_and_returns_ticket() -> None:
    result = create_support_ticket.func(
        category="shipping",
        issue_summary="Package marked delivered but customer says it never arrived",
        runtime=_runtime("cust_1"),
    )

    assert result.created is True
    assert result.ticket_id
