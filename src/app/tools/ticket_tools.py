from typing import Annotated

from langchain.tools import ToolRuntime, tool
from pydantic import Field

from app.data.store import save_ticket
from app.tools.schemas import CreateSupportTicketOutput


@tool
def create_support_ticket(
    category: Annotated[str, Field(description="Short category for the issue, e.g. 'shipping', 'billing'.")],
    issue_summary: Annotated[
        str,
        Field(
            description="A specific, detailed description of the customer's issue. "
            "Vague or one-word summaries will be rejected."
        ),
    ],
    runtime: ToolRuntime,
) -> CreateSupportTicketOutput:
    """Create a support ticket for the customer's issue. Only call this once per issue."""
    ticket = save_ticket(
        customer_id=runtime.state["customer_id"], category=category, issue_summary=issue_summary
    )
    return CreateSupportTicketOutput(created=True, ticket_id=ticket.ticket_id)
