from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.data.store import save_ticket
from app.graph.state import AgentState
from app.tools.schemas import CreateSupportTicketInput, CreateSupportTicketOutput


@tool(args_schema=CreateSupportTicketInput)
def create_support_ticket(
    category: str,
    issue_summary: str,
    state: Annotated[AgentState, InjectedState],
) -> CreateSupportTicketOutput:
    """Create a support ticket for the customer's issue. Only call this once per issue."""
    ticket = save_ticket(
        customer_id=state["customer_id"], category=category, issue_summary=issue_summary
    )
    return CreateSupportTicketOutput(created=True, ticket_id=ticket.ticket_id)
