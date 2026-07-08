from pydantic import BaseModel


class ChatRequest(BaseModel):
    """A single chat turn: which conversation, which customer, what they said."""

    conversation_id: str
    customer_id: str
    message: str


class ChatResponse(BaseModel):
    """The agent's reply, or a handoff notice if the turn ended in escalation."""

    conversation_id: str
    message: str
    escalated: bool
    escalation_reason: str | None = None
