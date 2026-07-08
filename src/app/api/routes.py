import structlog
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.graph.run import run_turn

logger = structlog.get_logger(__name__)

router = APIRouter()

HANDOFF_MESSAGE = "This conversation has been escalated to a human agent."

# In-memory session store: conversation_id -> customer_id, enforced for
# consistency across turns (not a lookup on the client's behalf).
_conversation_customers: dict[str, str] = {}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn of the support graph for a conversation, returning a reply or handoff notice."""
    known_customer_id = _conversation_customers.get(request.conversation_id)
    if known_customer_id is not None and known_customer_id != request.customer_id:
        logger.warning(
            "chat_customer_id_mismatch",
            conversation_id=request.conversation_id,
            known_customer_id=known_customer_id,
            requested_customer_id=request.customer_id,
        )
        raise HTTPException(
            status_code=400,
            detail="customer_id does not match the customer_id already associated with this conversation_id",
        )
    _conversation_customers[request.conversation_id] = request.customer_id

    final_state = run_turn(
        request.message,
        customer_id=request.customer_id,
        conversation_id=request.conversation_id,
    )

    interrupts = final_state.get("__interrupt__")
    if interrupts:
        reason = interrupts[0].value["reason"]
        return ChatResponse(
            conversation_id=request.conversation_id,
            message=HANDOFF_MESSAGE,
            escalated=True,
            escalation_reason=str(reason),
        )

    return ChatResponse(
        conversation_id=request.conversation_id,
        message=final_state["messages"][-1].content,
        escalated=False,
    )
