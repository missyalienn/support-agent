from langchain_core.messages import HumanMessage

from app.graph.build import build_graph
from app.graph.state import AgentState

_graph = build_graph()


def run_turn(user_message: str) -> AgentState:
    """Invoke the graph with a single user message and return the final state."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "order_id": None,
        "order": None,
        "order_status": None,
        "escalation_reason": None,
        "retry_count": 0,
        "turn_count": 0,
    }
    return _graph.invoke(initial_state)
