from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.graph.nodes import agent, check_result, guardrail, handle_escalation, human_handoff, intake
from app.graph.state import AgentState, EscalationReason
from app.tools.order_tools import get_order, get_order_status

ESCALATE_TOOL_NAME = "escalate_to_human"


def should_continue(state: AgentState) -> str:
    """Route to human_handoff on the loop guard, escalation on that tool call, else guardrail/end."""
    if state.get("escalation_reason") == EscalationReason.MAX_ITERATIONS_EXCEEDED:
        return "human_handoff"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return END
    if tool_calls[0]["name"] == ESCALATE_TOOL_NAME:
        return "handle_escalation"
    return "guardrail"


def route_after_guardrail(state: AgentState) -> str:
    """Route to tools on pass, back to agent on first failure, handoff on the second."""
    retry_count = state["retry_count"]
    if retry_count == 0:
        return "tools"
    if retry_count >= 2:
        return "human_handoff"
    return "agent"


def route_after_check_result(state: AgentState) -> str:
    """Route to handoff on an authorization mismatch, else back to agent."""
    if state["escalation_reason"] is not None:
        return "human_handoff"
    return "agent"


def build_graph() -> CompiledStateGraph:
    """Assemble the intake -> agent <-> (guardrail -> tools -> check_result) -> agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake)
    graph.add_node("agent", agent)
    graph.add_node("guardrail", guardrail)
    graph.add_node("tools", ToolNode([get_order, get_order_status]))
    graph.add_node("check_result", check_result)
    graph.add_node("handle_escalation", handle_escalation)
    graph.add_node("human_handoff", human_handoff)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"guardrail": "guardrail", "handle_escalation": "handle_escalation", "human_handoff": "human_handoff", END: END},
    )
    graph.add_edge("handle_escalation", "human_handoff")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"tools": "tools", "agent": "agent", "human_handoff": "human_handoff"},
    )
    graph.add_edge("tools", "check_result")
    graph.add_conditional_edges(
        "check_result",
        route_after_check_result,
        {"agent": "agent", "human_handoff": "human_handoff"},
    )
    graph.add_edge("human_handoff", END)

    return graph.compile(checkpointer=MemorySaver())
