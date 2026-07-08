from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.graph.nodes import agent, check_result, guardrail, human_handoff, intake
from app.graph.state import AgentState
from app.tools.order_tools import get_order, get_order_status


def should_continue(state: AgentState) -> str:
    """Route to guardrail if agent proposed a tool call, else end the turn."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "guardrail"
    return END


def route_after_guardrail(state: AgentState) -> str:
    """Route to tools on pass, back to agent on first failure, handoff on the second."""
    retry_count = state["retry_count"]
    if retry_count == 0:
        return "tools"
    if retry_count >= 2:
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
    graph.add_node("human_handoff", human_handoff)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "agent")
    graph.add_conditional_edges("agent", should_continue, {"guardrail": "guardrail", END: END})
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"tools": "tools", "agent": "agent", "human_handoff": "human_handoff"},
    )
    graph.add_edge("tools", "check_result")
    graph.add_edge("check_result", "agent")
    graph.add_edge("human_handoff", END)

    return graph.compile()
