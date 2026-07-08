from langchain_core.tools import tool

from app.tools.schemas import EscalateToHumanInput, EscalateToHumanOutput


@tool(args_schema=EscalateToHumanInput)
def escalate_to_human(reason: str, summary: str) -> EscalateToHumanOutput:
    """Flag this conversation for human handoff instead of answering directly."""
    return EscalateToHumanOutput(acknowledged=True)
