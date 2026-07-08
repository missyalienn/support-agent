import structlog
from langsmith import Client
from langsmith.evaluation import evaluate

from app.eval.dataset import DATASET_NAME, ensure_dataset
from app.graph.run import run_turn

logger = structlog.get_logger(__name__)


def target(inputs: dict) -> dict:
    """Run one turn of the graph and surface the fields the evaluators check."""
    final_state = run_turn(inputs["message"], customer_id=inputs["customer_id"])
    last_message = final_state["messages"][-1]
    escalation_reason = final_state.get("escalation_reason")
    return {
        "response": last_message.content,
        "escalation_reason": escalation_reason.value if escalation_reason is not None else None,
    }


def correct_escalation_reason(outputs: dict, reference_outputs: dict) -> bool:
    """Pass if the actual escalation reason matches the golden expectation."""
    return outputs.get("escalation_reason") == reference_outputs.get("expected_escalation_reason")


def contains_expected_substring(outputs: dict, reference_outputs: dict) -> bool:
    """Pass if the response contains the golden substring, or trivially pass if none is expected."""
    expected = reference_outputs.get("expected_substring")
    if expected is None:
        return True
    return expected in outputs.get("response", "")


def main() -> None:
    """Upsert the golden dataset and run the LangSmith eval against it."""
    client = Client()
    ensure_dataset(client)
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correct_escalation_reason, contains_expected_substring],
        experiment_prefix="support-agent-golden",
        client=client,
    )
    logger.info("eval_complete", results=str(results))


if __name__ == "__main__":
    main()
