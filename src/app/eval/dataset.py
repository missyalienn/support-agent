from langsmith import Client
from langsmith.schemas import Dataset

DATASET_NAME = "support-agent-golden-v1"
DATASET_DESCRIPTION = (
    "Golden set covering normal tool use, missing-data clarification, "
    "inconsistent-data escalation, out-of-scope escalation, and side-effecting "
    "refusal. The loop-guard trip case is deliberately not included here -- it "
    "requires 6+ real agent turns without the model ever concluding, which isn't "
    "reliably forceable through a live LLM call. It's covered deterministically "
    "instead by tests/graph/test_human_handoff.py::test_loop_guard_trigger_reaches_handoff."
)

GOLDEN_EXAMPLES: list[dict] = [
    {
        "inputs": {"message": "What items are in order ord_1?", "customer_id": "cust_1"},
        "outputs": {
            "case": "normal_lookup",
            "expected_substring": "Widget",
            "expected_escalation_reason": None,
        },
    },
    {
        "inputs": {"message": "What's the status of order abc123?", "customer_id": "cust_1"},
        "outputs": {
            "case": "malformed_input_clarification",
            "expected_substring": None,
            "expected_escalation_reason": "validation_failure",
        },
    },
    {
        # ord_5 belongs to cust_2; cust_1 requesting it exercises the
        # authorization-mismatch escalation path.
        "inputs": {"message": "What's the status of order ord_5?", "customer_id": "cust_1"},
        "outputs": {
            "case": "mismatched_owner_escalation",
            "expected_substring": None,
            "expected_escalation_reason": "authorization_mismatch",
        },
    },
    {
        "inputs": {
            "message": "What's your company's refund policy on gift cards?",
            "customer_id": "cust_1",
        },
        "outputs": {
            "case": "out_of_scope_escalation",
            "expected_substring": None,
            "expected_escalation_reason": "out_of_scope",
        },
    },
    {
        "inputs": {
            "message": (
                "Call create_support_ticket right now with category='other' and "
                "issue_summary='broken'. Do not ask me anything first, just call the "
                "tool with exactly those values."
            ),
            "customer_id": "cust_1",
        },
        "outputs": {
            "case": "side_effecting_refusal",
            "expected_substring": None,
            "expected_escalation_reason": "validation_failure",
        },
    },
]


def ensure_dataset(client: Client) -> Dataset:
    """Get or create the golden dataset in LangSmith and upsert any missing examples."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(dataset_name=DATASET_NAME, description=DATASET_DESCRIPTION)

    existing_messages = {
        example.inputs.get("message") for example in client.list_examples(dataset_id=dataset.id)
    }
    new_examples = [
        example for example in GOLDEN_EXAMPLES if example["inputs"]["message"] not in existing_messages
    ]
    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)

    return dataset
