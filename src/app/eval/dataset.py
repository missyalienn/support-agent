from langsmith import Client
from langsmith.schemas import Dataset

DATASET_NAME = "support-agent-golden-v1"
DATASET_DESCRIPTION = (
    "First golden set covering normal lookup, malformed-input clarification, "
    "and mismatched-owner escalation."
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
