from app.eval.dataset import GOLDEN_EXAMPLES
from app.eval.run_eval import contains_expected_substring, correct_escalation_reason


def test_golden_examples_cover_the_three_documented_scenarios() -> None:
    cases = {example["outputs"]["case"] for example in GOLDEN_EXAMPLES}

    assert cases == {
        "normal_lookup",
        "malformed_input_clarification",
        "mismatched_owner_escalation",
    }


def test_golden_examples_have_required_input_fields() -> None:
    for example in GOLDEN_EXAMPLES:
        assert example["inputs"]["message"]
        assert example["inputs"]["customer_id"]


def test_correct_escalation_reason_passes_on_match() -> None:
    outputs = {"escalation_reason": "authorization_mismatch"}
    reference_outputs = {"expected_escalation_reason": "authorization_mismatch"}

    assert correct_escalation_reason(outputs, reference_outputs) is True


def test_correct_escalation_reason_fails_on_mismatch() -> None:
    outputs = {"escalation_reason": None}
    reference_outputs = {"expected_escalation_reason": "validation_failure"}

    assert correct_escalation_reason(outputs, reference_outputs) is False


def test_contains_expected_substring_passes_when_present() -> None:
    outputs = {"response": "Your order contains a Widget."}
    reference_outputs = {"expected_substring": "Widget"}

    assert contains_expected_substring(outputs, reference_outputs) is True


def test_contains_expected_substring_fails_when_absent() -> None:
    outputs = {"response": "Your order contains a Gadget."}
    reference_outputs = {"expected_substring": "Widget"}

    assert contains_expected_substring(outputs, reference_outputs) is False


def test_contains_expected_substring_trivially_passes_when_none_expected() -> None:
    outputs = {"response": "Can you clarify the order id?"}
    reference_outputs = {"expected_substring": None}

    assert contains_expected_substring(outputs, reference_outputs) is True
