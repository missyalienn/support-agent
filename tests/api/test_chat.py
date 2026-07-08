from fastapi.testclient import TestClient

from app.data import store
from app.main import app

client = TestClient(app)


def test_lookup_then_ticket_conversation() -> None:
    conversation_id = "api-test-lookup-ticket"
    customer_id = "cust_1"

    lookup_response = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "message": "What items are in order ord_1?",
        },
    )
    assert lookup_response.status_code == 200
    lookup_body = lookup_response.json()
    assert lookup_body["escalated"] is False
    assert "Widget" in lookup_body["message"]

    ticket_count_before = len(store._TICKETS)

    ticket_response = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "message": (
                "Please create a support ticket with category 'damaged_item' and issue_summary "
                "'The widget from order ord_1 arrived cracked and unusable and needs a "
                "replacement.' Use exactly that category and summary."
            ),
        },
    )
    assert ticket_response.status_code == 200
    ticket_body = ticket_response.json()
    assert ticket_body["escalated"] is False
    assert len(store._TICKETS) == ticket_count_before + 1


def test_user_requested_handoff() -> None:
    response = client.post(
        "/chat",
        json={
            "conversation_id": "api-test-escalation",
            "customer_id": "cust_1",
            "message": "I don't want to talk to a bot, please connect me with a human agent.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["escalated"] is True
    assert body["escalation_reason"] == "user_requested"


def test_mismatched_customer_id_for_existing_conversation_is_rejected() -> None:
    conversation_id = "api-test-mismatch"
    client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "customer_id": "cust_1",
            "message": "What items are in order ord_1?",
        },
    )

    response = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "customer_id": "cust_2",
            "message": "What items are in order ord_1?",
        },
    )
    assert response.status_code == 400
