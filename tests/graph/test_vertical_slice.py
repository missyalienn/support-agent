from app.graph.run import run_turn


def test_order_query_produces_grounded_response() -> None:
    final_state = run_turn("What items are in order ord_1?", customer_id="cust_1")

    response = final_state["messages"][-1].content

    assert "Widget" in response
