# Support Agent

A LangGraph-based customer support agent that does more than answer FAQs: it looks up real order data, takes a side-effecting action (opening a support ticket) under explicit guardrails, and hands off to a human when a request is high-stakes, inconsistent, or out of scope — instead of guessing.

Built to demonstrate a specific set of behaviors end-to-end: tool-calling on live state, deterministic guardrails around side effects, a real triggerable human handoff, and evaluated reliability via LangSmith — not just a live demo that happens to work once. Order and customer data is currently backed by an in-memory store shaped like a real API's interface, ready to swap in an actual backend.

## Capabilities demonstrated

| Capability | Where it lives |
|---|---|
| Tool-calling on live data | `get_order` / `get_order_status` (`src/app/tools/order_tools.py`) look up real in-memory order records; the LLM picks the right tool per turn |
| A real side-effecting action, gated | `create_support_ticket` (`src/app/tools/ticket_tools.py`) is refused by `ticket_guardrail` if the issue description is too vague, or if a ticket for the same issue already exists in the conversation |
| Deterministic guardrails, not just LLM judgment | `guardrail` validates tool-call arguments (e.g. order ID format) before execution; `check_result` checks tool output for ownership mismatches — both are plain code, not model calls |
| Real human handoff | `human_handoff` uses LangGraph's `interrupt()` to actually pause the graph, not just return a canned "talk to a human" string. Every handoff carries one of five distinct, taggable reasons (see below) |
| Evaluated reliability | A golden dataset (`src/app/eval/dataset.py`) runs as a LangSmith eval, asserting expected outcomes across normal use, clarification, and every escalation path — not just traced, but graded |

**Escalation reasons** (`EscalationReason` in `src/app/graph/state.py`):
`out_of_scope`, `validation_failure`, `authorization_mismatch`, `max_iterations_exceeded`, `user_requested`.

## Architecture

```
intake → agent ⇄ (guardrail → tools → check_result) → human_handoff
                ↘ END (agent responds directly, no tool needed)
```

| Node | Type | Responsibility |
|---|---|---|
| `intake` | deterministic | Initializes per-turn counters and defaults. No LLM. |
| `agent` | LLM (tool-calling) | Picks a tool, asks a clarifying question, responds directly, or flags escalation. |
| `guardrail` | deterministic | Validates a proposed tool call's arguments before it executes. |
| `ticket_guardrail` | deterministic | Same idea, specific to `create_support_ticket`: rejects vague summaries and duplicate tickets. |
| `tools` | LangGraph `ToolNode` | Executes the validated tool call. |
| `check_result` | deterministic | Inspects tool output for anomalies (e.g. order/customer mismatch). |
| `human_handoff` | deterministic + `interrupt()` | Builds a handoff payload (reason, transcript, partial state) and pauses the graph for a human. |

Guiding principle: **the LLM decides intent, code decides permission.** Every escalation path is a hard branch in the graph, not left to the model's own judgment alone — including a turn-count loop guard that forces a handoff regardless of what the agent wants to do next.

## Quickstart

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd support-agent
uv sync
cp .env.example .env
```

Fill in `.env`:
- `ANTHROPIC_API_KEY` — required, the agent calls a real Claude model
- `LANGSMITH_API_KEY` — required for tracing/eval; tracing is on by default (`LANGSMITH_TRACING=true`)

Start the API:

```bash
uv run uvicorn src.app.main:app --reload
```

Talk to it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "customer_id": "cust_1", "message": "What items are in order ord_1?"}'
```

Response is either the agent's reply, or a handoff notice with `escalated: true` and an `escalation_reason` if the conversation was routed to a human.

## Tests

```bash
uv run pytest
```

Tests hit the real Claude model (no mocked LLM responses) to verify actual graph behavior, not just unit-level logic.

## Tracing and evals

Every graph run is traced to LangSmith under the `support-agent` project (`LANGSMITH_PROJECT` in `.env`) — each node, tool call, guardrail rejection, and handoff shows up as its own step in the trace.

To run the golden eval set:

```bash
uv run python -m app.eval.run_eval
```

This upserts the `support-agent-golden-v1` dataset in LangSmith and runs an experiment against it, scoring each example on whether the response/escalation matches the expected outcome. View results in the LangSmith UI under that dataset's experiments tab.

## Repo layout

```
src/app/
  api/          FastAPI request/response schemas and routes
  graph/        LangGraph nodes, edges, state, and the run() entrypoint
  tools/        Tool definitions (order lookup, ticket creation, escalation)
  data/         In-memory data store (customers, orders, tickets)
  eval/         LangSmith golden dataset and eval runner
  main.py       FastAPI app
tests/          Mirrors src/app/ by module
```

## What this isn't

- No external API integrations yet — tools operate on an in-memory store, shaped like a real API's interface, ready to swap in a live backend
- No UI beyond the HTTP endpoint
- No multi-agent orchestration, memory/vector stores, or RAG
- No auth, deployment, or scaling infrastructure
