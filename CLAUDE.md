# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project
`support-agent` — LangGraph-based customer support automation agent with
guardrails, human handoff logic, and LangSmith tracing. Portfolio project.

**Stack:** Python, `uv`, FastAPI, LangGraph, LangChain, LangSmith, Anthropic
SDK, Pydantic.

## Environment
- Use `uv` for all dependency management (`uv add`, `uv remove`, `uv run`).
  Never bare `pip install`.
- Secrets live in `.env` (see `.env.example`). Never commit `.env`.

## Running the Project
- Start the app: `uv run uvicorn src.app.main:app --reload`
- Run tests: `uv run pytest`

## Tracing
Any new LangGraph/LangChain component must be traceable through LangSmith. Don't bypass or reconfigure.

## Logging
- No `print()` statements, anywhere including scratch scripts.
- Structured logging via `structlog`, configured once in
  `src/app/logging_config.py`. Import the logger, don't reconfigure it.

## Testing
Write tests alongside new features, not after. Every new agent node,
guardrail, or tool should ship with a corresponding test in the same
change.

## Design Principles

**Separation of concerns.** Each module has one job. Agent logic, tool
definitions, API routes, and config should live in distinct modules — don't
let a route handler contain graph logic, or a graph node reach into request
parsing.

**DRY, but deliberately.** If the same logic (validation, error formatting,
a prompt template, a retry pattern) shows up in a second place, extract it
to a shared location instead of copy-pasting. Cross-cutting concerns —
logging, error handling, tracing setup, auth — must be standardized in one
place and imported, never re-implemented per-module.

**Avoid over-abstraction.** Don't build a generic framework for a problem
you've only solved once. No abstract base classes, plugin systems, or
config-driven indirection until there are at least 2–3 concrete cases that
actually need the shared shape. Prefer a plain function over a class, and a
class over a framework, unless there's a clear reason not to.

**Avoid tight coupling.** Modules should depend on interfaces/contracts
(e.g. a Pydantic model, a function signature), not on each other's
internals. A change to how the agent graph works shouldn't require touching
the API layer, and vice versa.

**Rule of thumb:** if you're choosing between duplicating a few lines and
adding a new abstraction layer, prefer duplication until the third
repetition — then extract.
