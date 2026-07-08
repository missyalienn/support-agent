"""Throwaway check confirming LangSmith tracing is wired before real agent logic exists.

Run: uv run python scripts/langsmith_smoke_check.py
Then confirm a "smoke_check" run appears in the LangSmith project dashboard.
"""

from langsmith import traceable

from app.config import settings


@traceable(name="smoke_check")
def smoke_check(message: str) -> str:
    return f"traced: {message}"


if __name__ == "__main__":
    if not settings.langsmith_tracing:
        raise SystemExit("LANGSMITH_TRACING is not enabled in .env")
    result = smoke_check("hello from support-agent")
    print(result)
    print(f"Check the '{settings.langsmith_project}' project in LangSmith for this run.")
