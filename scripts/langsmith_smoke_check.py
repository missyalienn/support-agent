"""Throwaway check confirming LangSmith tracing is wired before real agent logic exists.

Run: uv run python scripts/langsmith_smoke_check.py
Then confirm a "smoke_check" run appears in the LangSmith project dashboard.
"""

import structlog
from langsmith import traceable

from app.config import settings
from app.logging_config import configure_logging

logger = structlog.get_logger(__name__)


@traceable(name="smoke_check")
def smoke_check(message: str) -> str:
    return f"traced: {message}"


if __name__ == "__main__":
    configure_logging()
    if not settings.langsmith_tracing:
        raise SystemExit("LANGSMITH_TRACING is not enabled in .env")
    result = smoke_check("hello from support-agent")
    logger.info("smoke_check_complete", result=result, langsmith_project=settings.langsmith_project)
