import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Populates os.environ from .env — required not just for our own config below,
# but because langsmith reads LANGSMITH_* vars directly from os.environ itself.
load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str


settings = Settings(
    anthropic_api_key=_require("ANTHROPIC_API_KEY"),
    langsmith_tracing=os.environ.get("LANGSMITH_TRACING", "false").lower() == "true",
    langsmith_api_key=os.environ.get("LANGSMITH_API_KEY"),
    langsmith_project=os.environ.get("LANGSMITH_PROJECT", "support-agent"),
)
