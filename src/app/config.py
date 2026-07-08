from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# langsmith reads LANGSMITH_* vars directly from os.environ, not from our
# Settings object, so .env must be loaded into the actual process environment.
load_dotenv()


class Settings(BaseSettings):
    """Environment-backed config for model provider and LangSmith tracing."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "support-agent"


settings = Settings()
