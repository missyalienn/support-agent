import structlog

from app.logging_config import configure_logging

logger = structlog.get_logger(__name__)


def main() -> None:
    configure_logging()
    logger.info("startup", message="Hello from benchdog-api-starter!")
