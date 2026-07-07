"""Project-wide logger with file persistence and correlation ID tracking."""
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

# Correlation ID for tracing a single pipeline run across all modules
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="no-correlation")

LOGS_DIR = Path("logs")


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current run."""
    cid = str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


class CorrelationFilter(logging.Filter):
    """Injects correlation_id into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


def setup_logging(log_level: str = "INFO") -> Path:
    """
    Configure root rag_scholar logger with console + file handlers.
    Call once at application startup.

    Returns:
        Path to the created log file
    """
    LOGS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOGS_DIR / f"{timestamp}.log"

    root_logger = logging.getLogger("rag_scholar")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers (prevents duplicates on re-init)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    correlation_filter = CorrelationFilter()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(correlation_filter)
    root_logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)
    root_logger.addHandler(file_handler)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the rag_scholar namespace."""
    return logging.getLogger(f"rag_scholar.{name}")
