"""Utility helpers — global exception handling."""
import traceback
from functools import wraps
from typing import Callable

from utils.logger import get_logger

logger = get_logger(__name__)


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator for CLI entry points. Catches unexpected exceptions,
    logs the full traceback, and prints a clean error message.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            print(f"\n❌ {e}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            print(f"\n❌ Configuration error: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            print(f"\n❌ {e}")
        except Exception as e:
            logger.critical(f"Unexpected error: {e}\n{traceback.format_exc()}")
            print(f"\n❌ Unexpected error: {e}")
            print("Full traceback logged. Check the logs/ directory.")
    return wrapper
