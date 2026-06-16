from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from colorama import Fore


def retry_api(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Decorator to retry API calls with exponential backoff."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    retries += 1
                    new_temperature = None
                    if "temperature" in kwargs:
                        kwargs["temperature"] = float(kwargs["temperature"]) + 0.1
                        new_temperature = kwargs.get("temperature")
                    if kwargs.get("think") is not True:
                        kwargs["think"] = False
                    if warn_user:
                        temp_note = f" New temperature={new_temperature:.2f}." if new_temperature is not None else ""
                        logging.getLogger(__name__).warning(
                            f"{Fore.YELLOW}Ollama API error: {exc}. "
                            f"Retrying ({retries}/{max_retries}) Think ({kwargs.get('think')})...{temp_note}{Fore.RESET}"
                        )
                    time.sleep(backoff_base ** retries)
            raise RuntimeError(f"Failed after {max_retries} retries.")

        return wrapper

    return decorator
