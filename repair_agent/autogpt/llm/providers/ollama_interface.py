from __future__ import annotations

import functools
import re
import time
from typing import Any, Callable, List

from colorama import Fore
from autogpt.llm.base import MessageDict
from autogpt.logs import logger


def is_ollama_model(model: str) -> bool:
    """Check if a model name refers to an Ollama model."""
    return model.startswith("ollama-") or model.startswith("ollama:")


def _normalize_model_name(model: str) -> str:
    return re.sub(r"^ollama[-:]", "", model)


def _convert_messages(messages: List[MessageDict]) -> List[dict[str, Any]]:
    converted: List[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        if role == "function":
            role = "tool"
        converted.append({"role": role, "content": msg["content"]})
    return converted


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
                except Exception as e:
                    retries += 1
                    if warn_user:
                        logger.warn(
                            f"{Fore.YELLOW}Ollama API error: {e}. "
                            f"Retrying ({retries}/{max_retries})...{Fore.RESET}"
                        )
                    time.sleep(backoff_base ** retries)
            raise RuntimeError(f"Failed after {max_retries} retries.")

        return wrapper

    return decorator


@retry_api()
def create_chat_completion(
    messages: List[MessageDict],
    *_,
    **kwargs,
) -> dict:
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Ollama library is not installed. Please install it to use Ollama models."
        ) from exc

    model = _normalize_model_name(kwargs.pop("model", "gemma3:4b"))
    temperature = kwargs.pop("temperature", 0)
    max_tokens = kwargs.pop("max_tokens", None)
    options = dict(kwargs.pop("options", {}) or {})
    tools = kwargs.pop("tools", None)
    functions = kwargs.pop("functions", None)
    response_format = kwargs.pop("response_format", None)

    if "temperature" not in options:
        options["temperature"] = temperature
    if max_tokens is not None and "num_predict" not in options:
        options["num_predict"] = max_tokens

    if tools is None and functions is not None:
        tools = functions

    format_schema = None
    if response_format:
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                format_schema = {"type": "object"}
            else:
                format_schema = response_format
        else:
            format_schema = response_format

    chat_kwargs: dict[str, Any] = {
        "model": model,
        "messages": _convert_messages(messages),
        "options": options,
    }
    if tools is not None:
        chat_kwargs["tools"] = tools
    if format_schema is not None:
        chat_kwargs["format"] = format_schema

    response = ollama.chat(**chat_kwargs)

    return _OllamaResponse(response)


class _OllamaResponse:
    """Mimics OpenAI's ChatCompletion response structure for compatibility."""

    def __init__(self, response: dict[str, Any]):
        self.model = response.get("model")
        self.usage = response.get("usage")
        message = response.get("message", {})
        tool_calls = message.get("tool_calls")
        self.choices = [_Choice(message.get("content", ""), tool_calls)]

    def __contains__(self, key):
        return hasattr(self, key)


class _Choice:
    """Mimics OpenAI's Choice object."""

    def __init__(self, content: str, tool_calls: Any = None):
        self.message = {
            "role": "assistant",
            "content": content,
            "function_call": None,
            "tool_calls": tool_calls,
        }
    