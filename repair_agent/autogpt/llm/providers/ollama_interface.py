from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import signal
import threading
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
        role: str = msg["role"]
        if role == "function":
            role = "tool"
        converted.append({"role": role, "content": msg["content"]})
    return converted


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _normalize_tool_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    if "." in name:
        return name.split(".")[-1]
    return name


def _extract_function_call(message: Any) -> tuple[str | None, dict[str, Any]]:
    message_dict = _as_dict(message)
    tool_calls = message_dict.get("tool_calls") or []
    if not tool_calls:
        return None, {}

    first = tool_calls[0]
    first_dict = _as_dict(first)
    function_payload = first_dict.get("function")
    if function_payload is None:
        function_payload = getattr(first, "function", None)

    function_dict = _as_dict(function_payload)
    name = function_dict.get("name") or getattr(function_payload, "name", None)
    args = function_dict.get("arguments") or getattr(function_payload, "arguments", None) or {}

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}

    normalized_name = _normalize_tool_name(name) if name else None
    return normalized_name, args


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
                    # Increase temperature slightly on retry to escape repetition loops.
                    if "temperature" in kwargs:
                        try:
                            kwargs["temperature"] = float(kwargs["temperature"]) + 0.1
                        except Exception:
                            pass
                    # Apply repetition penalties on retry if they were not provided.
                    try:
                        current_freq = kwargs.get("frequency_penalty")
                        current_pres = kwargs.get("presence_penalty")
                        if current_freq is None:
                            kwargs["frequency_penalty"] = 0.2 + (0.05 * retries)
                        if current_pres is None:
                            kwargs["presence_penalty"] = 0.2 + (0.05 * retries)
                    except Exception:
                        pass
                    if warn_user:
                        logging.getLogger(__name__).warning(
                            f"{Fore.YELLOW}Ollama API error: {e}. "
                            f"Retrying ({retries}/{max_retries})...{Fore.RESET}"
                        )
                    time.sleep(backoff_base ** retries)
            raise RuntimeError(f"Failed after {max_retries} retries.")

        return wrapper

    return decorator


def _truncate_tail_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""

    text = str(text)
    if max_tokens <= 0:
        return ""

    # Best-effort tokenization: prefer tiktoken when available, fall back to words.
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return enc.decode(token_ids[-max_tokens:])
    except Exception:
        parts = text.split()
        if len(parts) <= max_tokens:
            return text
        return " ".join(parts[-max_tokens:])


def _should_stream_from_spinner() -> bool:
    """Enable streaming automatically when a rich spinner is active."""
    try:
        from autogpt.app import spinner as spinner_module

        current = spinner_module.get_current_spinner()
        return current is not None and current.running
    except Exception:
        return False


def _update_spinner_stream(thinking: str, content: str, token_limit: int = 200) -> None:
    try:
        from autogpt.app.spinner import update_current_spinner

        thinking_tail = _truncate_tail_tokens(thinking, token_limit)
        content_tail = _truncate_tail_tokens(content, token_limit)
        if content_tail.strip():
            update_current_spinner(f"[cyan]Thinking...[/cyan] [green]{content_tail}[/green]")
        elif thinking_tail.strip():
            update_current_spinner(f"[cyan]Thinking...[/cyan] [yellow dim]{thinking_tail}[/yellow dim]")
    except Exception:
        return


class _StallWatchdog:
    """Timeout watchdog that triggers if we don't receive stream chunks.

    Implemented via SIGALRM/setitimer, which works on Linux when running on the
    main thread. This allows us to break out of a stuck streaming read and let
    the retry wrapper re-attempt.
    """

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = float(timeout_s)
        self._enabled = False
        self._previous_handler: Any = None

    def _usable(self) -> bool:
        if self._timeout_s <= 0:
            return False
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
            return False
        return threading.current_thread() is threading.main_thread()

    def __enter__(self) -> "_StallWatchdog":
        if not self._usable():
            return self

        def _handler(signum: int, frame: Any) -> None:  # pragma: no cover
            raise TimeoutError(
                f"Ollama stream stalled (no output for {self._timeout_s:.0f}s)."
            )

        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self._timeout_s)
        self._enabled = True
        return self

    def kick(self) -> None:
        if not self._enabled:
            return
        signal.setitimer(signal.ITIMER_REAL, self._timeout_s)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if not self._enabled:
            return
        with contextlib.suppress(Exception):
            signal.setitimer(signal.ITIMER_REAL, 0)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGALRM, self._previous_handler)
        self._enabled = False


@retry_api()
def create_chat_completion(
    messages: List[MessageDict],
    *_,
    **kwargs,
) -> "_OllamaResponse":
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
    frequency_penalty = kwargs.pop("frequency_penalty", None)
    presence_penalty = kwargs.pop("presence_penalty", None)

    # Streaming behavior: when the app is showing a spinner, stream tokens to it.
    stream = bool(kwargs.pop("stream", False) or _should_stream_from_spinner())
    stream_token_limit = int(os.environ.get("REPAIRAGENT_OLLAMA_STREAM_TOKEN_LIMIT", "200"))
    stall_timeout_s = float(os.environ.get("REPAIRAGENT_OLLAMA_STREAM_STALL_TIMEOUT", "60"))

    if "temperature" not in options:
        options["temperature"] = temperature
    if max_tokens is not None and "num_predict" not in options:
        options["num_predict"] = max_tokens
    if frequency_penalty is not None:
        options["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        options["presence_penalty"] = presence_penalty

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
    chat_kwargs["stream"] = stream
    if tools is not None:
        chat_kwargs["tools"] = tools
    if format_schema is not None:
        chat_kwargs["format"] = format_schema

    if not stream:
        response = ollama.chat(**chat_kwargs)
        return _OllamaResponse(response)

    # Streaming path: accumulate the final response while updating the spinner.
    full_content_parts: list[str] = []
    full_thinking_parts: list[str] = []
    last_tool_calls: Any = None
    last_model: str | None = None
    last_usage: Any = None

    last_update = 0.0
    min_update_interval_s = float(
        os.environ.get("REPAIRAGENT_OLLAMA_STREAM_MIN_UPDATE_INTERVAL", "0.05")
    )
    with _StallWatchdog(stall_timeout_s) as watchdog:
        stream_iter = ollama.chat(**chat_kwargs)
        for chunk in stream_iter:
            watchdog.kick()
            chunk_dict = _as_dict(chunk)
            if chunk_dict.get("model"):
                last_model = chunk_dict.get("model")
            if chunk_dict.get("usage"):
                last_usage = chunk_dict.get("usage")

            message = chunk_dict.get("message")
            message_dict = _as_dict(message)
            delta_content = message_dict.get("content") or getattr(message, "content", "") or ""
            delta_thinking = message_dict.get("thinking") or getattr(message, "thinking", "") or ""
            delta_tool_calls = message_dict.get("tool_calls") or getattr(message, "tool_calls", None)
            if delta_tool_calls is not None:
                last_tool_calls = delta_tool_calls

            if delta_content:
                full_content_parts.append(str(delta_content))
            if delta_thinking:
                full_thinking_parts.append(str(delta_thinking))

            now = time.monotonic()
            if now - last_update >= min_update_interval_s:
                _update_spinner_stream(
                    "".join(full_thinking_parts),
                    "".join(full_content_parts),
                    token_limit=stream_token_limit,
                )
                last_update = now

    final_response = {
        "model": last_model or model,
        "usage": last_usage,
        "message": {
            "role": "assistant",
            "content": "".join(full_content_parts),
            "thinking": "".join(full_thinking_parts),
            "tool_calls": last_tool_calls,
        },
    }
    return _OllamaResponse(final_response)


class _OllamaResponse:
    """Mimics OpenAI's ChatCompletion response structure for compatibility."""

    def __init__(self, response: dict[str, Any]):
        response_dict = _as_dict(response)
        self.model = response_dict.get("model")
        self.usage = response_dict.get("usage")

        message = response_dict.get("message")
        message_dict = _as_dict(message)
        content = message_dict.get("content") or getattr(message, "content", "") or ""
        thinking = message_dict.get("thinking") or getattr(message, "thinking", "") or ""
        tool_calls = message_dict.get("tool_calls") or getattr(message, "tool_calls", None)

        function_name, function_args = _extract_function_call(message)
        function_call = None
        if function_name:
            function_call = {
                "name": function_name,
                "arguments": json.dumps(function_args),
            }

        # RepairAgent expects JSON in content. For tool-calling outputs where
        # content is empty, synthesize a compatible JSON payload.
        if not content and function_call:
            thoughts = thinking.strip() if isinstance(thinking, str) and thinking.strip() else "Command selected via Ollama tool call."
            content = json.dumps(
                {
                    "thoughts": thoughts,
                    "command": {
                        "name": function_call["name"],
                        "args": function_args,
                    },
                }
            )

        if not content:
            logging.getLogger(__name__).warning(
                f"{Fore.YELLOW}Ollama returned empty content. Full response: {response}{Fore.RESET}"
            )

        self.content = content
        self.choices = [_Choice(content, tool_calls, function_call)]

    def __contains__(self, key):
        return hasattr(self, key)


class _Choice:
    """Mimics OpenAI's Choice object."""

    def __init__(self, content: str, tool_calls: Any = None, function_call: dict[str, str] | None = None):
        self.message = {
            "role": "assistant",
            "content": content,
            "function_call": function_call,
            "tool_calls": tool_calls,
        }


def _coerce_langchain_role(role: str) -> str:
    """Best-effort mapping from LangChain message types to chat roles."""
    mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    return mapping.get(role, role)


def langchain_messages_to_message_dicts(messages: list[Any]) -> list[MessageDict]:
    """Convert LangChain-like messages into RepairAgent's MessageDict format.

    This avoids importing LangChain in callers. We only rely on common attributes
    like `type` and `content`, and also accept dicts with role/content.
    """
    converted: list[MessageDict] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type") or "user"
            content = msg.get("content") or ""
            converted.append({"role": str(role), "content": str(content)})
            continue

        role = getattr(msg, "type", None) or "user"
        role = _coerce_langchain_role(str(role))
        content = getattr(msg, "content", "")
        converted.append({"role": role, "content": str(content)})

    return converted


class OllamaChatWrapper:
    """Minimal wrapper with LangChain-like `invoke()`.

    Returns an `_OllamaResponse` that provides `.content` and `.choices`.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list[Any]) -> _OllamaResponse:
        message_dicts = langchain_messages_to_message_dicts(messages)
        return create_chat_completion(
            message_dicts,
            model=self.model,
            temperature=self.temperature,
        )
    