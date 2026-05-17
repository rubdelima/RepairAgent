from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List

from colorama import Fore

from autogpt.llm.base import MessageDict

from .retry import retry_api
from .stream import (
    DEFAULT_MIN_UPDATE_INTERVAL_S,
    DEFAULT_REPEAT_SIMILARITY,
    DEFAULT_REPEAT_THRESHOLD,
    DEFAULT_REPEAT_WINDOW_CHARS,
    DEFAULT_STREAM_TOKEN_LIMIT,
    count_similar_sentences,
    extract_last_sentence,
    normalize_repeat_window,
    summarize_repeat_text,
    should_stream_from_spinner,
    update_spinner_stream,
)
from .utils import (
    as_dict,
    convert_messages,
    extract_function_call,
    langchain_messages_to_message_dicts,
    normalize_model_name,
)


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

    model = normalize_model_name(kwargs.pop("model", "gemma3:4b"))
    temperature = kwargs.pop("temperature", 0)
    max_tokens = kwargs.pop("max_tokens", None)
    options = dict(kwargs.pop("options", {}) or {})
    tools = kwargs.pop("tools", None)
    functions = kwargs.pop("functions", None)
    response_format = kwargs.pop("response_format", None)
    think = kwargs.pop("think", None)

    stream = bool(kwargs.pop("stream", False) or should_stream_from_spinner())
    detect_repetition = bool(
        kwargs.pop("detect_repetition", False)
        or os.environ.get("REPAIRAGENT_OLLAMA_REPEAT_DETECT", "0").strip() in {"1", "true", "True"}
    )
    stream_timeout_s = float(os.environ.get("REPAIRAGENT_OLLAMA_STREAM_TIMEOUT_S", "180"))

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
        "messages": convert_messages(messages),
        "options": options,
    }
    chat_kwargs["stream"] = stream
    if think is not None:
        chat_kwargs["think"] = think
    if tools is not None:
        chat_kwargs["tools"] = tools
    if format_schema is not None:
        chat_kwargs["format"] = format_schema

    if not stream:
        response = ollama.chat(**chat_kwargs)
        return _OllamaResponse(response)

    full_content_parts: list[str] = []
    full_thinking_parts: list[str] = []
    last_tool_calls: Any = None
    last_model: str | None = None
    last_usage: Any = None

    last_update = 0.0
    repeat_count = 0

    start_time = time.monotonic()
    stream_iter = ollama.chat(**chat_kwargs)
    for chunk in stream_iter:
        if time.monotonic() - start_time > stream_timeout_s:
            raise TimeoutError(
                f"Ollama stream exceeded {stream_timeout_s:.0f}s without final response."
            )
        chunk_dict = as_dict(chunk)
        if chunk_dict.get("model"):
            last_model = chunk_dict.get("model")
        if chunk_dict.get("usage"):
            last_usage = chunk_dict.get("usage")

        message = chunk_dict.get("message")
        message_dict = as_dict(message)
        delta_content = message_dict.get("content") or getattr(message, "content", "") or ""
        delta_thinking = message_dict.get("thinking") or getattr(message, "thinking", "") or ""
        delta_tool_calls = message_dict.get("tool_calls") or getattr(message, "tool_calls", None)
        if delta_tool_calls is not None:
            last_tool_calls = delta_tool_calls

        if delta_content:
            full_content_parts.append(str(delta_content))
        if delta_thinking:
            full_thinking_parts.append(str(delta_thinking))

        if detect_repetition:
            combined_text = "".join(full_thinking_parts) + "\n" + "".join(full_content_parts)
            current_window = normalize_repeat_window(combined_text, DEFAULT_REPEAT_WINDOW_CHARS)
            repeated_sentence = extract_last_sentence(current_window)
            repeat_hits = count_similar_sentences(
                current_window,
                repeated_sentence,
                DEFAULT_REPEAT_SIMILARITY,
            )
            if repeat_hits >= DEFAULT_REPEAT_THRESHOLD:
                repeat_count += 1
            else:
                repeat_count = 0

            if repeat_count >= 1:
                logging.getLogger(__name__).warning(
                    f"{Fore.YELLOW}Detected repeated model output. "
                    f"Count={repeat_hits} Similarity>={DEFAULT_REPEAT_SIMILARITY:.2f} "
                    f"Snippet='{summarize_repeat_text(repeated_sentence)}'. "
                    f"Retrying with higher temperature...{Fore.RESET}"
                )
                raise RuntimeError("Repeated model output detected.")

        now = time.monotonic()
        if now - last_update >= DEFAULT_MIN_UPDATE_INTERVAL_S:
            update_spinner_stream(
                "".join(full_thinking_parts),
                "".join(full_content_parts),
                token_limit=DEFAULT_STREAM_TOKEN_LIMIT,
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
        response_dict = as_dict(response)
        self.model = response_dict.get("model")
        self.usage = response_dict.get("usage")

        message = response_dict.get("message")
        message_dict = as_dict(message)
        content = message_dict.get("content") or getattr(message, "content", "") or ""
        thinking = message_dict.get("thinking") or getattr(message, "thinking", "") or ""
        tool_calls = message_dict.get("tool_calls") or getattr(message, "tool_calls", None)

        function_name, function_args = extract_function_call(message)
        function_call = None
        if function_name:
            function_call = {
                "name": function_name,
                "arguments": json.dumps(function_args),
            }

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


class OllamaChatWrapper:
    """Minimal wrapper with LangChain-like `invoke()`."""

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
