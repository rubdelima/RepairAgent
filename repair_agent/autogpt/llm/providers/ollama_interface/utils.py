from __future__ import annotations

import json
import re
from typing import Any, List

from autogpt.llm.base import MessageDict


def is_ollama_model(model: str) -> bool:
    """Check if a model name refers to an Ollama model."""
    return model.startswith("ollama-") or model.startswith("ollama:")


def normalize_model_name(model: str) -> str:
    return re.sub(r"^ollama[-:]", "", model)


def convert_messages(messages: List[MessageDict]) -> List[dict[str, Any]]:
    converted: List[dict[str, Any]] = []
    for msg in messages:
        role: str = msg["role"]
        if role == "function":
            role = "tool"
        converted.append({"role": role, "content": msg["content"]})
    return converted


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def normalize_tool_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    if "." in name:
        return name.split(".")[-1]
    return name


def extract_function_call(message: Any) -> tuple[str | None, dict[str, Any]]:
    message_dict = as_dict(message)
    tool_calls = message_dict.get("tool_calls") or []
    if not tool_calls:
        return None, {}

    first = tool_calls[0]
    first_dict = as_dict(first)
    function_payload = first_dict.get("function")
    if function_payload is None:
        function_payload = getattr(first, "function", None)

    function_dict = as_dict(function_payload)
    name = function_dict.get("name") or getattr(function_payload, "name", None)
    args = function_dict.get("arguments") or getattr(function_payload, "arguments", None) or {}

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}

    normalized_name = normalize_tool_name(name) if name else None
    return normalized_name, args


def coerce_langchain_role(role: str) -> str:
    """Best-effort mapping from LangChain message types to chat roles."""
    mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    return mapping.get(role, role)


def langchain_messages_to_message_dicts(messages: list[Any]) -> list[MessageDict]:
    """Convert LangChain-like messages into RepairAgent's MessageDict format."""
    converted: list[MessageDict] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type") or "user"
            content = msg.get("content") or ""
            converted.append({"role": str(role), "content": str(content)})
            continue

        role = getattr(msg, "type", None) or "user"
        role = coerce_langchain_role(str(role))
        content = getattr(msg, "content", "")
        converted.append({"role": role, "content": str(content)})

    return converted
