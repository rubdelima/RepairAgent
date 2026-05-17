from __future__ import annotations

from difflib import SequenceMatcher

from autogpt.app.spinner import update_current_spinner

DEFAULT_STREAM_TOKEN_LIMIT = 200
DEFAULT_MIN_UPDATE_INTERVAL_S = 0.05
DEFAULT_REPEAT_THRESHOLD = 4
DEFAULT_REPEAT_SIMILARITY = 0.92
DEFAULT_REPEAT_WINDOW_CHARS = 800
DEFAULT_REPEAT_MIN_CHARS = 60


def truncate_tail_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""

    text = str(text)
    if max_tokens <= 0:
        return ""

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


def should_stream_from_spinner() -> bool:
    try:
        from autogpt.app import spinner as spinner_module

        current = spinner_module.get_current_spinner()
        return current is not None and current.running
    except Exception:
        return False


def update_spinner_stream(thinking: str, content: str, token_limit: int) -> None:
    thinking_tail = truncate_tail_tokens(thinking, token_limit)
    content_tail = truncate_tail_tokens(content, token_limit)
    if content_tail.strip():
        update_current_spinner(f"[cyan]Thinking...[/cyan] [green]{content_tail}[/green]")
    elif thinking_tail.strip():
        update_current_spinner(f"[cyan]Thinking...[/cyan] [yellow dim]{thinking_tail}[/yellow dim]")


def normalize_repeat_window(text: str, window_chars: int) -> str:
    if not text:
        return ""
    snippet = text[-window_chars:]
    return " ".join(snippet.split())


def extract_last_sentence(window: str) -> str:
    if not window:
        return ""
    parts = [p.strip() for p in window.replace("\n", " ").split(".") if p.strip()]
    if not parts:
        return ""
    sentence = parts[-1]
    if len(sentence) < DEFAULT_REPEAT_MIN_CHARS:
        return ""
    return sentence


def _sentence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def count_similar_sentences(window: str, sentence: str, threshold: float) -> int:
    if not window or not sentence:
        return 0
    sentences = [p.strip() for p in window.replace("\n", " ").split(".") if p.strip()]
    if not sentences:
        return 0
    count = 0
    for candidate in sentences:
        if len(candidate) < DEFAULT_REPEAT_MIN_CHARS:
            continue
        if _sentence_similarity(candidate, sentence) >= threshold:
            count += 1
    return count


def summarize_repeat_text(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
