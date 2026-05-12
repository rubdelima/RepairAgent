"""A spinner module using rich.

This module also exposes a best-effort "current spinner" so other parts of the
app (e.g. LLM providers) can surface progress while the agent is "Thinking...".
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console


_CURRENT_SPINNER: Optional["Spinner"] = None


def get_current_spinner() -> Optional["Spinner"]:
    return _CURRENT_SPINNER


def _set_current_spinner(spinner: Optional["Spinner"]) -> None:
    global _CURRENT_SPINNER
    _CURRENT_SPINNER = spinner


def update_current_spinner(message: str) -> None:
    spinner = _CURRENT_SPINNER
    if spinner is None or not spinner.running:
        return
    spinner.update(message)


class Spinner:
    """A spinner class using rich's Status for animated feedback."""

    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        plain_output: bool = False,
    ) -> None:
        self.plain_output = plain_output
        self.message = message
        self.running = False
        self._console = Console(highlight=False, markup=True)
        self._status = None

    def start(self):
        self.running = True
        _set_current_spinner(self)
        if self.plain_output:
            self._console.print(self.message)
            return
        self._status = self._console.status(self.message, spinner="dots")
        self._status.start()

    def stop(self):
        self.running = False
        if get_current_spinner() is self:
            _set_current_spinner(None)
        if self._status is not None:
            self._status.stop()
            self._status = None

    def update(self, message: str) -> None:
        """Update the spinner message while it's running."""
        self.message = message
        if self.plain_output:
            return
        if self._status is None:
            return
        self._status.update(message)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.stop()
