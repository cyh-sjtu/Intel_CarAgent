"""Foreground tool runtime context for one execute pass."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_runtime_tool_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "caragent_runtime_tool_context",
    default=None,
)


def get_runtime_tool_context() -> dict[str, Any]:
    """Return the current foreground tool context, if any."""

    context = _runtime_tool_context.get()
    return dict(context) if isinstance(context, dict) else {}


@contextmanager
def runtime_tool_context(context: dict[str, Any]) -> Iterator[None]:
    """Expose one execute-pass context to tools invoked by the ReAct loop."""

    token = _runtime_tool_context.set(dict(context or {}))
    try:
        yield
    finally:
        _runtime_tool_context.reset(token)
