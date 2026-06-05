"""Agent package exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_agent import AsyncAgent, create_async_agent

__all__ = ["AsyncAgent", "create_async_agent"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavy async-agent entry points."""

    if name in {"AsyncAgent", "create_async_agent"}:
        from .async_agent import AsyncAgent, create_async_agent

        exports = {
            "AsyncAgent": AsyncAgent,
            "create_async_agent": create_async_agent,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
