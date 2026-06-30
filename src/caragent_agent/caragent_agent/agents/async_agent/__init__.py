"""Async Agent module."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import execution, memory, orchestration, planning, response, runtime
    from .async_agent_graph import create_async_agent
    from .async_agent_interface import AsyncAgent
    from .async_agent_node import (
        AsyncAgentState,
        TaskItem,
        create_background_worker_node,
        create_bg_router,
        create_execute_node,
        create_ingest_node,
        create_orchestrate_node,
        create_plan_node,
        create_route_after_orchestrate,
        route_after_execute,
    )
    from .memory.run_memory import AsyncAgentRunMemory

__all__ = [
    "execution",
    "memory",
    "orchestration",
    "planning",
    "response",
    "runtime",
    "AsyncAgent",
    "AsyncAgentRunMemory",
    "create_async_agent",
    "TaskItem",
    "AsyncAgentState",
    "create_ingest_node",
    "create_orchestrate_node",
    "create_plan_node",
    "create_execute_node",
    "create_background_worker_node",
    "create_route_after_orchestrate",
    "route_after_execute",
    "create_bg_router",
]

_SUBMODULE_EXPORTS = {
    "execution": ".execution",
    "memory": ".memory",
    "orchestration": ".orchestration",
    "planning": ".planning",
    "response": ".response",
    "runtime": ".runtime",
}

_VALUE_EXPORTS = {
    "AsyncAgent": (".async_agent_interface", "AsyncAgent"),
    "AsyncAgentRunMemory": (".memory.run_memory", "AsyncAgentRunMemory"),
    "create_async_agent": (".async_agent_graph", "create_async_agent"),
    "TaskItem": (".async_agent_node", "TaskItem"),
    "AsyncAgentState": (".async_agent_node", "AsyncAgentState"),
    "create_ingest_node": (".async_agent_node", "create_ingest_node"),
    "create_orchestrate_node": (".async_agent_node", "create_orchestrate_node"),
    "create_plan_node": (".async_agent_node", "create_plan_node"),
    "create_execute_node": (".async_agent_node", "create_execute_node"),
    "create_background_worker_node": (
        ".async_agent_node",
        "create_background_worker_node",
    ),
    "create_route_after_orchestrate": (
        ".async_agent_node",
        "create_route_after_orchestrate",
    ),
    "route_after_execute": (".async_agent_node", "route_after_execute"),
    "create_bg_router": (".async_agent_node", "create_bg_router"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose async-agent exports without eagerly importing the whole stack."""

    if name in _SUBMODULE_EXPORTS:
        return import_module(_SUBMODULE_EXPORTS[name], __name__)

    if name in _VALUE_EXPORTS:
        module_name, attr_name = _VALUE_EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
