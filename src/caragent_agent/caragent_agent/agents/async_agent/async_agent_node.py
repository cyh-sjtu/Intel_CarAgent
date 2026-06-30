"""Public entrypoints for async-agent graph node factories."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from caragent_agent.agents.async_agent.execution import background as _background_module
from caragent_agent.agents.async_agent.execution.background import (
    _extract_recommendation_from_background_output,
    _foreground_has_claimed_task,
    background_selection_blocked_by_unresolved_decision,
    claim_next_background_task,
    create_bg_router,
    select_background_target_task,
)
from caragent_agent.agents.async_agent.execution.execute_node import (
    create_execute_node as _create_execute_node,
)
from caragent_agent.agents.async_agent.execution import execute_node as _execute_node_module
from caragent_agent.agents.async_agent.orchestration.ingest_node import create_ingest_node
from caragent_agent.agents.async_agent.orchestration.orchestrate_node import (
    create_orchestrate_node as _create_orchestrate_node,
)
from caragent_agent.agents.async_agent.orchestration import orchestrate_node as _orchestrate_node_module
from caragent_agent.agents.async_agent.orchestration.routing import (
    create_route_after_orchestrate,
    route_after_execute,
)
from caragent_agent.agents.async_agent.planning.helpers import build_decision_context
from caragent_agent.agents.async_agent.planning.plan_node import create_plan_node
from caragent_agent.agents.async_agent.planning.prompting import (
    AGENT_PROMPTS,
    classify_planning_requirement,
    is_require_planning,
)
from caragent_agent.agents.async_agent.runtime.control import set_background_enabled
from caragent_agent.agents.async_agent.runtime.types import AsyncAgentState, TaskItem
from caragent_agent.third_party.from_langgraph.react_agent import create_react_agent


def create_orchestrate_node(*args: Any, **kwargs: Any):
    """Create the orchestrate node."""

    _orchestrate_node_module.is_require_planning = is_require_planning
    return _create_orchestrate_node(*args, **kwargs)


def create_execute_node(*args: Any, **kwargs: Any):
    """Create the execute node."""

    _execute_node_module.create_react_agent = create_react_agent
    return _create_execute_node(*args, **kwargs)


def create_background_worker_node(*args: Any, **kwargs: Any):
    """Create a background worker."""

    kwargs.setdefault("react_agent_factory", create_react_agent)
    return _background_module.create_background_worker_node(*args, **kwargs)


__all__ = [
    "AGENT_PROMPTS",
    "AsyncAgentState",
    "END",
    "TaskItem",
    "_extract_recommendation_from_background_output",
    "_foreground_has_claimed_task",
    "background_selection_blocked_by_unresolved_decision",
    "build_decision_context",
    "claim_next_background_task",
    "classify_planning_requirement",
    "create_background_worker_node",
    "create_bg_router",
    "create_execute_node",
    "create_ingest_node",
    "create_orchestrate_node",
    "create_plan_node",
    "create_react_agent",
    "create_route_after_orchestrate",
    "is_require_planning",
    "route_after_execute",
    "select_background_target_task",
    "set_background_enabled",
]
