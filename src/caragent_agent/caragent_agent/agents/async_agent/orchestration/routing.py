"""Routing helpers for async-agent LangGraph transitions."""

from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END

from caragent_agent.agents.async_agent.runtime.types import AsyncAgentState


def create_route_after_orchestrate(logger: Optional[Any] = None) -> Callable[[AsyncAgentState], str]:
    """Create routing function after orchestrate node."""

    def route_after_orchestrate(state: AsyncAgentState) -> str:
        """Route from orchestrate into planning, execution, or graph termination."""

        next_action = state.get("next_action")
        next_action_type = (
            next_action.get("type")
            if isinstance(next_action, dict)
            else None
        )

        if next_action_type == "plan":
            if logger:
                logger.log_foreground("Route After Orchestrate: next_action=plan")
            return "plan"

        if next_action_type == "execute":
            if logger:
                logger.log_foreground("Route After Orchestrate: next_action=execute")
            return "execute"

        if logger:
            logger.log_foreground(
                f"Route After Orchestrate: next_action={next_action_type or 'missing'}, route=END"
            )
        return END

    return route_after_orchestrate


def route_after_execute(state: AsyncAgentState) -> str:
    """Route after execute node - always return to orchestrate for centralized control."""

    return "orchestrate"
