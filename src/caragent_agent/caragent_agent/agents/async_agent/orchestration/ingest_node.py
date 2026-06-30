"""Ingest node for converting user messages into structured workflow events."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage

from caragent_agent.agents.async_agent.execution.support import clear_turn_response
from caragent_agent.agents.async_agent.orchestration.node_common import (
    _record_run_memory_event,
    _strip_ignored_state_fields,
)
from caragent_agent.agents.async_agent.orchestration.runtime import (
    build_user_input_received_event,
    get_message_id,
)
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.runtime.types import AsyncAgentState


def create_ingest_node(
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
):
    """Convert user-language messages into structured events before orchestration."""

    def ingest_node(state: AsyncAgentState) -> AsyncAgentState:
        """Ingest the newest user message and synthesize missing user-input events."""

        state = _strip_ignored_state_fields(state)
        messages = state.get("messages", [])
        user_inputs = list(state.get("user_inputs", []))
        events = list(state.get("events", []))
        state_changed = False

        last_message = messages[-1] if messages else None

        if isinstance(last_message, HumanMessage):
            user_message_id = get_message_id(
                last_message,
                fallback_prefix="user_message",
                message_history=messages,
            )
            already_ingested = any(
                item.get("message_id") == user_message_id
                for item in user_inputs
            )
            if already_ingested:
                return state
            
            user_input_event, updated_user_inputs = build_user_input_received_event(
                last_message,
                messages=messages,
                user_inputs=user_inputs,
            )
      
            has_existing_user_input_event = any(
                event.get("type") == "user_input_received"
                and event.get("message_id") == user_message_id
                for event in events
            )
            user_inputs = updated_user_inputs
            if not has_existing_user_input_event:
                events.append(user_input_event)
                state_changed = True
                if updated_user_inputs:
                    try:
                        run_memory.record_user_input(updated_user_inputs[-1])
                    except Exception:
                        pass
                _record_run_memory_event(
                    run_memory,
                    user_input_event,
                    stage="ingest",
                )
                if logger:
                    logger.log_foreground(
                        "Ingest Debug: synthesized user_input_received event for message_id={message_id}, user_input_id={user_input_id}".format(
                            message_id=user_message_id,
                            user_input_id=user_input_event.get("user_input_id"),
                        )
                    )
                    print(
                        f"{Colors.ORCHESTRATE}Ingest Debug:{Colors.RESET} "
                        f"synthesized user_input_received for user_input_id={user_input_event.get('user_input_id')}"
                    )

        if (
            not state_changed
            and state.get("user_inputs") == user_inputs
        ):
            return state

        return clear_turn_response(
            {
                **state,
                "events": events,
                "user_inputs": user_inputs,
            }
        )

    return ingest_node
