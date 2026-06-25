"""Assemble the LangGraph-based async agent with orchestrate/plan/execute/background nodes."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from caragent_agent.agents.async_agent.async_agent_node import (
    AsyncAgentState,
    TaskItem,
    create_background_worker_node,
    create_execute_node,
    create_ingest_node,
    create_orchestrate_node,
    create_plan_node,
    create_route_after_orchestrate,
    create_bg_router
)
from caragent_agent.agents.async_agent.runtime.control import build_runtime_control
from caragent_agent.config.config import config


def _build_background_worker_nodes(
    *,
    num_background_workers: int,
    background_tools: Optional[Sequence[BaseTool]],
    background_llm: Optional[Any],
    shared_background_results: dict[int, Any],
    shared_processing_tasks: set[str],
    shared_runtime_control: dict[str, Any],
    logger: Optional[Any],
    run_memory: Optional[Any],
) -> list[tuple[str, Any]]:
    """Create background worker nodes when background analysis is enabled."""

    if num_background_workers <= 0 or not background_tools or background_llm is None:
        return []
    background_nodes: list[tuple[str, Any]] = []
    for worker_id in range(num_background_workers):
        bg_node = create_background_worker_node(
            worker_id,
            num_background_workers,
            background_llm,
            background_tools,
            shared_background_results,
            shared_processing_tasks,
            shared_runtime_control,
            logger=logger,
            run_memory=run_memory,
        )
        background_nodes.append((f"bg_worker_{worker_id}", bg_node))
    return background_nodes

def create_async_agent(
    tools: Union[Sequence[BaseTool], ToolNode],
    *,
    orchestrate_llm: Optional[Any] = None,
    planner_llm: Optional[Any] = None,
    executor_llm: Optional[Any] = None,
    background_llm: Optional[Any] = None,
    background_tools: Optional[Sequence[BaseTool]] = None,
    num_background_workers: int = 0,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    name: Optional[str] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
) -> CompiledStateGraph:
    """Wire up the async agent graph and return the compiled workflow.

    Each agent role receives its own LLM instance so that model selection is
    controlled by declarative config (``llm_routing``) rather than hard-coded.
    """

    if orchestrate_llm is None:
        raise ValueError("orchestrate_llm is required")
    if planner_llm is None:
        raise ValueError("planner_llm is required")
    if executor_llm is None:
        raise ValueError("executor_llm is required")

    if isinstance(tools, ToolNode):
        tool_node = tools
        tool_classes = list(tools.tools_by_name.values())
    else:
        tool_node = ToolNode(tools)
        tool_classes = [t for t in tools if isinstance(t, BaseTool)]

    # Shared caches used across nodes to coordinate background processing
    shared_background_results: dict[int, Any] = {}
    shared_processing_tasks: set[str] = set()
    shared_runtime_control = build_runtime_control(config)

    ingest_node = create_ingest_node(logger=logger, run_memory=run_memory)
    orchestrate_node = create_orchestrate_node(
        orchestrate_llm,
        shared_background_results,
        shared_processing_tasks,
        shared_runtime_control,
        logger=logger,
        run_memory=run_memory,
    )
    plan_node = create_plan_node(
        planner_llm,
        shared_runtime_control,
        shared_background_results,
        shared_processing_tasks,
        logger=logger,
        run_memory=run_memory,
    )
    execute_node = create_execute_node(
        executor_llm,
        tool_classes,
        tool_node,
        shared_background_results,
        shared_runtime_control,
        logger=logger,
        run_memory=run_memory,
    )

    background_nodes = _build_background_worker_nodes(
        num_background_workers=num_background_workers,
        background_tools=background_tools,
        background_llm=background_llm,
        shared_background_results=shared_background_results,
        shared_processing_tasks=shared_processing_tasks,
        shared_runtime_control=shared_runtime_control,
        logger=logger,
        run_memory=run_memory,
    )

    workflow = StateGraph(AsyncAgentState)

    workflow.add_node("ingest", ingest_node)
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)

    for node_name, node_func in background_nodes:
        workflow.add_node(node_name, node_func)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "orchestrate")

    workflow.add_conditional_edges(
        "orchestrate",
        create_route_after_orchestrate(logger),
        {
            "plan": "plan",
            "execute": "execute",
            END: END,
        },
    )

    if background_nodes:
        workflow.add_edge("plan", "orchestrate")
        for node_name, _ in background_nodes:
            workflow.add_edge("plan", node_name)
    else:
        workflow.add_edge("plan", "orchestrate")

    for node_name, _ in background_nodes:
        bg_router = create_bg_router(
            node_name,
            num_background_workers,
            shared_background_results,
            shared_processing_tasks,
            shared_runtime_control,
        )
        workflow.add_conditional_edges(
            node_name,
            bg_router,
            {
                node_name: node_name,
                END: END,
            },
        )

    workflow.add_edge("execute", "orchestrate")
    for node_name, _ in background_nodes:
        workflow.add_edge("execute", node_name)

    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        name=name or "AsynTaskPlanReactAgent",
    )


__all__ = [
    "create_async_agent",
    "AsyncAgentState",
    "TaskItem",
]
