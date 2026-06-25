"""Offline audit helper for semantic object background preanalysis flow."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from import_stubs import install_offline_import_stubs

install_offline_import_stubs()

from caragent_agent.agents.async_agent.execution.background import (
    BackgroundForegroundCoordinator,
    BackgroundResultStore,
    run_background_analysis,
    select_background_target_task,
)
from caragent_agent.agents.async_agent.execution.context import (
    background_result_is_reusable_for_task,
    build_background_reference,
)
from caragent_agent.agents.async_agent.execution.execute_node import (
    _try_complete_semantic_grounding_from_background,
)
from caragent_agent.agents.async_agent.runtime.control import (
    build_runtime_control,
    record_foreground_task,
)


PLAN_ID = "audit_plan"


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log_background(self, message: Any) -> None:
        self.lines.append(str(message))


class _SuccessPreanalysisTool:
    name = "preanalyze_object_on_keyframe"
    tags = ("object_preanalysis", "background_safe")

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "summary": "fake success",
            "data": {
                "destination": {
                    "type": "position",
                    "position": [1.2, 3.4, 0.0],
                    "yaw_deg": 12.0,
                },
                "status": "ok",
                "paths": {"summary_json": "/tmp/fake_success_summary.json"},
                "approach": {"status": "ok", "mode": "fake"},
            },
        }


class _FailPreanalysisTool:
    name = "preanalyze_object_on_keyframe"
    tags = ("object_preanalysis", "background_safe")

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "partial",
            "summary": "right image missing",
            "data": {
                "keyframe_id": int(args.get("keyframe_id") or -1),
                "summary_json": "/tmp/fake_failure_summary.json",
            },
            "error": {
                "code": "keyframe_right_image_missing",
                "message": "right image missing",
            },
        }


def _staging_raw_output(keyframe_id: int) -> str:
    payload = {
        "status": "ok",
        "data": {"destination": {"type": "keyframe", "keyframe_id": keyframe_id}},
    }
    return json.dumps(
        {
            "tool_results": [
                {
                    "name": "search_requirement_on_keyframe_nodes",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
            "final_ai_content": json.dumps(
                {"destination": {"type": "keyframe", "keyframe_id": keyframe_id}},
                ensure_ascii=False,
            ),
        },
        ensure_ascii=False,
    )


def _staging_raw_output_final_only(keyframe_id: int) -> str:
    """Mirror real logs: tools provide candidates, final answer provides destination."""

    tool_payload = {
        "status": "ok",
        "summary": "Retrieved keyframe-node metadata from scene memory.",
        "data": {
            "nodes": {
                str(keyframe_id): {
                    "kf_id": keyframe_id,
                    "position": [3.6, -0.1, 0.0],
                    "semantics": "Elevator entrance with fire extinguisher box visible.",
                }
            }
        },
    }
    return json.dumps(
        {
            "tool_results": [
                {
                    "name": "get_keyframe_nodes_info",
                    "content": json.dumps(tool_payload, ensure_ascii=False),
                }
            ],
            "final_ai_content": (
                "I have a clear winner.\n\n"
                "```json\n"
                + json.dumps(
                    {
                        "destination": {
                            "type": "keyframe",
                            "keyframe_id": keyframe_id,
                        },
                        "current_place_context": "Staging keyframe selected.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n```"
            ),
        },
        ensure_ascii=False,
    )


def _navigation_raw_output_target_keyframe(keyframe_id: int) -> str:
    """Mirror navigation tool output: destination is exposed as target_keyframe_id."""

    payload = {
        "status": "ok",
        "summary": f"Dispatched navigation to keyframe {keyframe_id}.",
        "data": {
            "target_keyframe_id": keyframe_id,
            "target_position": [2.9, -2.6, 0.0],
            "navigation_status": "sim_navigating",
        },
    }
    return json.dumps(
        {
            "tool_calls": [
                {"name": "go_to_keyframe", "args": {"keyframe_node_id": keyframe_id}}
            ],
            "tool_results": [
                {
                    "name": "go_to_keyframe",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
            "final_ai_content": "",
        },
        ensure_ascii=False,
    )


def _build_tasks(keyframe_id: int = 7) -> dict[int, dict[str, Any]]:
    return {
        1: {
            "task_id": 1,
            "task_type": "navigation_action",
            "type": "action",
            "description": "Navigate to staging keyframe",
            "status": "completed",
            "next_task_id": 2,
            "condition": None,
            "branches": None,
            "depends_on": [],
            "plan_id": PLAN_ID,
            "target": {
                "type": "semantic_keyframe",
                "query": "elevator entrance with fire extinguisher box",
                "selection_policy": "choose a clear, close view of the final object",
            },
            "outputs": ["destination", "current_place_context"],
            "result": [
                {
                    "event_id": "event_1",
                    "summary": "Resolved staging keyframe.",
                    "created_at": "audit",
                    "raw_output": _staging_raw_output(keyframe_id),
                }
            ],
        },
        2: {
            "task_id": 2,
            "task_type": "llm_action",
            "type": "action",
            "description": "Wait for arrival at staging keyframe",
            "status": "waiting",
            "next_task_id": 3,
            "condition": None,
            "branches": None,
            "depends_on": [1],
            "plan_id": PLAN_ID,
            "result": [],
        },
        3: {
            "task_id": 3,
            "task_type": "llm_action",
            "type": "action",
            "description": "Resolve the semantic object destination at the arrived staging place",
            "status": "pending",
            "next_task_id": 4,
            "condition": None,
            "branches": None,
            "depends_on": [2],
            "plan_id": PLAN_ID,
            "primary_target": "black chair",
            "target": {
                "type": "semantic_object",
                "object_description": "black chair",
                "inputs_from": {"place_context": "task1.current_place_context"},
                "stop_distance_m": 0.8,
            },
            "outputs": ["destination", "selected_object"],
            "result": [],
        },
        4: {
            "task_id": 4,
            "task_type": "navigation_action",
            "type": "action",
            "description": "Navigate to object position",
            "status": "pending",
            "next_task_id": None,
            "condition": None,
            "branches": None,
            "depends_on": [3],
            "plan_id": PLAN_ID,
            "target": {"type": "task_output", "task_id": 3, "field": "destination"},
            "result": [],
        },
    }


def _build_tasks_final_only_staging(keyframe_id: int = 33) -> dict[int, dict[str, Any]]:
    tasks = _build_tasks(keyframe_id)
    tasks[1]["result"] = [
        {
            "event_id": "event_final_only",
            "summary": "Resolved staging keyframe from final answer only.",
            "created_at": "audit",
            "raw_output": _staging_raw_output_final_only(keyframe_id),
        }
    ]
    return tasks


def _build_tasks_without_explicit_staging_metadata(keyframe_id: int = 74) -> dict[int, dict[str, Any]]:
    """Real planner shape: object task depends on navigation, not staging resolver."""

    tasks = _build_tasks(keyframe_id)
    tasks[1].pop("primary_target", None)
    tasks[2]["result"] = [
        {
            "event_id": "event_nav_wait",
            "summary": f"Heading to keyframe {keyframe_id}.",
            "created_at": "audit",
            "raw_output": _navigation_raw_output_target_keyframe(keyframe_id),
        }
    ]
    tasks[3].pop("primary_target", None)
    tasks[3]["description"] = (
        "Resolve the fire extinguisher box destination with semantic object localization, "
        "using the arrived staging context and live perception when needed"
    )
    tasks[3]["depends_on"] = [2]
    return tasks


def _runtime_control(current_task_id: int) -> dict[str, Any]:
    control = build_runtime_control({})
    control["active_plan_id"] = PLAN_ID
    control["background_generation"] = 1
    record_foreground_task(control, current_task_id)
    return control


def _run_background(tool: Any, *, keyframe_id: int = 7) -> tuple[dict[str, Any], _Logger]:
    tasks = _build_tasks(keyframe_id)
    state = {"tasks": tasks, "current_plan_id": PLAN_ID, "current_task_id": 2}
    shared_results: dict[int, Any] = {}
    logger = _Logger()
    store = BackgroundResultStore(
        task=tasks[3],
        node_name="bg_audit",
        active_generation=1,
        shared_background_results=shared_results,
        shared_processing_tasks=set(),
        shared_runtime_control=_runtime_control(2),
        logger=logger,
        run_memory=None,
    )
    coordinator = BackgroundForegroundCoordinator(
        state=state,
        task_id=3,
        shared_runtime_control=store.shared_runtime_control,
    )
    run_background_analysis(
        state=state,
        task_copy=tasks[3],
        llm=None,
        tools=[tool],
        background_prompt="",
        store=store,
        coordinator=coordinator,
        logger=logger,
        run_memory=None,
    )
    return shared_results[3], logger


def audit() -> dict[str, Any]:
    tasks_after_task1 = _build_tasks(7)
    tasks_after_task1[2]["status"] = "pending"
    selected_after_task1 = select_background_target_task(
        {"tasks": tasks_after_task1, "current_plan_id": PLAN_ID, "current_task_id": 1},
        worker_id=0,
        total_workers=2,
        shared_background_results={},
        shared_processing_tasks=set(),
        shared_runtime_control=_runtime_control(1),
    )

    tasks_during_nav = deepcopy(_build_tasks(7))
    selected_during_nav = select_background_target_task(
        {"tasks": tasks_during_nav, "current_plan_id": PLAN_ID, "current_task_id": 2},
        worker_id=0,
        total_workers=2,
        shared_background_results={},
        shared_processing_tasks=set(),
        shared_runtime_control=_runtime_control(2),
    )

    tasks_final_only = _build_tasks_final_only_staging(33)
    selected_final_only = select_background_target_task(
        {"tasks": tasks_final_only, "current_plan_id": PLAN_ID, "current_task_id": 2},
        worker_id=0,
        total_workers=2,
        shared_background_results={},
        shared_processing_tasks=set(),
        shared_runtime_control=_runtime_control(2),
    )

    tasks_nav_target_only = _build_tasks_without_explicit_staging_metadata(74)
    selected_nav_target_only = select_background_target_task(
        {"tasks": tasks_nav_target_only, "current_plan_id": PLAN_ID, "current_task_id": 2},
        worker_id=0,
        total_workers=2,
        shared_background_results={},
        shared_processing_tasks=set(),
        shared_runtime_control=_runtime_control(2),
    )

    success_record, success_logger = _run_background(_SuccessPreanalysisTool(), keyframe_id=7)
    success_fast_path = _try_complete_semantic_grounding_from_background(
        _build_tasks(7)[3],
        tasks=_build_tasks(7),
        background_result=success_record,
    )

    failure_record, failure_logger = _run_background(_FailPreanalysisTool(), keyframe_id=9)
    failure_reference = build_background_reference(failure_record)
    failure_fast_path = _try_complete_semantic_grounding_from_background(
        _build_tasks(9)[3],
        tasks=_build_tasks(9),
        background_result=failure_record,
    )

    results = {
        "selected_after_task1": selected_after_task1.get("task_id") if selected_after_task1 else None,
        "selected_during_nav": selected_during_nav.get("task_id") if selected_during_nav else None,
        "selected_final_only_staging": selected_final_only.get("task_id") if selected_final_only else None,
        "selected_nav_target_only": selected_nav_target_only.get("task_id") if selected_nav_target_only else None,
        "success_status": success_record.get("status"),
        "success_destination": success_record.get("recommended_destination"),
        "success_fast_path_event": success_fast_path.get("event_type") if success_fast_path else None,
        "success_fast_path_tool": success_fast_path.get("tool_name") if success_fast_path else None,
        "success_logs": success_logger.lines,
        "failure_status": failure_record.get("status"),
        "failure_reason": failure_record.get("failure_reason"),
        "failure_has_destination": "recommended_destination" in failure_record,
        "failure_fast_path": failure_fast_path,
        "failure_reference": failure_reference,
        "failure_logs": failure_logger.lines,
    }

    assert results["selected_after_task1"] == 3
    assert results["selected_during_nav"] == 3
    assert results["selected_final_only_staging"] == 3
    assert results["selected_nav_target_only"] == 3
    assert results["success_status"] == "completed"
    assert isinstance(results["success_destination"], dict)
    assert results["success_fast_path_event"] == "task_completed"
    assert results["success_fast_path_tool"] == "background_preanalysis"
    assert any("Historical object preanalysis on keyframe 7" in line for line in results["success_logs"])
    assert any("Historical object preanalysis completed" in line for line in results["success_logs"])
    assert results["failure_status"] == "failed"
    assert results["failure_reason"] == "right image missing"
    assert results["failure_has_destination"] is False
    assert results["failure_fast_path"] is None
    assert "right image missing" in str(results["failure_reference"])
    assert any("Historical object preanalysis failed" in line for line in results["failure_logs"])
    return results


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
