"""Run live planner API regressions for keyframe/object/image planning.

This script calls the configured planner LLM once per case. It does not execute
tools, navigation, or robot actions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from caragent_agent.agents.async_agent.planning.task_graph import (
    parse_planned_tasks_from_response,
)
from caragent_agent.config.config import config, ensure_api_key_env
from caragent_agent.utils.llm_request_generator import agent_prompts


FORBIDDEN_DESCRIPTION_PHRASES = (
    "after arrival",
    "in the current view",
    "current-view",
)


def _planner_model_name() -> str:
    routing = config.get("llm_routing")
    if isinstance(routing, dict) and routing.get("planner"):
        return str(routing["planner"]).strip()
    return str(config.get("agent_core_llm_model") or "deepseek-chat").strip()


def _request_timeout_sec() -> float:
    raw_timeout = config.get("llm_request_timeout_sec", 60)
    if isinstance(raw_timeout, dict):
        raw_timeout = raw_timeout.get("planner") or raw_timeout.get("default") or 60
    return float(raw_timeout or 60)


def _build_planner_llm() -> ChatOpenAI:
    model_name = _planner_model_name()
    normalized = model_name.lower()
    if normalized.startswith("qwen"):
        api_key = ensure_api_key_env("qwen")
        return ChatOpenAI(
            api_key=str(api_key).strip(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model_name,
            temperature=0.0,
            extra_body={"enable_thinking": False},
            timeout=_request_timeout_sec(),
        )
    if normalized.startswith("deepseek"):
        api_key = ensure_api_key_env("deepseek")
        return ChatOpenAI(
            api_key=str(api_key).strip(),
            base_url="https://api.deepseek.com/v1",
            model=model_name,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=_request_timeout_sec(),
        )
    raise ValueError(f"Unsupported planner model: {model_name}")


def _attached_message(text: str, description: str) -> str:
    image_ref = {
        "image_ref_id": "latest",
        "path": "/tmp/audit_attached_image.jpg",
        "source": "audit",
        "description": description,
    }
    return "\n\n".join(
        [
            f"User text request:\n{text}",
            "[ATTACHED_IMAGES_JSON]",
            json.dumps([image_ref], ensure_ascii=False, indent=2),
            "[/ATTACHED_IMAGES_JSON]",
            (
                "Instruction: the attached image metadata above is available to "
                "the planner and executor. Only tasks that must inspect the image "
                "should set image_refs:[\"latest\"]. For image-only navigation, "
                "resolve the closest matching keyframe or image-contained object "
                "before navigating."
            ),
        ]
    )


def _plan(llm: ChatOpenAI, user_request: str, case_name: str) -> dict[int, dict[str, Any]]:
    response = llm.invoke(
        [
            SystemMessage(content=agent_prompts.get("plan_system", "")),
            HumanMessage(content=user_request),
        ]
    )
    plan_text = str(response.content or "").strip()
    tasks, first_task_id = parse_planned_tasks_from_response(
        plan_text,
        plan_id=f"audit_{case_name}",
        user_input_id=f"audit_input_{case_name}",
        created_at="2026-06-20T00:00:00+08:00",
    )
    assert first_task_id in tasks, f"{case_name}: invalid first_task_id {first_task_id}"
    return tasks


def _descriptions(tasks: dict[int, dict[str, Any]]) -> str:
    return "\n".join(str(task.get("description") or "") for task in tasks.values()).lower()


def _assert_no_misleading_descriptions(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    text = _descriptions(tasks)
    for phrase in FORBIDDEN_DESCRIPTION_PHRASES:
        if phrase == "in the current view" and case_name == "attached_object":
            continue
        assert phrase not in text, f"{case_name}: misleading task phrase remained: {phrase}"


def _assert_has_navigation_pair(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    assert any(task.get("task_type") == "navigation_action" for task in tasks.values()), f"{case_name}: no navigation"


def _navigation_targets(tasks: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        task.get("target")
        for task in tasks.values()
        if task.get("task_type") == "navigation_action" and isinstance(task.get("target"), dict)
    ]


def _assert_plain_place(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    _assert_has_navigation_pair(case_name, tasks)
    assert not any(task.get("image_refs") for task in tasks.values()), f"{case_name}: unexpected image_refs"
    assert not any(task.get("resolver_kind") for task in tasks.values()), f"{case_name}: legacy resolver_kind remained"
    targets = _navigation_targets(tasks)
    assert any(target.get("type") == "semantic_keyframe" for target in targets), (
        f"{case_name}: plain place should use semantic_keyframe navigation"
    )


def _assert_object_plan(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    _assert_has_navigation_pair(case_name, tasks)
    assert not any(task.get("resolver_kind") for task in tasks.values()), f"{case_name}: legacy resolver_kind remained"
    targets = _navigation_targets(tasks)
    assert any(target.get("type") == "semantic_keyframe" for target in targets), (
        f"{case_name}: missing semantic staging keyframe navigation"
    )
    assert any(target.get("type") == "semantic_object" for target in targets), (
        f"{case_name}: missing semantic object navigation"
    )
    object_tasks = [
        task
        for task in tasks.values()
        if task.get("task_type") == "navigation_action"
        and isinstance(task.get("target"), dict)
        and task.get("target", {}).get("type") == "semantic_object"
    ]
    assert object_tasks, f"{case_name}: missing object navigation task"
    assert any(task.get("depends_on") or task.get("inputs_from") for task in object_tasks), (
        f"{case_name}: semantic_object task should depend on staging context"
    )


def _assert_image_place(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    _assert_has_navigation_pair(case_name, tasks)
    image_tasks = [task for task in tasks.values() if task.get("image_refs")]
    assert image_tasks, f"{case_name}: no task kept image_refs"
    assert not any(task.get("task_type") == "navigation_action" and task.get("image_refs") for task in tasks.values()), (
        f"{case_name}: navigation task should not inspect image"
    )
    assert not any(task.get("resolver_kind") for task in tasks.values()), f"{case_name}: legacy resolver_kind remained"


def _assert_image_object(case_name: str, tasks: dict[int, dict[str, Any]]) -> None:
    assert not any(task.get("resolver_kind") for task in tasks.values()), f"{case_name}: legacy resolver_kind remained"
    image_tasks = [task for task in tasks.values() if task.get("image_refs")]
    assert image_tasks, f"{case_name}: image object plan should keep image_refs on an analysis task"
    assert all(task.get("task_type") == "llm_action" for task in image_tasks), (
        f"{case_name}: only llm_action tasks should inspect attached images"
    )
    assert any(
        isinstance(task.get("target"), dict)
        and task.get("target", {}).get("type") == "task_output"
        for task in tasks.values()
        if task.get("task_type") == "navigation_action"
    ), f"{case_name}: attached-image object plan should navigate to the matched staging keyframe"
    assert any(
        isinstance(task.get("target"), dict)
        and task.get("target", {}).get("type") == "semantic_object"
        for task in tasks.values()
        if task.get("task_type") == "navigation_action"
    ), f"{case_name}: attached-image object plan should use semantic_object after staging arrival"


def main() -> int:
    llm = _build_planner_llm()
    cases = [
        ("plain_place", "Go to the elevator entrance.", _assert_plain_place),
        ("plain_stairs", "Go to the stair entrance.", _assert_plain_place),
        ("object_target", "Go to the gray four-legged table.", _assert_object_plan),
        (
            "attached_place",
            _attached_message(
                "Go to this image location.",
                "A hallway area with a doorway and a wall sign near the entrance.",
            ),
            _assert_image_place,
        ),
        (
            "attached_object",
            _attached_message(
                "Go to the black chair in this image.",
                "A black chair is visible near a desk in an indoor hallway scene.",
            ),
            _assert_image_object,
        ),
    ]

    summary: dict[str, Any] = {}
    for case_name, request, checker in cases:
        tasks = _plan(llm, request, case_name)
        _assert_no_misleading_descriptions(case_name, tasks)
        checker(case_name, tasks)
        summary[case_name] = [
            {
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "description": task.get("description"),
                "target": task.get("target"),
                "outputs": task.get("outputs"),
                "depends_on": task.get("depends_on"),
                "inputs_from": task.get("inputs_from"),
                "image_refs": task.get("image_refs"),
            }
            for _, task in sorted(tasks.items())
        ]

    output_path = Path("/tmp/caragent_planner_prompt_audit.json")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_path": str(output_path), "cases": list(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
