import json

from caragent_agent.agents.async_agent.guidance import navigation_waiting_text
from caragent_agent.agents.async_agent.planning.task_graph import (
    derive_display_label_from_user_text,
    parse_planned_tasks_from_response,
)


def test_semantic_navigation_keeps_user_language_display_label():
    plan_text = json.dumps(
        {
            "tasks": [
                {
                    "task_id": 1,
                    "task_type": "navigation_action",
                    "description": "Navigate to a seating area or place with chairs/benches.",
                    "target": {
                        "type": "semantic_keyframe",
                        "target_source": "scene_memory",
                        "target_kind": "place",
                        "query": "seating area with chairs or benches",
                    },
                    "outputs": ["destination", "current_place_context"],
                    "depends_on": [],
                    "next_task_id": None,
                }
            ]
        }
    )
    user_label = derive_display_label_from_user_text(
        "我想找一个可以坐下的地方，请帮我找一个合适的位置，并带我过去。"
    )

    tasks, first_task_id = parse_planned_tasks_from_response(
        plan_text,
        plan_id="plan_test",
        user_input_id="user_input_test",
        created_at="2026-06-28T00:00:00Z",
        user_display_label=user_label,
    )

    task = tasks[first_task_id]
    assert task["target"]["query"] == "seating area with chairs or benches"
    assert task["target"]["display_label"] == "一个可以坐下的地方"
    assert "一个可以坐下的地方" in navigation_waiting_text(task)
    assert "seating area" not in navigation_waiting_text(task)
