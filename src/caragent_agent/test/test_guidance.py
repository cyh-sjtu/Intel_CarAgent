from caragent_agent.agents.async_agent.guidance import (
    append_guidance,
    append_guidance_event,
    build_guidance_event,
    navigation_arrival_text,
    navigation_waiting_text,
    plan_created_text,
)


def test_navigation_guidance_text_is_chinese_and_bounded():
    task = {
        "task_id": 7,
        "description": "Navigate to the staircase",
        "target": {"type": "semantic_keyframe", "query": "楼梯口"},
    }

    waiting = navigation_waiting_text(task)
    arrived = navigation_arrival_text(task)

    assert "正在启动前往目标地点" in waiting
    assert "楼梯口" in waiting
    assert "旁站人员" in waiting
    assert "已到达目标地点" in arrived
    assert "前方安全" not in waiting


def test_navigation_guidance_prefers_user_display_label_over_work_query():
    task = {
        "task_id": 8,
        "description": "Navigate to a seating area or place with chairs/benches.",
        "target": {
            "type": "semantic_keyframe",
            "query": "seating area with chairs or benches",
            "display_label": "一个可以坐下的地方",
        },
    }

    waiting = navigation_waiting_text(task)

    assert "一个可以坐下的地方" in waiting
    assert "seating area" not in waiting


def test_guidance_event_dedupes_latest_duplicate():
    first = build_guidance_event(
        event_type="navigation_start",
        text="已确定目的地，准备前往楼梯口。",
        dedupe_key="navigation_start:7",
        task_id=7,
    )
    state = append_guidance_event({}, first)
    state = append_guidance(
        state,
        event_type="navigation_start",
        text="已确定目的地，准备前往楼梯口。",
        dedupe_key="navigation_start:7",
        task_id=7,
    )

    assert len(state["guidance_events"]) == 1
    assert state["guidance_events"][0]["task_id"] == 7


def test_plan_created_text_mentions_scene_memory():
    assert "已生成计划，包含 2 个任务" in plan_created_text(2)
