"""Foreground tool for approaching one object visible in the current camera view."""

from __future__ import annotations

import time
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    get_runtime_tool_context,
)
from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.agents.async_agent.runtime.legacy_task_metadata import legacy_object_kind
from caragent_agent.config.config import config


ARTIFACT_KEYS = (
    "output_dir",
    "summary_json",
    "status_json",
    "approach_goal_json",
    "approach_debug_png",
    "debug_png",
    "selected_grounding_json",
    "mono_guard_json",
    "stereo_json",
    "segmentation_json",
)


def _object_approach_brief(result: dict[str, Any]) -> str:
    output_dir = str(result.get("output_dir") or "")
    summary_json = str(result.get("summary_json") or "")
    stages = result.get("stages") if isinstance(result.get("stages"), list) else []
    stage_text = " -> ".join(
        f"{stage.get('name')}:{stage.get('status')}"
        for stage in stages
        if isinstance(stage, dict) and stage.get("name")
    )
    parts = []
    if output_dir:
        parts.append(f"output_dir={output_dir}")
    if summary_json:
        parts.append(f"summary_json={summary_json}")
    if stage_text:
        parts.append(f"stages={stage_text}")
    return "; ".join(parts)


def _artifact_paths(result: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    raw_paths = result.get("paths") if isinstance(result.get("paths"), dict) else {}
    for key in ARTIFACT_KEYS:
        value = result.get(key) or raw_paths.get(key)
        if value:
            paths[key] = str(value)
    approach = result.get("approach") if isinstance(result.get("approach"), dict) else {}
    if approach.get("debug_png"):
        paths.setdefault("approach_debug_png", str(approach.get("debug_png")))
    return paths


def _stage_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    stages = result.get("stages") if isinstance(result.get("stages"), list) else []
    compact = []
    for stage in stages[:12]:
        if not isinstance(stage, dict):
            continue
        item = {
            "name": stage.get("name"),
            "status": stage.get("status"),
        }
        if stage.get("error"):
            item["error"] = str(stage.get("error"))
        compact.append({key: value for key, value in item.items() if value not in (None, "")})
    return compact


def _approach_key_metrics(result: dict[str, Any]) -> dict[str, Any]:
    approach = result.get("approach") if isinstance(result.get("approach"), dict) else {}
    checks = approach.get("checks") if isinstance(approach.get("checks"), dict) else {}
    metrics: dict[str, Any] = {
        "elapsed_sec": result.get("elapsed_sec"),
        "depth_backend": result.get("depth_backend"),
        "stop_distance_m": approach.get("stop_distance_m"),
        "object_base_xyz_m": approach.get("object_base_xyz_m"),
        "object_map_xy_m": approach.get("object_map_xy_m"),
    }
    for key in (
        "candidate_count",
        "grid_source",
        "line_of_sight_blocked",
        "target_depth_m",
        "selected_depth_m",
        "selected_source",
        "fit_rmse",
        "correction_delta",
        "mono_guard_status",
        "mono_guard_selected_source",
        "mono_guard_reason",
        "mono_guard_selected_depth_m",
        "mono_guard_fused_median_m",
        "mono_guard_correction_delta_m",
        "mono_guard_fused_iqr_m",
        "mono_guard_anchor_count",
        "mono_guard_fit_rmse_m",
    ):
        if checks.get(key) is not None:
            metrics[key] = checks.get(key)
    return {key: value for key, value in metrics.items() if value not in (None, "", [], {})}


def _compact_object_approach_result(
    result: dict[str, Any],
    *,
    target_description: str,
    destination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    approach = result.get("approach") if isinstance(result.get("approach"), dict) else {}
    status = str(approach.get("status") or result.get("status") or "").strip() or "unknown"
    reason = approach.get("reason") or result.get("error")
    planner = approach.get("planner") if isinstance(approach.get("planner"), dict) else {}
    compact = {
        "status": result.get("status") or status,
        "summary": result.get("summary"),
        "destination": destination or result.get("destination"),
        "target_description": target_description,
        "depth_backend": result.get("depth_backend"),
        "approach_status": status,
        "approach_reason": reason,
        "mode": approach.get("mode") or planner.get("mode"),
        "artifact_paths": _artifact_paths(result),
        "key_metrics": _approach_key_metrics(result),
        "stages": _stage_summary(result),
    }
    if reason and status not in {"ok", "already_close"}:
        compact["failure_reason"] = str(reason)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _navigation_config() -> dict[str, Any]:
    nav_cfg = config.get("navigation")
    return nav_cfg if isinstance(nav_cfg, dict) else {}


def _simulation_direct_fallback_allowed(
    current_state: dict[str, Any],
    approach: dict[str, Any],
) -> bool:
    nav_cfg = _navigation_config()
    if not bool(nav_cfg.get("simulation_mode")):
        return False
    if not bool(nav_cfg.get("simulation_allow_direct_object_approach_without_costmap", True)):
        return False
    if str(approach.get("status") or "").strip().lower() != "degraded":
        return False
    if str(approach.get("reason") or "").strip().lower() != "costmap_unavailable":
        return False
    planner = approach.get("planner") if isinstance(approach.get("planner"), dict) else {}
    if str(planner.get("mode") or approach.get("mode") or "").strip().lower() != "direct_fallback":
        return False
    state_source = str(current_state.get("source") or "").strip().lower()
    simulation_meta = current_state.get("simulation") if isinstance(current_state.get("simulation"), dict) else {}
    return state_source == "simulation" or bool(simulation_meta.get("enabled"))


def _flatten_reference_text(value: Any) -> list[str]:
    """Collect compact schema-reference text from task metadata."""

    chunks: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            chunks.append(text)
        return chunks
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).strip()
            if key_text:
                chunks.append(key_text)
            chunks.extend(_flatten_reference_text(item))
        return chunks
    if isinstance(value, (list, tuple, set)):
        for item in value:
            chunks.extend(_flatten_reference_text(item))
    return chunks


def _task_is_historical_object_preanalysis_candidate(task: Any) -> bool:
    """Return True only for semantic-object tasks staged from scene memory.

    Current-view semantic_object tasks should run live perception immediately.
    Historical preanalysis is useful only when the task explicitly consumes a
    prior place/keyframe context, such as a staging navigation result.
    """

    if not isinstance(task, dict):
        return False
    target = task.get("target")
    if (
        task.get("task_type") == "navigation_action"
        and isinstance(target, dict)
        and str(target.get("type") or "").strip() == "semantic_object"
    ):
        reference_text = " ".join(
            _flatten_reference_text(task.get("inputs_from"))
            + _flatten_reference_text(target.get("inputs_from"))
        ).lower()
        return any(
            token in reference_text
            for token in (
                "current_place_context",
                "semantic_keyframe",
                "keyframe",
                "staging",
            )
        )
    return legacy_object_kind(task)


def _artifact_paths_from_preanalysis(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    paths = value.get("paths") if isinstance(value.get("paths"), dict) else {}
    artifacts = value.get("artifact_paths") if isinstance(value.get("artifact_paths"), dict) else {}
    result: dict[str, str] = {}
    for key in ARTIFACT_KEYS:
        item = value.get(key) or paths.get(key) or artifacts.get(key)
        if item:
            result[key] = str(item)
    return result


def _key_metrics_from_preanalysis(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    approach = value.get("approach") if isinstance(value.get("approach"), dict) else {}
    checks = approach.get("checks") if isinstance(approach.get("checks"), dict) else {}
    metrics = {
        "depth_backend": value.get("depth_backend"),
        "approach_status": approach.get("status"),
        "approach_reason": approach.get("reason"),
        "mode": approach.get("mode"),
        "object_base_xyz_m": approach.get("object_base_xyz_m"),
        "object_map_xy_m": approach.get("object_map_xy_m"),
    }
    for key in (
        "mono_guard_selected_source",
        "mono_guard_reason",
        "mono_guard_selected_depth_m",
        "stereo_valid_ratio",
        "stereo_valid_pixels",
        "candidate_count",
        "grid_source",
    ):
        if checks.get(key) is not None:
            metrics[key] = checks.get(key)
    return {key: val for key, val in metrics.items() if val not in (None, "", [], {})}


class ApproachObjectInCurrentViewTool(ToolBase):
    """Resolve one visible object to a concrete map pose for later navigation."""

    def __init__(self, controller: Any):
        super().__init__(
            name="approach_object_in_current_view",
            description=(
                "Resolve a specific local visual target visible in the current camera view "
                "into a concrete navigation destination pose. This is semantic object grounding, similar "
                "to keyframe matching, and should be followed by a navigation_action "
                "or handled by a semantic_object navigation target. The tool manages long-running background reuse and "
                "bounded live perception attempts internally; do not call it repeatedly in the "
                "same task just to retry the same unresolved object."
            ),
            capability_tags=("semantic_grounding", "foreground_only", "object_approach"),
        )
        self.controller = controller
        self._object_approach_pipeline = None
        self._preload_error = ""
        self._preload_started = False
        self._preload_completed = False
        self._preload_duration_sec: float | None = None
        self._pipeline_lock = threading.RLock()
        self._task_attempt_cache: dict[str, dict[str, Any]] = {}
        if self._preload_on_init():
            self._start_preload(async_preload=self._preload_async())

    def execute(
        self,
        target_description: str,
        grounding_query: str = "",
        vlm_query: str = "",
        sam_query: str = "",
        stop_distance_m: float = 0.8,
        dispatch: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target = str(target_description or "").strip()
        target_source = str(kwargs.get("target_source") or "").strip().lower()
        if not target:
            return self.blocked(
                "Object approach needs a target description.",
                error={"code": "missing_target_description"},
                provenance={"source_type": "live_perception"},
            )
        if self.controller is None:
            return self.blocked(
                "Object approach is unavailable because no navigation controller is attached.",
                error={"code": "controller_unavailable"},
                provenance={"source_type": "live_perception"},
            )

        task_cache_key = self._current_task_cache_key()
        attempt_record = self._task_attempt_record(task_cache_key)
        requested_backend = self._configured_depth_backend()
        call_signature = self._call_signature(
            target=target,
            grounding_query=grounding_query,
            vlm_query=vlm_query,
            sam_query=sam_query,
            stop_distance_m=stop_distance_m,
            depth_backend=requested_backend,
        )

        def finish(tool_result: dict[str, Any], *, phase: str, live_attempted: bool = False) -> dict[str, Any]:
            self._store_task_attempt_result(
                task_cache_key,
                tool_result,
                phase=phase,
                live_attempted=live_attempted,
                call_signature=call_signature,
            )
            return tool_result

        use_historical_preanalysis = self._use_historical_preanalysis_policy(target_source)
        if use_historical_preanalysis:
            background_result = self._wait_for_background_object_preanalysis(
                target,
                task_cache_key=task_cache_key,
                attempt_record=attempt_record,
            )
            if background_result is not None:
                return finish(background_result, phase="background_preanalysis_reused")
            if self._historical_preanalysis_required(attempt_record):
                return finish(
                    self._background_preanalysis_unavailable_result(
                        target=target,
                        task_cache_key=task_cache_key,
                        attempt_record=attempt_record,
                    ),
                    phase="background_preanalysis_unavailable",
                )

        duplicate_result = self._cached_attempt_if_same_call(
            target,
            task_cache_key,
            attempt_record,
            call_signature,
        )
        if duplicate_result is not None:
            return duplicate_result

        image = self._current_image()
        if image is None:
            return finish(self.blocked(
                "Current camera image is unavailable; cannot locate the object.",
                error={"code": "current_image_unavailable"},
                provenance={"source_type": "live_perception"},
            ), phase="precondition_blocked")
        right_image = self._current_right_image()
        if requested_backend in {"stereo", "stereo_primary_mono_guard"} and right_image is None:
            return finish(self.blocked(
                "Current right stereo image is unavailable; object approach waits for the verified stereo localization path.",
                error={"code": "current_right_image_unavailable"},
                provenance={"source_type": "live_perception"},
            ), phase="precondition_blocked")
        scan = self._current_scan()
        if requested_backend == "mono_relative_lidar" and scan is None:
            return finish(self.blocked(
                "Current LaserScan is unavailable; mono-relative + LiDAR localization cannot run.",
                error={"code": "current_scan_unavailable"},
                provenance={"source_type": "live_perception"},
            ), phase="precondition_blocked")

        current_state = {}
        if hasattr(self.controller, "get_current_state"):
            current_state = self.controller.get_current_state()

        live_attempt_index = self._reserve_live_attempt(task_cache_key, attempt_record)

        output_root = Path(
            (config.get("paths") or {}).get("workspace_root", "/home/car/caragent_ws")
        ) / "perception_outputs" / "object_approach"
        try:
            pipeline = self._pipeline(output_root)
            def log_progress(event: dict[str, Any]) -> None:
                name = str(event.get("name") or "unknown")
                status = str(event.get("status") or "unknown")
                elapsed = float(event.get("elapsed_sec") or 0.0)
                message = str(event.get("message") or "").strip()
                text = (
                    f"ObjectApproach: {name}={status} elapsed={elapsed:.1f}s "
                    f"live_attempt={live_attempt_index}"
                )
                if message:
                    text = f"{text} - {message}"
                self._emit_progress(name, status, message=text)

            result = pipeline.run(
                image=image,
                right_image=right_image,
                scan_msg=scan,
                target_description=target,
                current_state=current_state,
                grounding_query=grounding_query,
                vlm_query=vlm_query,
                sam_query=sam_query,
                stop_distance_m=float(stop_distance_m or 0.8),
                depth_backend=requested_backend,
                dispatch=False,
                progress_callback=log_progress,
            )
        except Exception as exc:
            return finish(self.partial(
                f"Could not safely approach '{target}': semantic object perception failed with {type(exc).__name__}: {exc}",
                data={
                    "target_description": target,
                    "grounding_query": grounding_query,
                    "vlm_query": vlm_query,
                    "sam_query": sam_query,
                    "stop_distance_m": float(stop_distance_m or 0.8),
                    "depth_backend": requested_backend,
                    "output_root": str(output_root),
                },
                error={
                    "code": "object_approach_exception",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                provenance={"source_type": "live_perception"},
            ), phase="live_exception", live_attempted=True)

        approach = result.get("approach") if isinstance(result, dict) else None
        if not isinstance(approach, dict):
            compact_data = (
                _compact_object_approach_result(result, target_description=target)
                if isinstance(result, dict)
                else {"target_description": target}
            )
            self._attach_attempt_metadata(
                compact_data,
                task_cache_key=task_cache_key,
                live_attempt_index=live_attempt_index,
            )
            return finish(self.error_result(
                "Object approach pipeline failed before producing an approach result.",
                data=compact_data,
                error={"code": "pipeline_failed", "message": result.get("error") if isinstance(result, dict) else ""},
                provenance={"source_type": "live_perception"},
            ), phase="live_pipeline_failed", live_attempted=True)

        status = str(approach.get("status") or result.get("status") or "").strip()
        if status == "ok":
            goal = (approach.get("map_goal") or {})
            position = goal.get("position")
            yaw_deg = goal.get("yaw_deg")
            if not isinstance(position, list) or len(position) < 2 or yaw_deg is None:
                compact_data = _compact_object_approach_result(
                    result,
                    target_description=target,
                    destination=result.get("destination") if isinstance(result, dict) else None,
                )
                self._attach_attempt_metadata(
                    compact_data,
                    task_cache_key=task_cache_key,
                    live_attempt_index=live_attempt_index,
                )
                return finish(self.error_result(
                    "Object approach computed an invalid destination pose.",
                    data=compact_data,
                    error={"code": "invalid_map_goal"},
                    provenance={"source_type": "live_perception"},
                ), phase="live_invalid_goal", live_attempted=True)
            destination = {
                "type": "position",
                "position": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2] if len(position) > 2 else 0.0),
                ],
                "yaw_deg": float(yaw_deg),
                "source": "object_approach",
                "target_description": target,
            }
            result["destination"] = destination
            result["nav2_dispatched"] = False
            compact_data = _compact_object_approach_result(
                result,
                target_description=target,
                destination=destination,
            )
            self._attach_attempt_metadata(
                compact_data,
                task_cache_key=task_cache_key,
                live_attempt_index=live_attempt_index,
            )
            summary = (
                f"Resolved semantic object destination for '{target}' at "
                f"x={destination['position'][0]:.2f}, y={destination['position'][1]:.2f}, "
                f"yaw={destination['yaw_deg']:.1f} deg."
            )
            brief = _object_approach_brief(result)
            if brief:
                summary = f"{summary} {brief}"
            return finish(self.ok(
                summary,
                data=compact_data,
                provenance={"source_type": "live_perception"},
            ), phase="live_ok", live_attempted=True)

        if status == "degraded" and _simulation_direct_fallback_allowed(current_state, approach):
            goal = (approach.get("map_goal") or {})
            position = goal.get("position")
            yaw_deg = goal.get("yaw_deg")
            if isinstance(position, list) and len(position) >= 2 and yaw_deg is not None:
                destination = {
                    "type": "position",
                    "position": [
                        float(position[0]),
                        float(position[1]),
                        float(position[2] if len(position) > 2 else 0.0),
                    ],
                    "yaw_deg": float(yaw_deg),
                    "source": "object_approach_simulation_direct_fallback",
                    "target_description": target,
                }
                result["destination"] = destination
                result["nav2_dispatched"] = False
                compact_data = _compact_object_approach_result(
                    result,
                    target_description=target,
                    destination=destination,
                )
                compact_data.pop("failure_reason", None)
                compact_data["simulation_policy"] = {
                    "allowed": True,
                    "reason": "simulation_mode_without_costmap_uses_direct_fallback",
                    "real_robot_requires_costmap": True,
                }
                self._attach_attempt_metadata(
                    compact_data,
                    task_cache_key=task_cache_key,
                    live_attempt_index=live_attempt_index,
                )
                summary = (
                    f"Resolved simulated semantic object destination for '{target}' at "
                    f"x={destination['position'][0]:.2f}, y={destination['position'][1]:.2f}, "
                    f"yaw={destination['yaw_deg']:.1f} deg using direct fallback because costmap is unavailable in workflow simulation."
                )
                brief = _object_approach_brief(result)
                if brief:
                    summary = f"{summary} {brief}"
                return finish(self.ok(
                    summary,
                    data=compact_data,
                    provenance={"source_type": "simulation_live_perception"},
                ), phase="live_simulation_direct_fallback", live_attempted=True)

        if status == "already_close":
            brief = _object_approach_brief(result)
            compact_data = _compact_object_approach_result(
                result,
                target_description=target,
                destination=result.get("destination") if isinstance(result, dict) else None,
            )
            self._attach_attempt_metadata(
                compact_data,
                task_cache_key=task_cache_key,
                live_attempt_index=live_attempt_index,
            )
            return finish(self.ok(
                (
                    f"Already close enough to '{target}'; no backward or extra approach goal was dispatched."
                    + (f" {brief}" if brief else "")
                ),
                data=compact_data,
                provenance={"source_type": "live_perception"},
            ), phase="live_already_close", live_attempted=True)

        brief = _object_approach_brief(result)
        compact_data = _compact_object_approach_result(
            result,
            target_description=target,
            destination=result.get("destination") if isinstance(result, dict) else None,
        )
        self._attach_attempt_metadata(
            compact_data,
            task_cache_key=task_cache_key,
            live_attempt_index=live_attempt_index,
        )
        return finish(self.partial(
            (
                f"Could not safely approach '{target}': {approach.get('reason') or result.get('error') or status}"
                + (f" {brief}" if brief else "")
            ),
            data=compact_data,
            error={"code": "object_approach_unreliable", "reason": approach.get("reason") or status},
            provenance={"source_type": "live_perception"},
        ), phase="live_partial", live_attempted=True)

    def _current_task_cache_key(self) -> str:
        context = get_runtime_tool_context()
        current_task = context.get("current_task")
        if not isinstance(current_task, dict):
            return ""
        try:
            task_id = int(current_task.get("task_id"))
        except Exception:
            return ""
        plan_id = (
            current_task.get("plan_id")
            or context.get("plan_id")
            or context.get("session_id")
            or "current_plan"
        )
        return f"{plan_id}:{task_id}"

    def _task_attempt_record(self, task_cache_key: str) -> dict[str, Any]:
        if not task_cache_key:
            return {}
        if len(self._task_attempt_cache) > 64:
            for key in list(self._task_attempt_cache.keys())[:16]:
                self._task_attempt_cache.pop(key, None)
        return self._task_attempt_cache.setdefault(
            task_cache_key,
            {
                "live_attempt_count": 0,
                "last_phase": "",
                "last_result": None,
                "background_status": "",
            },
        )

    def _historical_preanalysis_required(self, attempt_record: dict[str, Any]) -> bool:
        context = get_runtime_tool_context()
        current_task = context.get("current_task")
        if not _task_is_historical_object_preanalysis_candidate(current_task):
            return False
        status = str(attempt_record.get("background_status") or "").strip().lower()
        return status not in {"not_applicable", ""}

    def _use_historical_preanalysis_policy(self, target_source: str) -> bool:
        """Gate costly live localization after staged historical object tasks.

        The resolver may ask for arrived-scene live localization, but the object
        pipeline is still heavy enough that failed historical preanalysis should
        remain a default safety stop until an explicit, bounded live retry policy
        is introduced.
        """

        normalized = str(target_source or "").strip().lower()
        if normalized == "current_view":
            return False
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        if normalized in {"arrived_scene", "upstream_result"}:
            return not bool(
                agent_cfg.get(
                    "object_approach_force_live_after_arrival_preanalysis_failure",
                    False,
                )
            )
        return True

    def _configured_depth_backend(self) -> str:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        backend = str(agent_cfg.get("object_approach_depth_backend") or "stereo").strip().lower()
        if backend == "auto":
            backend = "stereo"
        allowed = {"stereo", "stereo_primary_mono_guard", "mono_relative_lidar"}
        if backend not in allowed:
            backend = "stereo"
        return backend

    def _reserve_live_attempt(
        self,
        task_cache_key: str,
        attempt_record: dict[str, Any],
    ) -> int:
        if not task_cache_key or not isinstance(attempt_record, dict):
            return 1
        current_count = int(attempt_record.get("live_attempt_count") or 0)
        attempt_record["live_attempt_count"] = current_count + 1
        attempt_record["last_phase"] = "live_running"
        attempt_record["job_status"] = "live_running"
        self._emit_progress(
            "object_live_attempt",
            "started",
            message=(
                "ObjectApproach: object_live_attempt=started "
                f"attempt={current_count + 1}"
            ),
        )
        return current_count + 1

    def _call_signature(
        self,
        *,
        target: str,
        grounding_query: str,
        vlm_query: str,
        sam_query: str,
        stop_distance_m: float,
        depth_backend: str,
    ) -> dict[str, Any]:
        try:
            stop_distance = round(float(stop_distance_m or 0.8), 3)
        except Exception:
            stop_distance = 0.8
        return {
            "target": str(target or "").strip(),
            "grounding_query": str(grounding_query or "").strip(),
            "vlm_query": str(vlm_query or "").strip(),
            "sam_query": str(sam_query or "").strip(),
            "stop_distance_m": stop_distance,
            "depth_backend": str(depth_backend or "").strip().lower(),
        }

    def _attach_attempt_metadata(
        self,
        data: dict[str, Any],
        *,
        task_cache_key: str,
        live_attempt_index: int | None,
    ) -> None:
        if not isinstance(data, dict) or not task_cache_key:
            return
        data["execution_policy"] = {
            "task_cache_key": task_cache_key,
            "live_attempt_index": live_attempt_index,
            "object_job_status": "live_attempt_completed",
            "same_call_signature_returns_cached_result": True,
        }

    def _store_task_attempt_result(
        self,
        task_cache_key: str,
        tool_result: dict[str, Any],
        *,
        phase: str,
        live_attempted: bool = False,
        call_signature: dict[str, Any] | None = None,
    ) -> None:
        if not task_cache_key:
            return
        record = self._task_attempt_record(task_cache_key)
        record["last_phase"] = phase
        record["last_result"] = deepcopy(tool_result)
        record["last_result_was_live_attempt"] = bool(live_attempted)
        record["job_status"] = self._job_status_from_tool_result(tool_result, phase=phase)
        if call_signature:
            record["last_call_signature"] = deepcopy(call_signature)
        record["last_updated_monotonic"] = time.monotonic()

    def _job_status_from_tool_result(self, tool_result: dict[str, Any], *, phase: str) -> str:
        status = str(tool_result.get("status") or "").strip().lower()
        data = tool_result.get("data") if isinstance(tool_result.get("data"), dict) else {}
        if status == "ok" and isinstance(data.get("destination"), dict):
            return "completed"
        if status == "ok":
            return "completed_without_destination"
        if status == "partial":
            return "partial"
        if status in {"blocked", "error"}:
            return status
        return phase or "unknown"

    def _cached_attempt_if_same_call(
        self,
        target: str,
        task_cache_key: str,
        attempt_record: dict[str, Any],
        call_signature: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not task_cache_key or not isinstance(attempt_record, dict):
            return None
        previous_signature = attempt_record.get("last_call_signature")
        if not isinstance(previous_signature, dict) or previous_signature != call_signature:
            return None
        if not isinstance(attempt_record.get("last_result"), dict):
            return None
        return self._cached_attempt_result(
            target,
            task_cache_key,
            attempt_record,
            reuse_reason="same_call_signature",
        )

    def _cached_attempt_result(
        self,
        target: str,
        task_cache_key: str,
        attempt_record: dict[str, Any],
        *,
        reuse_reason: str,
    ) -> dict[str, Any] | None:
        previous = attempt_record.get("last_result")
        if not isinstance(previous, dict):
            return None
        result = deepcopy(previous)
        live_attempt_count = int(attempt_record.get("live_attempt_count") or 0)
        data = result.setdefault("data", {})
        if isinstance(data, dict):
            data["cached_from_same_task"] = True
            data["target_description"] = target
            data["execution_policy"] = {
                "task_cache_key": task_cache_key,
                "live_attempt_count": live_attempt_count,
                "reuse_reason": reuse_reason,
                "object_job_status": attempt_record.get("job_status") or attempt_record.get("last_phase"),
                "same_call_signature_returns_cached_result": True,
            }
            data["retry_recommendation"] = (
                "This is the same semantic object grounding call as the previous one in this task. "
                "Use the returned evidence instead of repeating the same perception run; "
                "only call again if the task has genuinely new visual constraints."
            )
        previous_summary = str(result.get("summary") or "").strip()
        prefix = (
            "Semantic object grounding reused the previous result because this task made "
            "the same tool call again."
        )
        result["summary"] = prefix + (f" Previous result: {previous_summary}" if previous_summary else "")
        provenance = result.setdefault("provenance", {})
        if isinstance(provenance, dict):
            provenance["source_type"] = "same_task_attempt_cache"
        self._emit_progress(
            "object_live_attempt_cache",
            "reused",
            message=(
                "ObjectApproach: object_live_attempt_cache=reused "
                f"reason={reuse_reason} attempts={live_attempt_count}"
            ),
        )
        return result

    def _background_preanalysis_unavailable_result(
        self,
        *,
        target: str,
        task_cache_key: str,
        attempt_record: dict[str, Any],
    ) -> dict[str, Any]:
        status = str(attempt_record.get("background_status") or "unknown").strip().lower()
        reason = str(
            attempt_record.get("background_failure_reason")
            or attempt_record.get("background_error")
            or status
            or "background_preanalysis_unavailable"
        )
        record = attempt_record.get("background_record")
        object_preanalysis = (
            record.get("object_preanalysis")
            if isinstance(record, dict) and isinstance(record.get("object_preanalysis"), dict)
            else {}
        )
        candidate_keyframes = (
            record.get("candidate_keyframe_ids")
            if isinstance(record, dict) and isinstance(record.get("candidate_keyframe_ids"), list)
            else []
        )
        data = {
            "status": status,
            "preanalysis_status": status,
            "target_description": target,
            "source": "background_preanalysis",
            "task_cache_key": task_cache_key,
            "failure_reason": reason,
            "requires_budgeted_live_localization": True,
            "required_next_step": "needs_budgeted_live_localization",
            "candidate_keyframe_ids": candidate_keyframes,
            "background_summary": record.get("summary") if isinstance(record, dict) else None,
            "background_error": record.get("error") if isinstance(record, dict) else None,
            "budget_policy": self._budgeted_live_after_arrival_policy(),
            "artifact_paths": _artifact_paths_from_preanalysis(object_preanalysis),
            "key_metrics": _key_metrics_from_preanalysis(object_preanalysis),
            "object_preanalysis": {
                key: object_preanalysis.get(key)
                for key in ("status", "summary_json", "output_dir", "depth_backend")
                if object_preanalysis.get(key) not in (None, "", [], {})
            },
            "stages": [
                {"name": "object_preanalysis_waiting", "status": status or "unknown"},
                {"name": "object_preanalysis_result", "status": "failed"},
            ],
        }
        data = {key: value for key, value in data.items() if value not in (None, "", [], {})}
        if status in {"failed", "completed_without_destination"}:
            summary = (
                f"Historical object preanalysis for '{target}' failed: {reason}. "
                "A bounded live localization pass is required before this object can become a navigation target."
            )
            return self.partial(
                summary,
                data=data,
                error={"code": "background_object_preanalysis_failed", "reason": reason},
                provenance={"source_type": "background_preanalysis"},
            )
        summary = (
            f"Historical object preanalysis for '{target}' did not reach a final result "
            f"within the watchdog window: {reason}. A bounded live localization pass is required before navigation."
        )
        return self.blocked(
            summary,
            data=data,
            error={"code": "background_object_preanalysis_not_ready", "reason": reason},
            provenance={"source_type": "background_preanalysis"},
        )

    def _budgeted_live_after_arrival_policy(self) -> dict[str, Any]:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        enabled = bool(agent_cfg.get("object_approach_budgeted_live_after_arrival_enabled", False))
        return {
            "enabled": enabled,
            "max_live_attempts_per_task": int(
                agent_cfg.get("object_approach_budgeted_live_after_arrival_max_attempts", 1)
                or 1
            ),
            "auto_retry": False,
            "runs_in_foreground": False,
            "reason": (
                "budgeted_live_after_arrival_disabled"
                if not enabled
                else "budgeted_live_after_arrival_enabled"
            ),
            "config_key": "agent.object_approach_budgeted_live_after_arrival_enabled",
        }

    def _pipeline(self, output_root: Path):
        with self._pipeline_lock:
            pipeline = getattr(self, "_object_approach_pipeline", None)
            if pipeline is None or Path(getattr(pipeline, "output_root", "")) != output_root:
                from caragent_agent.perception.fusion.object_approach_pipeline import ObjectApproachPipeline

                started = time.perf_counter()
                pipeline = ObjectApproachPipeline(output_root=output_root)
                try:
                    pipeline.preload_models()
                except Exception as exc:
                    self._preload_completed = False
                    self._preload_duration_sec = time.perf_counter() - started
                    self._preload_error = f"{type(exc).__name__}: {exc}"
                    raise
                self._object_approach_pipeline = pipeline
                self._preload_completed = True
                self._preload_duration_sec = time.perf_counter() - started
                self._preload_error = ""
            return pipeline

    def _preload_default_pipeline(self) -> None:
        output_root = Path(
            (config.get("paths") or {}).get("workspace_root", "/home/car/caragent_ws")
        ) / "perception_outputs" / "object_approach"
        try:
            self._pipeline(output_root)
        except Exception as exc:
            self._preload_error = f"{type(exc).__name__}: {exc}"

    def _preload_on_init(self) -> bool:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return bool(agent_cfg.get("object_approach_preload_on_init", True))

    def _preload_async(self) -> bool:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return bool(agent_cfg.get("object_approach_preload_async", True))

    def _start_preload(self, *, async_preload: bool) -> None:
        with self._pipeline_lock:
            if self._preload_started:
                return
            self._preload_started = True
        if async_preload:
            threading.Thread(
                target=self._preload_default_pipeline,
                name="caragent-object-approach-preload",
                daemon=True,
            ).start()
            return
        self._preload_default_pipeline()

    def _wait_for_background_object_preanalysis(
        self,
        target: str,
        *,
        task_cache_key: str = "",
        attempt_record: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        context = get_runtime_tool_context()
        current_task = context.get("current_task")
        if not _task_is_historical_object_preanalysis_candidate(current_task):
            if isinstance(attempt_record, dict):
                attempt_record["background_status"] = "not_applicable"
            return None
        try:
            task_id = int(current_task.get("task_id"))
        except Exception:
            return None

        shared_results = context.get("shared_background_results")
        if not isinstance(shared_results, dict):
            return None

        if task_cache_key and isinstance(attempt_record, dict):
            background_status = str(attempt_record.get("background_status") or "").lower()
            if background_status in {"completed_without_destination", "failed"}:
                self._emit_progress(
                    "object_preanalysis_waiting",
                    "skipped",
                    message=(
                        "ObjectApproach: object_preanalysis_waiting=skipped "
                        f"task={task_id} previous_status={background_status}"
                    ),
                )
                return None

        wait_timeout = self._background_wait_timeout_sec()
        wait_for_record_sec = min(wait_timeout, self._background_wait_for_record_sec())
        poll_sec = self._background_wait_poll_sec()
        start = time.monotonic()
        last_status = ""
        saw_background_record = False
        self._emit_progress(
            "object_preanalysis_waiting",
            "started",
            message=(
                "ObjectApproach: object_preanalysis_waiting=started "
                f"task={task_id} timeout={wait_timeout:.1f}s"
            ),
        )

        while True:
            record = shared_results.get(task_id)
            if isinstance(record, dict):
                saw_background_record = True
                status = str(record.get("status") or "").strip().lower()
                if status != last_status:
                    self._emit_progress(
                        "object_preanalysis_waiting",
                        status or "unknown",
                        message=(
                            "ObjectApproach: object_preanalysis_waiting="
                            f"{status or 'unknown'} task={task_id}"
                        ),
                    )
                    last_status = status
                if status == "completed":
                    if isinstance(attempt_record, dict):
                        attempt_record["background_record"] = deepcopy(record)
                    destination = record.get("recommended_destination")
                    if isinstance(destination, dict):
                        if isinstance(attempt_record, dict):
                            attempt_record["background_status"] = "completed"
                        return self._background_preanalysis_tool_result(
                            target=target,
                            task_id=task_id,
                            record=record,
                            destination=destination,
                        )
                    if isinstance(attempt_record, dict):
                        attempt_record["background_status"] = "completed_without_destination"
                        attempt_record["background_failure_reason"] = (
                            record.get("failure_reason")
                            or record.get("error")
                            or "completed_without_destination"
                        )
                    return None
                if status == "failed":
                    if isinstance(attempt_record, dict):
                        attempt_record["background_record"] = deepcopy(record)
                        attempt_record["background_status"] = "failed"
                        attempt_record["background_failure_reason"] = (
                            record.get("failure_reason") or record.get("error") or ""
                        )
                    self._emit_progress(
                        "object_preanalysis_failed",
                        "failed",
                        message=(
                            "ObjectApproach: object_preanalysis_failed=failed "
                            f"task={task_id} reason={record.get('failure_reason') or record.get('error') or ''}"
                        ),
                    )
                    return None

            elapsed = time.monotonic() - start
            if not saw_background_record and elapsed >= wait_for_record_sec:
                if isinstance(attempt_record, dict):
                    attempt_record["background_status"] = "no_record"
                    attempt_record["background_failure_reason"] = "background_job_record_not_created"
                    attempt_record["background_last_wait_elapsed_sec"] = elapsed
                self._emit_progress(
                    "object_preanalysis_waiting",
                    "no_record",
                    message=(
                        "ObjectApproach: object_preanalysis_waiting=no_record "
                        f"task={task_id} elapsed={elapsed:.1f}s"
                    ),
                )
                return None
            if elapsed >= wait_timeout:
                if isinstance(attempt_record, dict):
                    attempt_record["background_status"] = "timeout"
                    attempt_record["background_failure_reason"] = "background_job_watchdog_timeout"
                    attempt_record["background_last_wait_elapsed_sec"] = elapsed
                self._emit_progress(
                    "object_preanalysis_waiting",
                    "timeout",
                    message=(
                        "ObjectApproach: object_preanalysis_waiting=timeout "
                        f"task={task_id} elapsed={elapsed:.1f}s"
                    ),
                )
                return None
            time.sleep(poll_sec)

    def _background_preanalysis_tool_result(
        self,
        *,
        target: str,
        task_id: int,
        record: dict[str, Any],
        destination: dict[str, Any],
    ) -> dict[str, Any]:
        preanalysis = (
            record.get("object_preanalysis")
            if isinstance(record.get("object_preanalysis"), dict)
            else {}
        )
        self._emit_progress(
            "object_preanalysis_reused",
            "ok",
            message=f"ObjectApproach: object_preanalysis_reused=ok task={task_id}",
        )
        data = {
            "status": "ok",
            "summary": record.get("summary"),
            "destination": destination,
            "target_description": target,
            "source": "background_preanalysis",
            "background_task_id": task_id,
            "depth_backend": preanalysis.get("depth_backend"),
            "artifact_paths": _artifact_paths_from_preanalysis(preanalysis),
            "key_metrics": _key_metrics_from_preanalysis(preanalysis),
            "stages": [
                {"name": "object_preanalysis_waiting", "status": "completed"},
                {"name": "object_preanalysis_reused", "status": "ok"},
            ],
            "object_preanalysis": {
                key: preanalysis.get(key)
                for key in ("status", "summary_json", "output_dir", "depth_backend")
                if preanalysis.get(key) not in (None, "", [], {})
            },
        }
        position = destination.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            position = [0.0, 0.0, 0.0]
        summary = (
            "Reused historical object preanalysis for "
            f"'{target}' at x={float(position[0]):.2f}, "
            f"y={float(position[1]):.2f}."
        )
        return self.ok(
            summary,
            data=data,
            provenance={"source_type": "background_preanalysis"},
        )

    def _emit_progress(self, stage: str, status: str, *, message: str = "") -> None:
        text = message or f"ObjectApproach: {stage}={status}"
        context = get_runtime_tool_context()
        logger = context.get("logger")
        if logger is not None and hasattr(logger, "log_foreground"):
            try:
                logger.log_foreground(text)
            except Exception:
                pass
        print(text, flush=True)

    def _background_wait_timeout_sec(self) -> float:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return max(0.0, float(agent_cfg.get("object_preanalysis_wait_timeout_sec", 25.0)))

    def _background_wait_poll_sec(self) -> float:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return max(0.1, float(agent_cfg.get("object_preanalysis_wait_poll_sec", 0.5)))

    def _background_wait_for_record_sec(self) -> float:
        agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        return max(0.0, float(agent_cfg.get("object_preanalysis_wait_for_record_sec", 3.0)))

    def _current_image(self) -> PILImage.Image | None:
        if not hasattr(self.controller, "get_current_image"):
            return None
        image = self.controller.get_current_image()
        if image is None:
            return None
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        return None

    def _current_scan(self) -> Any:
        if not hasattr(self.controller, "get_current_scan"):
            return None
        return self.controller.get_current_scan()

    def _current_right_image(self) -> PILImage.Image | None:
        if not hasattr(self.controller, "get_current_right_image"):
            return None
        image = self.controller.get_current_right_image()
        if image is None:
            return None
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        return None
