"""Object-level target selection, localization, and approach goal planning."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import LaserScan

from caragent_agent.config.config import config
from caragent_agent.perception.fusion.live_scan_monodepth_validation import (
    DEFAULT_CALIB,
    DEFAULT_DEPTH_MODEL_DIR,
    DEFAULT_EXTR,
    DEFAULT_GROUNDING_MODEL_DIR,
    DEFAULT_GROUNDING_MODEL_ID,
    DEFAULT_SAM_DECODER_XML,
    DEFAULT_SAM_ENCODER_XML,
    DEFAULT_WORKSPACE,
    PipelineRunner,
    save_scan_npz,
)
from caragent_agent.perception.fusion.approach_goal_planner import (
    ApproachPlannerParams,
    base_xy_to_map,
    draw_approach_debug,
    grid_from_mapping,
    plan_approach_goal,
    pose_from_state,
)
from caragent_agent.perception.fusion.project_scan_fit_monodepth import (
    load_calib,
    load_extrinsics,
    load_optional_mask,
    rodrigues_xyz,
)
from caragent_agent.perception.fusion.stereo_mono_anchor_fusion import (
    compute_stereo_mono_guard,
    write_guard_payload,
)
from caragent_agent.perception.grounding.vlm_select_box import (
    build_parser as build_vlm_select_parser,
    run_selection,
)
from caragent_agent.utils.llm_handler import UnifiedLLMClient


DEFAULT_OUTPUT_ROOT = DEFAULT_WORKSPACE / "perception_outputs" / "object_approach"


def _json_from_text(text: str) -> dict[str, Any] | None:
    clean = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL | re.IGNORECASE)
    if fenced:
        clean = fenced.group(1).strip()
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        parsed = json.loads(clean)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _pil_to_bgr(image: PILImage.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _stamp_sec(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def _quat_to_yaw(q: list[float] | tuple[float, ...] | None) -> float | None:
    if not q or len(q) < 4:
        return None
    x, y, z, w = [float(v) for v in q[:4]]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _safe_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _metric_depth_path(fit_payload: dict[str, Any]) -> Path | None:
    path = (fit_payload.get("outputs") or {}).get("metric_depth_npy")
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def _paths_from_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    paths: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, (str, Path)):
            text = str(item)
            if "/" in text or text.endswith((".json", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".csv")):
                paths[str(key)] = text
    return paths


@dataclass
class ApproachPlannerConfig:
    stop_distance_m: float = 0.8
    already_close_tolerance_m: float = 0.15
    min_depth_m: float = 0.20
    max_depth_m: float = 15.0
    min_forward_goal_m: float = 0.10
    max_forward_goal_m: float = 2.5
    max_abs_bearing_deg: float = 65.0
    max_fit_p90_m: float = 0.70
    min_fit_inliers: int = 8
    min_valid_mask_depth_pixels: int = 40
    min_mask_depth_valid_ratio: float = 0.05


class ObjectApproachPipeline:
    """Headless pipeline used by the agent object approach tool."""

    def __init__(self, output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
        self.output_root = Path(output_root)
        self._runner: PipelineRunner | None = None

    def preload_models(self) -> None:
        preload_dir = self.output_root / "_preload"
        self._get_runner(preload_dir)

    async def plan_labels(
        self,
        *,
        image: PILImage.Image,
        target_description: str,
        grounding_query: str = "",
        vlm_query: str = "",
        sam_query: str = "",
    ) -> dict[str, Any]:
        target = str(target_description or "").strip()
        if grounding_query and vlm_query and sam_query:
            return {
                "status": "provided",
                "target_description": target,
                "grounding_query": grounding_query.strip(),
                "vlm_query": vlm_query.strip(),
                "sam_query": sam_query.strip(),
                "reason": "All labels were provided by caller.",
            }

        fallback = {
            "status": "fallback",
            "target_description": target,
            "grounding_query": (grounding_query or target).strip(),
            "vlm_query": (vlm_query or target).strip(),
            "sam_query": (sam_query or grounding_query or target).strip(),
            "reason": "Fallback labels derived from the target description.",
        }
        if not target:
            return fallback

        prompt = f"""
You generate perception labels for a mobile robot object approach tool.

User target description:
{target}

Return strict JSON only:
{{
  "grounding_query": "short open-vocabulary detection noun phrase, usually 1-4 words",
  "vlm_query": "detailed phrase that uniquely identifies the target in the current image",
  "sam_query": "short segmentation object label, usually same as grounding_query",
  "target_type": "static|dynamic|unknown",
  "reason": "short reason"
}}

Rules:
- GroundingDINO should receive a coarse object/category phrase, not a long relational sentence.
- VLM should receive the detailed user intent, including spatial or relational constraints.
- SAM should receive the coarse object/category phrase.
- Prefer English object phrases.
""".strip()
        try:
            from caragent_agent.io_adapters import image_to_data_url

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a robot perception label planner. Return only strict JSON.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            client = UnifiedLLMClient()
            model = str(config.get("vlm_model_analyse_images", "qwen3-vl-plus"))
            response = await client.chat_completion(model, messages)
            parsed = _json_from_text(response)
            if not parsed:
                fallback["raw_response"] = response
                return fallback
            plan = {
                "status": "ok",
                "target_description": target,
                "grounding_query": str(parsed.get("grounding_query") or grounding_query or target).strip(),
                "vlm_query": str(parsed.get("vlm_query") or vlm_query or target).strip(),
                "sam_query": str(parsed.get("sam_query") or sam_query or parsed.get("grounding_query") or target).strip(),
                "target_type": str(parsed.get("target_type") or "unknown").strip(),
                "reason": str(parsed.get("reason") or "").strip(),
                "raw_response": response,
            }
            if grounding_query:
                plan["grounding_query"] = grounding_query.strip()
            if vlm_query:
                plan["vlm_query"] = vlm_query.strip()
            if sam_query:
                plan["sam_query"] = sam_query.strip()
            return plan
        except Exception as exc:
            fallback["error"] = str(exc)
            return fallback

    def run(
        self,
        *,
        image: PILImage.Image,
        scan_msg: LaserScan | None = None,
        right_image: PILImage.Image | None = None,
        target_description: str,
        current_state: dict[str, Any] | None,
        grounding_query: str = "",
        vlm_query: str = "",
        sam_query: str = "",
        stop_distance_m: float = 0.8,
        depth_backend: str = "auto",
        dispatch: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = datetime.now().strftime("object_approach_%Y%m%d_%H%M%S")
        output_dir = self.output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_id = run_id
        status_path = output_dir / f"{sample_id}_status.json"
        progress_events: list[dict[str, Any]] = []

        def emit_progress(
            name: str,
            status: str,
            message: str = "",
            paths: dict[str, str] | None = None,
        ) -> None:
            event = {
                "name": name,
                "status": status,
                "message": message,
                "elapsed_sec": time.perf_counter() - started,
            }
            if paths:
                event["paths"] = paths
            progress_events.append(event)
            payload = {
                "run_id": run_id,
                "target_description": target_description,
                "output_dir": str(output_dir),
                "status_json": str(status_path),
                "current_stage": event,
                "events": progress_events,
            }
            try:
                status_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass
            if progress_callback is not None:
                try:
                    progress_callback(event)
                except Exception:
                    pass

        image_path = output_dir / f"{sample_id}.png"
        right_image_path = output_dir / f"{sample_id}_right.png"
        scan_path = output_dir / f"{sample_id}_scan.npz"
        emit_progress("snapshot", "running", "Saving current camera snapshot.")
        image.convert("RGB").save(image_path)
        if right_image is not None:
            right_image.convert("RGB").save(right_image_path)
        if scan_msg is not None:
            save_scan_npz(scan_msg, scan_path)
        emit_progress(
            "snapshot",
            "ok",
            "Saved current camera snapshot.",
            {"image": str(image_path), "right_image": str(right_image_path) if right_image is not None else ""},
        )
        backend = str(depth_backend or "auto").strip().lower()
        if backend == "auto":
            backend = "stereo_primary_mono_guard" if right_image is not None else "mono_relative_lidar"
        if backend not in {"stereo", "stereo_primary_mono_guard", "mono_relative_lidar"}:
            raise ValueError(f"Unsupported object approach depth backend: {depth_backend}")
        if backend in {"stereo", "stereo_primary_mono_guard"} and right_image is None:
            raise ValueError(f"{backend} object approach requires a right image")
        if backend == "mono_relative_lidar" and scan_msg is None:
            raise ValueError("mono_relative_lidar object approach requires LaserScan")

        emit_progress("label_plan", "running", "Preparing object labels.")
        if grounding_query and vlm_query and not sam_query:
            label_plan = {
                "status": "provided",
                "target_description": str(target_description or "").strip(),
                "grounding_query": str(grounding_query or "").strip(),
                "vlm_query": str(vlm_query or "").strip(),
                "sam_query": str(grounding_query or target_description or "").strip(),
                "reason": "Grounding and VLM labels were provided; SAM label was derived from grounding_query.",
            }
        else:
            label_plan = asyncio.run(
                self.plan_labels(
                    image=image,
                    target_description=target_description,
                    grounding_query=grounding_query,
                    vlm_query=vlm_query,
                    sam_query=sam_query,
                )
            )
        label_plan_path = output_dir / f"{sample_id}_label_plan.json"
        label_plan_path.write_text(json.dumps(label_plan, indent=2, ensure_ascii=False), encoding="utf-8")
        emit_progress(
            "label_plan",
            str(label_plan.get("status") or "ok"),
            str(label_plan.get("reason") or "Object labels are ready."),
            {"label_plan_json": str(label_plan_path)},
        )

        result: dict[str, Any] = {
            "status": "running",
            "run_id": run_id,
            "target_description": target_description,
            "output_dir": str(output_dir),
            "stages": [],
            "paths": {
                "output_dir": str(output_dir),
                "image": str(image_path),
                "label_plan_json": str(label_plan_path),
                "status_json": str(status_path),
            },
            "image": str(image_path),
            "label_plan": label_plan,
            "label_plan_json": str(label_plan_path),
            "status_json": str(status_path),
            "progress_events": progress_events,
            "depth_backend": backend,
            "dispatch_requested": bool(dispatch),
        }
        if right_image is not None:
            result["right_image"] = str(right_image_path)
            result["paths"]["right_image"] = str(right_image_path)
        if scan_msg is not None:
            result["scan"] = str(scan_path)
            result["paths"]["scan"] = str(scan_path)
        try:
            result["stages"].append({"name": "snapshot", "status": "ok", "paths": dict(result["paths"])})
            result["stages"].append({"name": "label_plan", "status": label_plan.get("status", "ok"), "paths": {"label_plan_json": str(label_plan_path)}})
            emit_progress("grounding_vlm_select", "running", "Running GroundingDINO detection and VLM box selection.")
            vlm_payload = self._run_vlm_selection(image_path, output_dir, label_plan)
            result["vlm_selection"] = vlm_payload
            vlm_paths = _paths_from_mapping(vlm_payload.get("paths"))
            result["paths"].update(vlm_paths)
            result["stages"].append({"name": "grounding_vlm_select", "status": (vlm_payload.get("selection") or {}).get("status", "unknown"), "paths": vlm_paths})
            emit_progress(
                "grounding_vlm_select",
                str((vlm_payload.get("selection") or {}).get("status") or "unknown"),
                "GroundingDINO and VLM box selection finished.",
                vlm_paths,
            )
            selection = vlm_payload.get("selection") or {}
            if selection.get("status") != "selected":
                approach = {
                    "status": "unreliable",
                    "reason": str(selection.get("status") or "vlm_selection_failed"),
                    "checks": {
                        "vlm_selection_status": selection.get("status"),
                        "vlm_selection_reason": selection.get("reason"),
                        "candidate_count": len(vlm_payload.get("candidates") or []),
                    },
                }
                approach_path = output_dir / f"{sample_id}_approach_goal.json"
                approach_path.write_text(json.dumps(approach, indent=2, ensure_ascii=False), encoding="utf-8")
                result["approach"] = approach
                result["approach_goal_json"] = str(approach_path)
                result["paths"]["approach_goal_json"] = str(approach_path)
                result["status"] = "unreliable"
                result["elapsed_sec"] = time.perf_counter() - started
                result["progress_events"] = progress_events
                summary_path = output_dir / f"{sample_id}_summary.json"
                result["summary_json"] = str(summary_path)
                result["paths"]["summary_json"] = str(summary_path)
                summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                emit_progress("complete", "unreliable", str(approach.get("reason") or "Object selection was unreliable."), {"summary_json": str(summary_path)})
                return result

            selected_grounding_json = self._write_selected_grounding_json(vlm_payload, output_dir, sample_id)
            result["selected_grounding_json"] = str(selected_grounding_json)
            result["paths"]["selected_grounding_json"] = str(selected_grounding_json)

            runner = self._get_runner(output_dir)
            emit_progress("sam_segmentation", "running", "Running SAM segmentation for selected object.")
            runner._run_sam(sample_id, selected_grounding_json, str(label_plan.get("sam_query") or target_description))
            segmentation_json = runner._segmentation_json_path(sample_id)
            seg_payload = json.loads(segmentation_json.read_text(encoding="utf-8"))
            sam_paths = {
                "segmentation_json": str(segmentation_json),
                "mask": str(seg_payload.get("mask_path") or ""),
                "mask_overlay": str(seg_payload.get("overlay_path") or ""),
            }
            result["paths"].update({k: v for k, v in sam_paths.items() if v})
            result["stages"].append({"name": "sam_segmentation", "status": "ok", "paths": {k: v for k, v in sam_paths.items() if v}})
            emit_progress("sam_segmentation", "ok", "SAM segmentation finished.", {k: v for k, v in sam_paths.items() if v})
            result.update(
                {
                    "segmentation_json": str(segmentation_json),
                    "segmentation_overlay": seg_payload.get("overlay_path"),
                }
            )

            if backend in {"stereo", "stereo_primary_mono_guard"}:
                emit_progress("stereo_object_depth", "running", "Estimating object depth from stereo images.")
                runner._run_stereo(sample_id, image_path, right_image_path, segmentation_json)
                stereo_json = runner._stereo_json_path(sample_id)
                stereo_payload = json.loads(stereo_json.read_text(encoding="utf-8"))
                stereo_paths = {"stereo_json": str(stereo_json)}
                stereo_paths.update(_paths_from_mapping(stereo_payload))
                result["paths"].update(stereo_paths)
                result["stages"].append({"name": "stereo_object_depth", "status": "ok", "paths": stereo_paths})
                emit_progress("stereo_object_depth", "ok", "Stereo object depth finished.", stereo_paths)
                result["stereo"] = stereo_payload
                object_base = ((stereo_payload.get("object_base") or {}).get("median_xyz_m") or [])
                mono_guard_payload: dict[str, Any] | None = None
                mono_guard_json: Path | None = None
                if backend == "stereo_primary_mono_guard":
                    emit_progress("mono_guard_depth", "running", "Running mono relative depth guard for stereo.")
                    mono_depth, _, _ = runner._run_depth(
                        sample_id,
                        image_path,
                        runner.get_relative_depth_model(),
                        runner.args.depth_model_dir,
                        runner.args.depth_device,
                        output_suffix="mono_guard",
                    )
                    mono_depth_npy = output_dir / f"{sample_id}_mono_guard_depth.npy"
                    mono_depth_paths = {
                        "mono_guard_depth_npy": str(mono_depth_npy),
                        "mono_guard_depth_gray": str(output_dir / f"{sample_id}_mono_guard_depth_gray.png"),
                        "mono_guard_depth_color": str(output_dir / f"{sample_id}_mono_guard_depth_color.png"),
                        "mono_guard_depth_json": str(output_dir / f"{sample_id}_mono_guard_depth.json"),
                    }
                    result["paths"].update(mono_depth_paths)
                    result["stages"].append({"name": "mono_guard_depth", "status": "ok", "paths": mono_depth_paths})
                    emit_progress("mono_guard_depth", "ok", "Mono relative depth guard image finished.", mono_depth_paths)
                    emit_progress("stereo_mono_guard", "running", "Checking stereo depth with mono-relative anchors.")
                    try:
                        mono_guard_payload = compute_stereo_mono_guard(
                            stereo_payload=stereo_payload,
                            mono_depth=mono_depth,
                            mono_depth_source=mono_depth_npy,
                        )
                    except Exception as exc:
                        mono_guard_payload = {
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    mono_guard_json = output_dir / f"{sample_id}_stereo_mono_guard.json"
                    write_guard_payload(mono_guard_json, mono_guard_payload)
                    guard_paths = {"mono_guard_json": str(mono_guard_json)}
                    result["paths"].update(guard_paths)
                    result["stages"].append(
                        {
                            "name": "stereo_mono_guard",
                            "status": str(mono_guard_payload.get("status") or "unknown"),
                            "paths": guard_paths,
                        }
                    )
                    emit_progress(
                        "stereo_mono_guard",
                        str(mono_guard_payload.get("status") or "unknown"),
                        str(mono_guard_payload.get("reason") or "Stereo/mono guard finished."),
                        guard_paths,
                    )
                    result["mono_guard"] = mono_guard_payload
                    guarded_base = self._object_base_xyz_from_mono_guard(
                        stereo_object_base_xyz=object_base,
                        guard_payload=mono_guard_payload,
                    )
                    if guarded_base is not None:
                        object_base = guarded_base
                emit_progress("approach_goal", "running", "Planning object approach goal.")
                checks = {
                    "depth_backend": backend,
                    "stereo_valid_ratio": (stereo_payload.get("mask") or {}).get("valid_ratio"),
                    "stereo_valid_pixels": (stereo_payload.get("mask") or {}).get("valid_stereo_pixels"),
                    "stereo_json": str(stereo_json),
                }
                if mono_guard_payload is not None:
                    mono_guard_fit = mono_guard_payload.get("fit") if isinstance(mono_guard_payload.get("fit"), dict) else {}
                    mono_guard_fused = (
                        mono_guard_payload.get("fused_base_x_m")
                        if isinstance(mono_guard_payload.get("fused_base_x_m"), dict)
                        else {}
                    )
                    checks.update(
                        {
                            "mono_guard_status": mono_guard_payload.get("status"),
                            "mono_guard_selected_source": mono_guard_payload.get("selected_source"),
                            "mono_guard_reason": mono_guard_payload.get("reason"),
                            "mono_guard_selected_depth_m": mono_guard_payload.get("selected_depth_m"),
                            "mono_guard_fused_median_m": mono_guard_fused.get("median"),
                            "mono_guard_correction_delta_m": mono_guard_payload.get("correction_delta_m"),
                            "mono_guard_fused_iqr_m": mono_guard_payload.get("fused_iqr_m"),
                            "mono_guard_anchor_count": mono_guard_payload.get("anchor_count"),
                            "mono_guard_fit_rmse_m": mono_guard_fit.get("rmse_keep"),
                            "mono_guard_json": str(mono_guard_json) if mono_guard_json is not None else None,
                        }
                    )
                approach = self._compute_approach_from_base_xyz(
                    object_base_xyz=object_base,
                    current_state=current_state or {},
                    stop_distance_m=stop_distance_m,
                    output_dir=output_dir,
                    sample_id=sample_id,
                    checks=checks,
                )
            else:
                emit_progress("relative_depth", "running", "Running relative monocular depth.")
                runner._run_depth(
                    sample_id,
                    image_path,
                    runner.get_relative_depth_model(),
                    runner.args.depth_model_dir,
                    runner.args.depth_device,
                )
                depth_npy = output_dir / f"{sample_id}_depth.npy"
                depth_paths = {
                    "relative_depth_npy": str(depth_npy),
                    "relative_depth_gray": str(output_dir / f"{sample_id}_depth_gray.png"),
                    "relative_depth_color": str(output_dir / f"{sample_id}_depth_color.png"),
                    "relative_depth_json": str(output_dir / f"{sample_id}_depth.json"),
                }
                result["paths"].update(depth_paths)
                result["stages"].append({"name": "relative_depth", "status": "ok", "paths": depth_paths})
                emit_progress("relative_depth", "ok", "Relative monocular depth finished.", depth_paths)
                emit_progress("scan_monodepth_fit", "running", "Fitting relative depth scale with LaserScan.")
                runner._run_fit(sample_id, image_path, scan_path, depth_npy, segmentation_json)
                fit_json = runner.fit_output_dir / f"{sample_id}_scan_monodepth_fit.json"
                fit_payload = json.loads(fit_json.read_text(encoding="utf-8"))
                fit_paths = {"fit_json": str(fit_json)}
                fit_paths.update(_paths_from_mapping(fit_payload.get("outputs")))
                result["paths"].update(fit_paths)
                result["stages"].append({"name": "scan_monodepth_fit", "status": "ok", "paths": fit_paths})
                emit_progress("scan_monodepth_fit", "ok", "Relative-depth/LaserScan fit finished.", fit_paths)
                result.update({"fit_json": str(fit_json), "fit": fit_payload})
                emit_progress("approach_goal", "running", "Planning object approach goal.")
                approach = self._compute_approach(
                    image_path=image_path,
                    segmentation_json=segmentation_json,
                    fit_payload=fit_payload,
                    current_state=current_state or {},
                    stop_distance_m=stop_distance_m,
                    output_dir=output_dir,
                    sample_id=sample_id,
                )
            approach_path = output_dir / f"{sample_id}_approach_goal.json"
            approach_path.write_text(json.dumps(approach, indent=2, ensure_ascii=False), encoding="utf-8")
            result["approach"] = approach
            result["approach_goal_json"] = str(approach_path)
            result["paths"]["approach_goal_json"] = str(approach_path)
            approach_paths = {"approach_goal_json": str(approach_path)}
            if approach.get("debug_png"):
                approach_paths["approach_debug_png"] = str(approach.get("debug_png"))
                result["paths"]["approach_debug_png"] = str(approach.get("debug_png"))
            result["stages"].append({"name": "approach_goal", "status": approach.get("status", "unknown"), "paths": approach_paths})
            emit_progress(
                "approach_goal",
                str(approach.get("status") or "unknown"),
                str(approach.get("reason") or "Approach goal planning finished."),
                approach_paths,
            )
            result["status"] = approach.get("status", "failed")
            result["elapsed_sec"] = time.perf_counter() - started
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            result["stages"].append({"name": "exception", "status": "failed", "error": str(exc)})
            result["elapsed_sec"] = time.perf_counter() - started
            emit_progress("exception", "failed", str(exc))

        summary_path = output_dir / f"{sample_id}_summary.json"
        result["summary_json"] = str(summary_path)
        result["paths"]["summary_json"] = str(summary_path)
        result["progress_events"] = progress_events
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        emit_progress("complete", str(result.get("status") or "unknown"), "Object approach pipeline finished.", {"summary_json": str(summary_path)})
        return result

    def _get_runner(self, output_dir: Path) -> PipelineRunner:
        runner = self._runner
        if runner is None:
            runner = PipelineRunner(self._runner_args(output_dir), lambda text: None)
            self._runner = runner
        else:
            runner.output_dir = output_dir.resolve()
            runner.samples_dir = runner.output_dir / "samples"
            runner.command_logs_dir = runner.output_dir / "command_logs"
            runner.fit_output_dir = runner.output_dir / "scan_monodepth_fit"
            runner.stereo_output_dir = runner.output_dir / "stereo_object_depth"
            runner.absolute_output_dir = runner.output_dir / "mono_absolute_depth"
            runner.summary_csv = runner.output_dir / "validation_results.csv"
            runner.summary_jsonl = runner.output_dir / "validation_results.jsonl"
            for path in [
                runner.samples_dir,
                runner.command_logs_dir,
                runner.fit_output_dir,
                runner.stereo_output_dir,
                runner.absolute_output_dir,
            ]:
                path.mkdir(parents=True, exist_ok=True)
        return runner

    def _run_vlm_selection(self, image_path: Path, output_dir: Path, label_plan: dict[str, Any]) -> dict[str, Any]:
        runner = self._get_runner(output_dir)
        return self._run_vlm_selection_with_runner(runner, image_path, output_dir, label_plan)

    def _run_vlm_selection_with_runner(
        self,
        runner: PipelineRunner,
        image_path: Path,
        output_dir: Path,
        label_plan: dict[str, Any],
    ) -> dict[str, Any]:
        from caragent_agent.perception.grounding.vlm_select_box import (
            ask_vlm_to_select,
            draw_candidate_overlay,
            draw_selected_overlay,
            make_crop_grid,
            prepare_candidates,
        )

        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        grounding_png = output_dir / f"{stem}_grounding_openvino.png"
        grounding_json = output_dir / f"{stem}_grounding_openvino.json"
        candidates_png = output_dir / f"{stem}_vlm_candidates.png"
        crops_png = output_dir / f"{stem}_vlm_crops.png"
        selected_png = output_dir / f"{stem}_vlm_selected.png"
        selection_json = output_dir / f"{stem}_vlm_box_selection.json"

        vlm_query = str(label_plan.get("vlm_query") or label_plan.get("target_description") or "").strip()
        grounding_query = str(label_plan.get("grounding_query") or vlm_query).strip()
        if grounding_query and grounding_query[-1] not in ".。!?！？":
            grounding_query = f"{grounding_query} ."

        if runner.grounding_model is None:
            raise RuntimeError("GroundingDINO model is not loaded for object approach.")
        grounding = runner.grounding_model.detect(
            image_path=image_path,
            text_prompt=grounding_query,
            box_threshold=0.25,
            text_threshold=0.20,
        )
        image = PILImage.open(image_path).convert("RGB")
        all_candidates = prepare_candidates(
            grounding.get("detections", []),
            image_size=image.size,
            max_candidates=999,
        )
        candidates = prepare_candidates(
            grounding.get("detections", []),
            image_size=image.size,
            max_candidates=8,
        )
        grounding["overlay_path"] = str(grounding_png)
        grounding["vlm_candidates_path"] = str(candidates_png)
        grounding["candidate_crops_path"] = str(crops_png)
        grounding_json.write_text(json.dumps(grounding, indent=2, ensure_ascii=False), encoding="utf-8")
        draw_candidate_overlay(image_path, all_candidates, grounding_png)
        draw_candidate_overlay(image_path, candidates, candidates_png)
        make_crop_grid(image_path, candidates, crops_png)

        raw_vlm_response = ""
        if not candidates:
            selection = {
                "status": "no_detection",
                "selected_id": None,
                "confidence": 0.0,
                "reason": "GroundingDINO returned no candidate boxes.",
                "query": vlm_query,
            }
        else:
            selection, raw_vlm_response = asyncio.run(
                ask_vlm_to_select(
                    vlm_query=vlm_query,
                    grounding_query=grounding_query,
                    image_size=image.size,
                    candidates=candidates,
                    candidate_overlay_path=candidates_png,
                    crop_grid_path=crops_png,
                    model=str(config.get("vlm_model_analyse_images", "qwen3-vl-plus")),
                )
            )
        draw_selected_overlay(image_path, candidates, selection, selected_png)

        payload = {
            "image": str(image_path),
            "query": vlm_query,
            "vlm_query": vlm_query,
            "grounding_query": grounding_query,
            "selection": selection,
            "candidates": candidates,
            "vlm_model": str(config.get("vlm_model_analyse_images", "qwen3-vl-plus")),
            "vlm_raw_response": raw_vlm_response,
            "paths": {
                "grounding_json": str(grounding_json),
                "grounding_overlay": str(grounding_png),
                "vlm_candidates": str(candidates_png),
                "vlm_crops": str(crops_png),
                "vlm_selected": str(selected_png),
                "selection_json": str(selection_json),
            },
        }
        selection_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _write_selected_grounding_json(self, vlm_payload: dict[str, Any], output_dir: Path, sample_id: str) -> Path:
        selection = vlm_payload.get("selection") or {}
        if selection.get("status") != "selected":
            raise RuntimeError(f"VLM box selection did not select a box: {selection.get('status')}")
        selected_id = int(selection.get("selected_id"))
        candidates = vlm_payload.get("candidates") or []
        selected = next((item for item in candidates if int(item.get("id", -1)) == selected_id), None)
        if selected is None:
            raise RuntimeError(f"Selected candidate #{selected_id} is missing from VLM candidates.")

        grounding_json_path = Path((vlm_payload.get("paths") or {}).get("grounding_json", ""))
        grounding_payload = json.loads(grounding_json_path.read_text(encoding="utf-8"))
        selected_detection = {
            "label": selected.get("label") or "object",
            "score": float(selected.get("score") or 0.0),
            "box": [float(v) for v in selected.get("box") or selected.get("box_int")],
            "box_int": [int(v) for v in selected.get("box_int")],
            "vlm_selected_id": selected_id,
            "vlm_selection": selection,
        }
        grounding_payload["detections"] = [selected_detection]
        grounding_payload["selected_by_vlm"] = {
            "selection": selection,
            "candidate": selected,
            "source_vlm_selection_json": (vlm_payload.get("paths") or {}).get("selection_json"),
        }
        output_path = output_dir / f"{sample_id}_selected_grounding_openvino.json"
        output_path.write_text(json.dumps(grounding_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _runner_args(self, output_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            workspace=DEFAULT_WORKSPACE,
            output_dir=output_dir,
            calib_file=DEFAULT_CALIB,
            extrinsics_json=DEFAULT_EXTR,
            grounding_model_dir=DEFAULT_GROUNDING_MODEL_DIR,
            grounding_model_id=DEFAULT_GROUNDING_MODEL_ID,
            grounding_device="GPU",
            depth_model_dir=DEFAULT_DEPTH_MODEL_DIR,
            depth_device="GPU",
            absolute_depth_model_dir=DEFAULT_WORKSPACE / "models" / "depth_anything_v2_metric_indoor_small_openvino",
            absolute_depth_device="GPU",
            sam_device="GPU",
            sam_encoder_device="GPU",
            sam_decoder_device="CPU",
            sam_encoder_xml=DEFAULT_SAM_ENCODER_XML,
            sam_decoder_xml=DEFAULT_SAM_DECODER_XML,
            enable_stereo_preview=False,
            stereo_num_disparities=96,
            stereo_block_size=5,
            stereo_min_depth=0.15,
            stereo_max_depth=8.0,
            selection_p90_tolerance=0.10,
            mask_lidar_support_check=True,
            min_mask_lidar_points=2,
            min_mask_lidar_density=0.035,
            skip_grounding_model=False,
        )

    def _compute_approach(
        self,
        *,
        image_path: Path,
        segmentation_json: Path,
        fit_payload: dict[str, Any],
        current_state: dict[str, Any],
        stop_distance_m: float,
        output_dir: Path,
        sample_id: str,
    ) -> dict[str, Any]:
        cfg = ApproachPlannerConfig(stop_distance_m=max(0.2, float(stop_distance_m or 0.8)))
        image = PILImage.open(image_path).convert("RGB")
        mtx, dist = load_calib(DEFAULT_CALIB)
        extrinsics, extrinsics_source = load_extrinsics(DEFAULT_EXTR)
        mask = load_optional_mask(segmentation_json, image.size)
        metric_path = _metric_depth_path(fit_payload)
        if mask is None or metric_path is None:
            return {"status": "unreliable", "reason": "missing_mask_or_metric_depth"}
        metric_depth = np.load(metric_path).astype(np.float32)
        if metric_depth.shape != mask.shape:
            metric_depth = cv2.resize(metric_depth, image.size, interpolation=cv2.INTER_LINEAR)

        stats = fit_payload.get("object_mask_metric_depth_m") or {}
        depth_m = _safe_float(stats.get("p10")) or _safe_float(stats.get("p05")) or _safe_float(stats.get("median"))
        selected_fit = fit_payload.get("selected_fit") or {}
        fit_p90 = _safe_float(selected_fit.get("p90_abs_error_m"))
        fit_inliers = int(selected_fit.get("inlier_count") or 0)
        valid_values = metric_depth[mask]
        valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0)]
        valid_ratio = float(len(valid_values) / max(1, int(np.count_nonzero(mask))))

        checks = {
            "depth_m": depth_m,
            "valid_mask_depth_pixels": int(len(valid_values)),
            "valid_mask_depth_ratio": valid_ratio,
            "fit_p90_abs_error_m": fit_p90,
            "fit_inliers": fit_inliers,
        }
        if depth_m is None or depth_m < cfg.min_depth_m or depth_m > cfg.max_depth_m:
            return {"status": "unreliable", "reason": "invalid_depth", "checks": checks}
        if len(valid_values) < cfg.min_valid_mask_depth_pixels or valid_ratio < cfg.min_mask_depth_valid_ratio:
            return {"status": "unreliable", "reason": "too_few_valid_mask_depth_pixels", "checks": checks}
        if fit_inliers < cfg.min_fit_inliers:
            return {"status": "unreliable", "reason": "too_few_fit_inliers", "checks": checks}
        mask_support = fit_payload.get("object_mask_lidar_support") or {}
        if mask_support.get("enabled") and not mask_support.get("has_support"):
            checks["mask_lidar_support"] = mask_support
            return {"status": "unreliable", "reason": "target_mask_lacks_projected_lidar_support", "checks": checks}
        if fit_p90 is not None and fit_p90 > cfg.max_fit_p90_m:
            return {"status": "unreliable", "reason": "fit_residual_too_large", "checks": checks}

        u, v = self._select_representative_pixel(mask, metric_depth, depth_m)
        object_base = self._pixel_depth_to_base(u, v, depth_m, mtx, dist, extrinsics)
        object_xy = np.asarray(object_base[:2], dtype=np.float64)
        range_xy = float(np.linalg.norm(object_xy))
        bearing = math.atan2(float(object_xy[1]), float(object_xy[0]))
        checks.update(
            {
                "representative_pixel": [float(u), float(v)],
                "object_base_xyz_m": [float(x) for x in object_base],
                "object_base_range_xy_m": range_xy,
                "object_bearing_deg": math.degrees(bearing),
                "extrinsics_source": extrinsics_source,
            }
        )
        if object_base[0] <= 0:
            return {"status": "unreliable", "reason": "target_not_in_front", "checks": checks}
        if abs(math.degrees(bearing)) > cfg.max_abs_bearing_deg:
            return {"status": "unreliable", "reason": "target_bearing_too_large", "checks": checks}
        return self._compute_approach_from_base_xyz(
            object_base_xyz=[float(x) for x in object_base],
            current_state=current_state,
            stop_distance_m=stop_distance_m,
            output_dir=output_dir,
            sample_id=sample_id,
            checks=checks,
        )

    def _compute_approach_from_base_xyz(
        self,
        *,
        object_base_xyz: Any,
        current_state: dict[str, Any],
        stop_distance_m: float,
        output_dir: Path,
        sample_id: str,
        checks: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = ApproachPlannerConfig(stop_distance_m=max(0.2, float(stop_distance_m or 0.8)))
        if not isinstance(object_base_xyz, (list, tuple)) or len(object_base_xyz) < 2:
            return {"status": "unreliable", "reason": "missing_object_base_xyz", "checks": checks}
        object_base = np.asarray([float(object_base_xyz[0]), float(object_base_xyz[1])], dtype=np.float64)
        range_xy = float(np.linalg.norm(object_base))
        bearing = math.atan2(float(object_base[1]), float(object_base[0]))
        checks.update(
            {
                "object_base_xyz_m": [float(v) for v in object_base_xyz[:3]],
                "object_base_range_xy_m": range_xy,
                "object_bearing_deg": math.degrees(bearing),
            }
        )
        if not math.isfinite(range_xy) or range_xy <= cfg.min_depth_m or range_xy > cfg.max_depth_m:
            return {"status": "unreliable", "reason": "invalid_object_range", "checks": checks}
        if object_base[0] <= 0:
            return {"status": "unreliable", "reason": "target_not_in_front", "checks": checks}
        if abs(math.degrees(bearing)) > cfg.max_abs_bearing_deg:
            return {"status": "unreliable", "reason": "target_bearing_too_large", "checks": checks}
        if range_xy <= cfg.stop_distance_m + cfg.already_close_tolerance_m:
            return {"status": "already_close", "reason": "target_already_within_stop_distance", "checks": checks}

        robot_pose = pose_from_state(current_state)
        if robot_pose is None:
            return {"status": "unreliable", "reason": "current_pose_unavailable", "checks": checks}

        object_map_xy = base_xy_to_map(object_base, robot_pose)
        grid = grid_from_mapping((current_state or {}).get("occupancy_grid") or (current_state or {}).get("map"))
        params = ApproachPlannerParams(
            stop_distance_m=cfg.stop_distance_m,
            min_stop_distance_m=max(0.45, cfg.stop_distance_m - 0.20),
            max_stop_distance_m=max(cfg.stop_distance_m + 0.35, cfg.stop_distance_m),
        )
        goal = plan_approach_goal(
            robot_pose=robot_pose,
            object_map_xy=object_map_xy,
            grid=grid,
            params=params,
        )
        debug_path = output_dir / f"{sample_id}_approach_debug.png"
        draw_approach_debug(
            output_path=debug_path,
            robot_pose=robot_pose,
            object_map_xy=object_map_xy,
            goal=goal,
            grid=grid,
            params=params,
        )
        map_goal = {
            "status": "ok" if goal.get("status") in {"ok", "degraded"} else "unavailable",
            "position": goal.get("position"),
            "yaw_rad": goal.get("yaw_rad"),
            "yaw_deg": goal.get("yaw_deg"),
            "current_pose_source": current_state.get("source") if isinstance(current_state, dict) else None,
        }
        return {
            "status": goal.get("status", "unreliable"),
            "reason": goal.get("reason") or "approach_goal_computed",
            "stop_distance_m": cfg.stop_distance_m,
            "object_base_xyz_m": [float(v) for v in object_base_xyz[:3]],
            "object_map_xy_m": [float(object_map_xy[0]), float(object_map_xy[1])],
            "map_goal": map_goal,
            "planner": goal,
            "checks": checks,
            "debug_png": str(debug_path),
        }

    def _object_base_xyz_from_mono_guard(
        self,
        *,
        stereo_object_base_xyz: Any,
        guard_payload: dict[str, Any] | None,
    ) -> list[float] | None:
        """Return a guarded base-frame object point when mono guard overrides stereo.

        The guard is deliberately conservative: if it keeps stereo or fails, the
        caller should use the original stereo median point unchanged.
        """

        if not isinstance(guard_payload, dict):
            return None
        if str(guard_payload.get("status") or "").lower() != "ok":
            return None
        if str(guard_payload.get("selected_source") or "") != "mono_guard":
            return None
        if not isinstance(stereo_object_base_xyz, (list, tuple)) or len(stereo_object_base_xyz) < 2:
            return None
        try:
            stereo_x = float(stereo_object_base_xyz[0])
            stereo_y = float(stereo_object_base_xyz[1])
            selected_x = float(guard_payload.get("selected_depth_m"))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(stereo_x) and math.isfinite(stereo_y) and math.isfinite(selected_x)):
            return None
        if stereo_x <= 0.05 or selected_x <= 0.05:
            return None
        lateral_ratio = stereo_y / stereo_x
        guarded = [
            float(selected_x),
            float(selected_x * lateral_ratio),
            float(stereo_object_base_xyz[2]) if len(stereo_object_base_xyz) >= 3 else 0.0,
        ]
        guard_payload["guarded_base_xyz_m"] = guarded
        guard_payload["guarded_lateral_ratio_source"] = "stereo_object_base_median"
        return guarded

    def _select_representative_pixel(self, mask: np.ndarray, metric_depth: np.ndarray, depth_m: float) -> tuple[float, float]:
        valid = mask & np.isfinite(metric_depth) & (metric_depth > 0)
        if not np.any(valid):
            ys, xs = np.where(mask)
            return float(np.median(xs)), float(np.median(ys))
        values = metric_depth[valid]
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 25))
        near = valid & (metric_depth >= min(lo, depth_m)) & (metric_depth <= max(hi, depth_m))
        if np.count_nonzero(near) < 10:
            near = valid
        ys, xs = np.where(near)
        return float(np.median(xs)), float(np.median(ys))

    def _pixel_depth_to_base(
        self,
        u: float,
        v: float,
        depth_m: float,
        mtx: np.ndarray,
        dist: np.ndarray,
        extrinsics: Any,
    ) -> np.ndarray:
        uv = np.asarray([[[float(u), float(v)]]], dtype=np.float64)
        norm = cv2.undistortPoints(uv, mtx, dist).reshape(2)
        z = float(depth_m)
        point_opt = np.array([norm[0] * z, norm[1] * z, z], dtype=np.float64)
        point_camera_project = np.array([point_opt[2], -point_opt[0], -point_opt[1]], dtype=np.float64)
        r_camera_to_base = rodrigues_xyz(
            extrinsics.camera_roll_rad,
            extrinsics.camera_pitch_rad,
            extrinsics.camera_yaw_rad,
        )
        camera_t = np.array(
            [extrinsics.camera_x_m, extrinsics.camera_y_m, extrinsics.camera_z_m],
            dtype=np.float64,
        )
        return camera_t + point_camera_project @ r_camera_to_base.T

    def _base_goal_to_map(
        self,
        goal_base_xy: np.ndarray,
        object_base_xy: np.ndarray,
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        position = current_state.get("position") if isinstance(current_state, dict) else None
        orientation = current_state.get("orientation") if isinstance(current_state, dict) else None
        yaw = _quat_to_yaw(orientation)
        if not isinstance(position, (list, tuple)) or len(position) < 2 or yaw is None:
            return {"status": "unavailable", "current_state": current_state}
        c = math.cos(yaw)
        s = math.sin(yaw)
        gx = float(position[0]) + c * float(goal_base_xy[0]) - s * float(goal_base_xy[1])
        gy = float(position[1]) + s * float(goal_base_xy[0]) + c * float(goal_base_xy[1])
        face_vec = object_base_xy - goal_base_xy
        relative_yaw = math.atan2(float(face_vec[1]), float(face_vec[0]))
        map_yaw = _normalize_angle(yaw + relative_yaw)
        return {
            "status": "ok",
            "position": [gx, gy, 0.0],
            "yaw_rad": map_yaw,
            "yaw_deg": math.degrees(map_yaw),
            "current_pose_source": current_state.get("source"),
        }


def run_object_approach_from_snapshot(
    *,
    image: PILImage.Image,
    scan_msg: LaserScan | None = None,
    right_image: PILImage.Image | None = None,
    target_description: str,
    current_state: dict[str, Any] | None,
    grounding_query: str = "",
    vlm_query: str = "",
    sam_query: str = "",
    stop_distance_m: float = 0.8,
    depth_backend: str = "auto",
    dispatch: bool = True,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    return ObjectApproachPipeline(output_root=output_root).run(
        image=image,
        scan_msg=scan_msg,
        right_image=right_image,
        target_description=target_description,
        current_state=current_state,
        grounding_query=grounding_query,
        vlm_query=vlm_query,
        sam_query=sam_query,
        stop_distance_m=stop_distance_m,
        depth_backend=depth_backend,
        dispatch=dispatch,
    )


__all__ = ["ObjectApproachPipeline", "run_object_approach_from_snapshot"]
