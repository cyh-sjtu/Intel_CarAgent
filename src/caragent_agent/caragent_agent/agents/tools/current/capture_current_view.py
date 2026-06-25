"""Tool for saving the robot's current camera view as an artifact."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.config.config import config


def _workspace_root() -> Path:
    paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    workspace = paths.get("workspace") or config.get("workspace") or "/home/car/caragent_ws"
    return Path(str(workspace)).expanduser()


class CaptureCurrentViewTool(ToolBase):
    """Save the current live camera image for inspection, reporting, or evidence."""

    def __init__(self, controller: Any):
        super().__init__(
            name="capture_current_view",
            description="""
                Save the robot's current front camera image to a local artifact file.

                Use this when the task explicitly needs a photo/snapshot/evidence
                of what the robot sees now, for example after arriving at an
                inspection target, documenting task completion, or capturing the
                current scene for later user review. This tool only captures and
                saves the current view; it does not analyze the image and does
                not decide whether navigation is complete.

                Args:
                    note (str): Short reason for the capture, such as
                        "after arriving at the target" or "inspection evidence".
                    image_format (str): "jpg" or "png"; jpg is the default.

                Returns:
                    A compact artifact reference with image path, image_ref_id,
                    size, format, and note.
            """,
            capability_tags=("live_view", "artifact_capture", "background_unsafe"),
        )
        self.controller = controller

    def execute(self, note: str = "", image_format: str = "jpg") -> dict[str, Any]:
        if self.controller is None:
            return self.blocked(
                "Controller is unavailable, so the current view cannot be captured.",
                data={"note": note},
                error={"code": "controller_unavailable"},
                provenance={"source_type": "live_view"},
            )

        try:
            image = self.controller.get_current_image()
        except Exception as exc:
            return self.error_result(
                "Failed to read the current camera image from the controller.",
                data={"note": note},
                error={"code": "current_image_read_failed", "message": str(exc)},
                provenance={"source_type": "live_view"},
            )

        if image is None:
            return self.blocked(
                "Current camera image is unavailable, so no photo was saved.",
                data={"note": note},
                error={"code": "current_image_unavailable"},
                provenance={"source_type": "live_view"},
            )
        image_metadata = {}
        try:
            get_metadata = getattr(self.controller, "get_current_image_metadata", None)
            if callable(get_metadata):
                image_metadata = get_metadata() or {}
        except Exception:
            image_metadata = {}

        fmt = str(image_format or "jpg").strip().lower()
        if fmt in {"jpeg", "jpg"}:
            suffix = "jpg"
            pil_format = "JPEG"
        elif fmt == "png":
            suffix = "png"
            pil_format = "PNG"
        else:
            return self.error_result(
                "Unsupported image format for current-view capture.",
                data={"note": note, "image_format": image_format},
                error={"code": "unsupported_image_format", "supported": ["jpg", "png"]},
                provenance={"source_type": "live_view"},
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_ref_id = f"capture_{stamp}"
        output_dir = _workspace_root() / "perception_outputs" / "agent_captures"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{image_ref_id}.{suffix}"
        image.convert("RGB").save(image_path, format=pil_format)

        data = {
            "image_ref_id": image_ref_id,
            "path": str(image_path),
            "source": "current_view",
            "note": str(note or "").strip(),
            "image_format": suffix,
            "width": int(image.width),
            "height": int(image.height),
            "image_source": image_metadata,
        }
        return self.ok(
            (
                "Captured the current camera view."
                if image_metadata.get("source_type") != "simulated_keyframe_view"
                else "Captured a simulated current view from historical keyframe memory."
            ),
            data=data,
            provenance={
                **(image_metadata or {"source_type": "live_view"}),
                "artifact_path": str(image_path),
            },
        )
