"""Semantic object depth tools are disabled until CarAgent has depth memory."""

from __future__ import annotations

from caragent_agent.agents.tools.base.tool_base import ToolBase


class _DepthUnavailableTool(ToolBase):
    def execute(self, *args, **kwargs):
        del args, kwargs
        return self.blocked(
            "Semantic object 3D analysis is unavailable because CarAgent v1 scene memory has no depth images.",
            error={
                "code": "depth_unavailable",
                "message": "Use keyframe image analysis or scene-memory search instead.",
            },
            provenance={"source_type": "scene_memory"},
        )


class AnalyseWithObjectsOnSelectedKF(_DepthUnavailableTool):
    def __init__(self):
        super().__init__(
            name="analyse_objects_on_selected_keyframes",
            description="Unavailable CarAgent v1 semantic object 3D analysis tool.",
            capability_tags=("scene_memory_search", "background_safe"),
        )


class AnalyseWithObjectsTool(_DepthUnavailableTool):
    def __init__(self):
        super().__init__(
            name="analyse_objects",
            description="Unavailable CarAgent v1 semantic object 3D analysis tool.",
            capability_tags=("scene_memory_search", "background_safe"),
        )
