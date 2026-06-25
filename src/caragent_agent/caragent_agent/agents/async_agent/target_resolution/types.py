"""Small data shapes for semantic target resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict


TargetKind = Literal["keyframe", "object", "local_structure"]
TargetSource = Literal[
    "scene_memory",
    "current_view",
    "attached_image",
    "arrived_scene",
    "upstream_result",
    "session_memory",
    "explicit",
    "unknown",
]
ResolutionStatus = Literal["resolved", "needs_observation", "ambiguous", "failed", "unsupported"]
AnchorType = Literal["keyframe", "position"]
ResolutionStage = Literal[
    "semantic_keyframe",
    "attached_image_keyframe",
    "attached_image_object_staging",
    "current_view_object",
    "arrived_scene_object",
    "session_anchor_reuse",
    "unsupported",
]


class TargetDraft(TypedDict):
    """Planner-provided semantic target, before resolver policy is applied."""

    target_type: str
    description: str
    source_hint: TargetSource
    raw_target: dict[str, Any]


class TargetRef(TypedDict):
    """Normalized target reference consumed by resolver policy."""

    kind: TargetKind
    source: TargetSource
    description: str
    query: NotRequired[str]
    inputs_from: NotRequired[Any]
    stop_distance_m: NotRequired[float]
    image_refs: NotRequired[Any]
    image_focus: NotRequired[str]
    target_kind: NotRequired[str]
    selection_policy: NotRequired[Any]


class Evidence(TypedDict):
    """Compact evidence returned by one resolver tool call."""

    tool_name: str
    status: str
    summary: str
    data: NotRequired[dict[str, Any]]


class NavigableAnchor(TypedDict):
    """Only this shape can be converted into a navigation dispatch."""

    anchor_type: AnchorType
    source: str
    keyframe_id: NotRequired[int]
    position: NotRequired[list[float]]
    yaw_deg: NotRequired[float]


class ClarificationRequest(TypedDict):
    """Non-blocking shape for future user clarification turns."""

    request_id: str
    task_id: NotRequired[int]
    reason: str
    question: str
    options: NotRequired[list[dict[str, Any]]]
    free_text_allowed: bool
    expected_answer_type: str
    evidence: NotRequired[list[Evidence]]
    created_at: str


class RequiredNextStep(TypedDict):
    """A semantic target needs more evidence before movement is allowed."""

    step_type: str
    reason: str
    clarification_request: NotRequired[ClarificationRequest]
    budget_policy: NotRequired[dict[str, Any]]
    preanalysis_status: NotRequired[str]


class ResolutionDiagnostics(TypedDict):
    """Compact resolver timing and decision metadata for logs."""

    stage: ResolutionStage
    resolver_total_sec: float
    tool_elapsed_sec: float
    tool_names: list[str]
    decision: str


class ResolutionResult(TypedDict):
    """Resolver output consumed by structured navigation execution."""

    status: ResolutionStatus
    draft: TargetDraft
    target_ref: TargetRef
    evidence: list[Evidence]
    anchor: NotRequired[NavigableAnchor]
    required_next_step: NotRequired[RequiredNextStep]
    failure_reason: NotRequired[str]
    diagnostics: NotRequired[ResolutionDiagnostics]


def build_clarification_request(
    *,
    reason: str,
    question: str,
    task_id: int | None = None,
    options: list[dict[str, Any]] | None = None,
    free_text_allowed: bool = True,
    expected_answer_type: str = "text",
    evidence: list[Evidence] | None = None,
) -> ClarificationRequest:
    """Build a clarification request without wiring it into runtime pausing."""

    request: ClarificationRequest = {
        "request_id": f"clarify_{uuid4().hex[:12]}",
        "reason": str(reason or "").strip(),
        "question": str(question or "").strip(),
        "free_text_allowed": bool(free_text_allowed),
        "expected_answer_type": str(expected_answer_type or "text").strip() or "text",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if task_id is not None:
        request["task_id"] = int(task_id)
    if options:
        request["options"] = list(options)
    if evidence:
        request["evidence"] = list(evidence)
    return request


def build_ambiguous_result(
    *,
    draft: TargetDraft,
    target_ref: TargetRef,
    reason: str,
    question: str,
    task_id: int | None = None,
    options: list[dict[str, Any]] | None = None,
    evidence: list[Evidence] | None = None,
) -> ResolutionResult:
    """Return an explicit ambiguous resolver result for future clarification flow."""

    clarification_request = build_clarification_request(
        reason=reason,
        question=question,
        task_id=task_id,
        options=options,
        evidence=evidence,
    )
    return {
        "status": "ambiguous",
        "draft": draft,
        "target_ref": target_ref,
        "evidence": list(evidence or []),
        "required_next_step": {
            "step_type": "needs_user_clarification",
            "reason": reason,
            "clarification_request": clarification_request,
        },
    }
