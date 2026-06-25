"""Per-execute tool-call budget hints for the async agent executor."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from typing import Any, Iterator


KEYFRAME_MATCH_TOOL_NAMES = {
    "search_requirement_on_keyframe_nodes",
    "search_keywords_on_keyframe_nodes",
    "match_attached_image_to_keyframes",
    "analyse_on_each_kf_images",
    "co_analyse_on_kf_images",
}

_tool_counts: ContextVar[dict[str, int] | None] = ContextVar(
    "caragent_execute_tool_counts",
    default=None,
)
_tool_call_signatures: ContextVar[dict[str, int] | None] = ContextVar(
    "caragent_execute_tool_call_signatures",
    default=None,
)


@contextmanager
def execute_tool_budget_context() -> Iterator[None]:
    """Reset tool-call counters for one foreground execute pass."""

    token = _tool_counts.set({})
    signature_token = _tool_call_signatures.set({})
    try:
        yield
    finally:
        _tool_call_signatures.reset(signature_token)
        _tool_counts.reset(token)


def _canonical_tool_args(args: Any) -> str:
    """Build a stable signature for semantic duplicate tool calls."""

    try:
        return json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(args or {})


def maybe_block_repeated_tool_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Return a compact no-new-information result for identical repeated calls."""

    signatures = _tool_call_signatures.get()
    if signatures is None:
        return None

    signature = f"{tool_name}:{_canonical_tool_args(args)}"
    count = int(signatures.get(signature, 0)) + 1
    signatures[signature] = count
    if count <= 1:
        return None

    return {
        "status": "partial",
        "summary": (
            "Skipped an identical repeated tool call in this execute pass. "
            "The previous result for the same tool name and arguments is already "
            "in the conversation; use that result to finish the current task or "
            "change the query arguments if genuinely new evidence is needed."
        ),
        "data": {
            "repeated_tool_call": True,
            "tool_name": tool_name,
            "repeat_count": count,
            "args": args,
            "guidance": (
                "Do not call this exact tool with identical arguments again. "
                "Use the previous result, choose from available evidence, or "
                "finish with a clear failure reason."
            ),
        },
        "error": {
            "code": "repeated_tool_call_no_new_information",
            "message": "Identical tool call was repeated inside one execute pass.",
        },
        "provenance": {
            "source_type": "execute_tool_budget",
            "tool_name": tool_name,
            "contract_version": "tool_result_v1",
        },
    }


def maybe_add_keyframe_match_budget_hint(tool_name: str, result: Any) -> Any:
    """Append a compact runtime hint when scene-memory matching keeps looping."""

    if tool_name not in KEYFRAME_MATCH_TOOL_NAMES:
        return result

    counts = _tool_counts.get()
    if counts is None:
        return result

    count = int(counts.get(tool_name, 0)) + 1
    counts[tool_name] = count
    total = sum(int(counts.get(name, 0)) for name in KEYFRAME_MATCH_TOOL_NAMES)
    if total <= 2:
        return result

    guidance = {
        "type": "keyframe_match_budget_hint",
        "message": (
            "你已经检索/检查关键帧太久了。关键帧场景和当前检索场景可能存在差异；"
            "如果已有候选接近目标，请优先相信历史关键帧描述、候选证据和当前任务的选择策略，"
            "选择最合适的关键帧并发送/返回导航目标，不要继续过度检索。"
        ),
        "keyframe_match_tool_calls": total,
        "trigger_tool": tool_name,
    }

    if isinstance(result, dict):
        annotated = dict(result)
        existing = annotated.get("runtime_guidance")
        if isinstance(existing, list):
            annotated["runtime_guidance"] = [*existing, guidance]
        elif existing:
            annotated["runtime_guidance"] = [existing, guidance]
        else:
            annotated["runtime_guidance"] = [guidance]
        return annotated

    if isinstance(result, str):
        return (
            result
            + "\n\n[runtime_guidance] "
            + guidance["message"]
            + f" keyframe_match_tool_calls={total}; trigger_tool={tool_name}"
        )

    return {"result": result, "runtime_guidance": [guidance]}
