"""Resolver that turns semantic navigation targets into anchors or next steps."""

from __future__ import annotations

import json
import time
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool

from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    get_runtime_tool_context,
)
from caragent_agent.agents.async_agent.execution.tool_results import parse_json_like_payload

from .policies import (
    draft_from_navigation_target,
    next_step_for_unsupported_object,
    target_ref_from_draft,
)
from .session_anchors import (
    anchor_to_navigable,
    find_reusable_anchor,
)
from .types import Evidence, NavigableAnchor, ResolutionResult, TargetDraft, TargetRef


def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> Any:
    invoke = getattr(tool, "invoke", None)
    if not callable(invoke):
        execute = getattr(tool, "execute", None)
        if callable(execute):
            return execute(**args)
        raise TypeError(f"Tool {getattr(tool, 'name', tool)} is not invokable.")
    try:
        return invoke(args)
    except TypeError:
        return invoke(**args)


def _parse_tool_payload(raw_value: Any) -> Any:
    parsed = parse_json_like_payload(raw_value)
    return parsed if parsed is not None else raw_value


def _tool_status_ok(raw_value: Any) -> bool:
    parsed = _parse_tool_payload(raw_value)
    if isinstance(parsed, dict):
        status = str(parsed.get("status") or "").strip().lower()
        if status and status not in {"ok", "success", "succeeded"}:
            return False
    return True


def _tool_summary(raw_value: Any) -> str:
    parsed = _parse_tool_payload(raw_value)
    if isinstance(parsed, dict):
        return str(parsed.get("summary") or "").strip()
    return str(raw_value or "").strip()[:240]


def _tool_data(raw_value: Any) -> dict[str, Any]:
    parsed = _parse_tool_payload(raw_value)
    if not isinstance(parsed, dict):
        return {}
    data = parsed.get("data")
    return data if isinstance(data, dict) else parsed


def _evidence(tool_name: str, raw_value: Any) -> Evidence:
    data = _tool_data(raw_value)
    compact_data = {
        key: data.get(key)
        for key in (
            "resolution_status",
            "recommended_keyframe_id",
            "candidate_keyframe_ids",
            "candidate_keyframes",
            "recommended_destination",
            "recommendation_reason",
            "retrieval_mode",
            "destination",
            "target_description",
            "source",
            "failure_reason",
            "query",
            "image_ref",
            "image_focus",
            "elapsed_sec",
            "requires_budgeted_live_localization",
            "required_next_step",
            "preanalysis_status",
            "background_summary",
            "budget_policy",
        )
        if data.get(key) not in (None, "", [], {})
    }
    return {
        "tool_name": tool_name,
        "status": str((_parse_tool_payload(raw_value) or {}).get("status") or "ok")
        if isinstance(_parse_tool_payload(raw_value), dict)
        else "ok",
        "summary": _tool_summary(raw_value),
        "data": compact_data,
    }


def _anchor_evidence(anchor: dict[str, Any]) -> Evidence:
    return {
        "tool_name": "session_anchor",
        "status": "ok",
        "summary": "Reused a session-local target anchor.",
        "data": {
            key: anchor.get(key)
            for key in (
                "anchor_id",
                "kind",
                "source",
                "task_id",
                "description",
                "keyframe_id",
                "position",
                "yaw_deg",
                "mobility",
                "freshness",
                "_matched_by",
                "_match_reason",
                "_text_score",
                "_match_score",
                "_matched_upstream_task_ids",
            )
            if anchor.get(key) not in (None, "", [], {})
        },
    }


def _unique_tool_names(evidence: list[Evidence]) -> list[str]:
    names: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _evidence_tool_elapsed_sec(evidence: list[Evidence]) -> float:
    total = 0.0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        try:
            total += float(data.get("elapsed_sec") or 0.0)
        except Exception:
            continue
    return round(max(0.0, total), 3)


def _result_stage(
    draft: TargetDraft,
    target_ref: TargetRef,
    result: ResolutionResult,
) -> str:
    evidence = [item for item in list(result.get("evidence") or []) if isinstance(item, dict)]
    if any(str(item.get("tool_name") or "") == "session_anchor" for item in evidence):
        return "session_anchor_reuse"
    target_type = str(draft.get("target_type") or "").strip()
    source = str(target_ref.get("source") or "").strip()
    image_focus = str(target_ref.get("image_focus") or "").strip()
    target_kind = str(target_ref.get("target_kind") or target_ref.get("kind") or "").strip()
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    if target_type == "semantic_keyframe":
        if source == "attached_image":
            if (
                image_focus == "object"
                or target_kind == "object"
                or str(anchor.get("source") or "") == "attached_image_object_staging"
            ):
                return "attached_image_object_staging"
            return "attached_image_keyframe"
        return "semantic_keyframe"
    if target_type == "semantic_object":
        if source == "attached_image":
            return "attached_image_object_staging"
        if source == "current_view":
            return "current_view_object"
        if source in {"arrived_scene", "upstream_result"}:
            return "arrived_scene_object"
    return "unsupported"


def _result_decision(result: ResolutionResult) -> str:
    if result.get("status") == "resolved":
        anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
        anchor_type = str(anchor.get("anchor_type") or "").strip()
        if anchor_type == "keyframe":
            return f"resolved_to_keyframe:{anchor.get('keyframe_id')}"
        if anchor_type == "position":
            return "resolved_to_position"
        return "resolved_without_supported_anchor"
    required_next_step = result.get("required_next_step")
    if isinstance(required_next_step, dict):
        return str(required_next_step.get("step_type") or "required_next_step")
    if result.get("failure_reason"):
        return str(result.get("failure_reason"))
    return str(result.get("status") or "unknown")


def _budgeted_live_required_step(
    raw_value: Any,
    *,
    target_ref: TargetRef,
) -> Optional[dict[str, Any]]:
    data = _tool_data(raw_value)
    if not data:
        return None
    if not bool(data.get("requires_budgeted_live_localization")):
        return None
    source = str(target_ref.get("source") or "").strip()
    if source not in {"arrived_scene", "upstream_result"}:
        return None
    reason = (
        str(data.get("summary") or "").strip()
        or str(data.get("failure_reason") or "").strip()
        or "Historical object preanalysis did not produce a destination; bounded live localization is required."
    )
    return {
        "step_type": "needs_budgeted_live_localization",
        "reason": reason,
        "preanalysis_status": str(data.get("status") or data.get("preanalysis_status") or "").strip(),
        "budget_policy": data.get("budget_policy") if isinstance(data.get("budget_policy"), dict) else {},
    }


def _with_diagnostics(
    result: ResolutionResult,
    *,
    started_at: float,
) -> ResolutionResult:
    draft = result.get("draft") if isinstance(result.get("draft"), dict) else {}
    target_ref = result.get("target_ref") if isinstance(result.get("target_ref"), dict) else {}
    evidence = [item for item in list(result.get("evidence") or []) if isinstance(item, dict)]
    result["diagnostics"] = {
        "stage": _result_stage(draft, target_ref, result),  # type: ignore[arg-type]
        "resolver_total_sec": round(max(0.0, time.perf_counter() - started_at), 3),
        "tool_elapsed_sec": _evidence_tool_elapsed_sec(evidence),
        "tool_names": _unique_tool_names(evidence),
        "decision": _result_decision(result),
    }
    return result


def _contains_unresolved_template(value: Any) -> bool:
    if isinstance(value, str):
        return "{{" in value or "}}" in value
    if isinstance(value, dict):
        return any(_contains_unresolved_template(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_unresolved_template(item) for item in value)
    return False


def _task_image_ref(current_task: dict[str, Any], target_ref: TargetRef) -> Optional[str]:
    runtime_context = get_runtime_tool_context()
    selected_packet = runtime_context.get("selected_execution_context_packet")
    current_user_input = (
        selected_packet.get("current_user_input")
        if isinstance(selected_packet, dict)
        else None
    )
    attached_images = (
        list(current_user_input.get("attached_images") or [])
        if isinstance(current_user_input, dict)
        else []
    )
    by_ref_id = {
        str(item.get("image_ref_id") or "").strip(): item
        for item in attached_images
        if isinstance(item, dict) and str(item.get("image_ref_id") or "").strip()
    }

    raw_refs = []
    image_refs = target_ref.get("image_refs")
    if isinstance(image_refs, (list, tuple)):
        raw_refs.extend(list(image_refs))
    elif image_refs:
        raw_refs.append(image_refs)
    task_refs = current_task.get("image_refs")
    if isinstance(task_refs, (list, tuple)):
        raw_refs.extend(list(task_refs))
    elif task_refs:
        raw_refs.append(task_refs)

    for value in raw_refs:
        text = str(value or "").strip()
        if text.lower() == "latest" and attached_images:
            return json.dumps(attached_images[0], ensure_ascii=False)
        if text in by_ref_id:
            return json.dumps(by_ref_id[text], ensure_ascii=False)
        if text:
            return text
    return None


def _candidate_keyframe_ids(data: dict[str, Any]) -> list[int]:
    ids = data.get("matched_keyframe_ids") or data.get("candidate_keyframe_ids") or []
    result: list[int] = []
    for value in list(ids):
        try:
            item = int(value)
        except Exception:
            continue
        if item not in result:
            result.append(item)
    return result


def _extract_keyframe_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        parsed = parse_json_like_payload(stripped)
        if parsed is not None and parsed is not value:
            return _extract_keyframe_id(parsed)
        return None
    if not isinstance(value, dict):
        return None
    for key in ("recommended_keyframe_id", "keyframe_id", "target_keyframe_id", "keyframe_node_id", "kf_id"):
        if value.get(key) is not None:
            try:
                return int(value[key])
            except Exception:
                continue
    for key in ("recommended_destination", "destination", "target", "data", "result"):
        nested = value.get(key)
        if nested is value:
            continue
        keyframe_id = _extract_keyframe_id(nested)
        if keyframe_id is not None:
            return keyframe_id
    ids = _candidate_keyframe_ids(value)
    return ids[0] if ids else None


def _resolved_keyframe_id_from_tool_data(data: dict[str, Any]) -> Optional[int]:
    resolution_status = str(data.get("resolution_status") or "").strip().lower()
    if resolution_status and resolution_status != "resolved":
        return None
    for key in ("recommended_keyframe_id", "keyframe_id", "target_keyframe_id"):
        if data.get(key) is not None:
            try:
                return int(data[key])
            except Exception:
                continue
    destination = data.get("recommended_destination") or data.get("destination")
    return _extract_keyframe_id(destination)


def _extract_position_destination(value: Any) -> Optional[dict[str, Any]]:
    parsed = _parse_tool_payload(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
        nested = _extract_position_destination(parsed["data"])
        if nested is not None:
            return nested
    if isinstance(parsed, dict):
        destination = parsed.get("destination")
        if destination is not parsed:
            nested = _extract_position_destination(destination)
            if nested is not None:
                return nested
        if parsed.get("type") == "position" and parsed.get("position") is not None:
            position = parsed.get("position")
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                try:
                    return {
                        "position": [
                            float(position[0]),
                            float(position[1]),
                            float(position[2]) if len(position) >= 3 else 0.0,
                        ],
                        "yaw_deg": float(parsed.get("yaw_deg", parsed.get("yaw", 0.0)) or 0.0),
                    }
                except Exception:
                    return None
        approach = parsed.get("approach")
        if isinstance(approach, dict):
            goal = approach.get("map_goal")
            if isinstance(goal, dict):
                return _extract_position_destination(
                    {
                        "type": "position",
                        "position": goal.get("position"),
                        "yaw_deg": goal.get("yaw_deg"),
                    }
                )
    return None


def find_tool_by_name(tools: Sequence[BaseTool], tool_name: str) -> Optional[BaseTool]:
    for tool in tools:
        if str(getattr(tool, "name", "") or "").strip() == tool_name:
            return tool
    return None


class TargetResolver:
    """Resolve one planner semantic target without letting the planner navigate."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        background_result: Optional[dict[str, Any]] = None,
    ):
        self.tools = tools
        self.background_result = background_result

    def draft_and_ref(
        self,
        current_task: dict[str, Any],
        target: dict[str, Any],
    ) -> tuple[TargetDraft, TargetRef]:
        draft = draft_from_navigation_target(current_task, target)
        return draft, target_ref_from_draft(draft, current_task)

    def resolve(
        self,
        current_task: dict[str, Any],
        target: dict[str, Any],
    ) -> ResolutionResult:
        started_at = time.perf_counter()
        draft, target_ref = self.draft_and_ref(current_task, target)
        if draft["target_type"] == "semantic_keyframe":
            result = self._resolve_semantic_keyframe(draft, target_ref, current_task=current_task)
        elif draft["target_type"] == "semantic_object":
            result = self._resolve_semantic_object(draft, target_ref, current_task=current_task)
        else:
            result = {
                "status": "unsupported",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": f"Unsupported semantic target type: {draft['target_type'] or 'missing'}.",
            }
        return _with_diagnostics(result, started_at=started_at)

    def dry_run(
        self,
        current_task: dict[str, Any],
        target: dict[str, Any],
    ) -> ResolutionResult:
        started_at = time.perf_counter()
        draft, target_ref = self.draft_and_ref(current_task, target)
        result: ResolutionResult = {
            "status": "needs_observation",
            "draft": draft,
            "target_ref": target_ref,
            "evidence": [],
            "required_next_step": {
                "step_type": "dry_run_only",
                "reason": "Target resolution dry-run recorded draft/ref without tool calls.",
            },
        }
        return _with_diagnostics(result, started_at=started_at)

    def _resolve_semantic_keyframe(
        self,
        draft: TargetDraft,
        target_ref: TargetRef,
        *,
        current_task: dict[str, Any],
    ) -> ResolutionResult:
        source = target_ref["source"]
        if source != "attached_image":
            reusable = self._reusable_anchor_for_target(target_ref, current_task)
            if reusable is not None:
                navigable = anchor_to_navigable(reusable)
                if navigable is not None and navigable.get("anchor_type") == "keyframe":
                    return {
                        "status": "resolved",
                        "draft": draft,
                        "target_ref": target_ref,
                        "evidence": [_anchor_evidence(reusable)],
                        "anchor": navigable,
                    }
        if source == "attached_image":
            return self._resolve_attached_image_keyframe(
                draft,
                target_ref,
                current_task=current_task,
            )
        if source not in {"scene_memory", "explicit"}:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": (
                    "semantic_keyframe target_source must be scene_memory, explicit, "
                    f"or attached_image; got {source or 'missing'}."
                ),
            }
        query = str(target_ref.get("query") or target_ref.get("description") or "").strip()
        if not query:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "semantic_keyframe target is missing query.",
            }
        background_anchor = self._semantic_keyframe_anchor_from_background()
        if background_anchor is not None:
            evidence = [
                {
                    "tool_name": "background_preanalysis",
                    "status": "ok",
                    "summary": str(
                        (self.background_result or {}).get("summary")
                        or (self.background_result or {}).get("recommendation_reason")
                        or "Reused completed background semantic keyframe preanalysis."
                    ),
                    "data": {
                        "recommended_keyframe_id": background_anchor["keyframe_id"],
                        "candidate_keyframe_ids": (self.background_result or {}).get("candidate_keyframe_ids"),
                    },
                }
            ]
            return {
                "status": "resolved",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": evidence,
                "anchor": background_anchor,
            }
        search_tool = find_tool_by_name(self.tools, "search_requirement_on_keyframe_nodes")
        if search_tool is None:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "Keyframe semantic search tool search_requirement_on_keyframe_nodes is unavailable.",
            }
        search_started = time.perf_counter()
        try:
            raw_search = _invoke_tool(search_tool, {"requirement": query})
        except Exception as exc:
            raw_search = {
                "status": "error",
                "summary": "Semantic keyframe search raised an exception.",
                "error": {"message": str(exc)},
            }
        evidence = [_evidence("search_requirement_on_keyframe_nodes", raw_search)]
        evidence_data = evidence[0].setdefault("data", {})
        if isinstance(evidence_data, dict):
            evidence_data["elapsed_sec"] = round(max(0.0, time.perf_counter() - search_started), 3)
        keyframe_id = _resolved_keyframe_id_from_tool_data(_tool_data(raw_search))
        if keyframe_id is None or not _tool_status_ok(raw_search):
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": evidence,
                "failure_reason": _tool_summary(raw_search)
                or "Semantic keyframe search did not produce a recommended destination.",
            }
        anchor: NavigableAnchor = {
            "anchor_type": "keyframe",
            "keyframe_id": int(keyframe_id),
            "source": "search_requirement_on_keyframe_nodes",
        }
        return {
            "status": "resolved",
            "draft": draft,
            "target_ref": target_ref,
            "evidence": evidence,
            "anchor": anchor,
        }

    def _resolve_attached_image_keyframe(
        self,
        draft: TargetDraft,
        target_ref: TargetRef,
        *,
        current_task: dict[str, Any],
    ) -> ResolutionResult:
        match_tool = find_tool_by_name(self.tools, "match_attached_image_to_keyframes")
        if match_tool is None:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "Attached-image keyframe matcher match_attached_image_to_keyframes is unavailable.",
            }
        query = str(target_ref.get("query") or target_ref.get("description") or "").strip()
        image_ref = _task_image_ref(current_task, target_ref)
        if not image_ref:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "attached-image semantic_keyframe target is missing image_refs.",
            }
        raw_target = draft.get("raw_target") or {}
        if _contains_unresolved_template(
            {
                "query": query,
                "target": raw_target,
                "image_ref": image_ref,
            }
        ):
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "attached-image semantic_keyframe target contains unresolved upstream template text.",
            }
        focus = str(target_ref.get("image_focus") or "").strip().lower()
        if focus not in {"scene", "object"}:
            focus = "scene"
        match_args = {
            "image_ref": image_ref,
            "query": query,
            "focus": focus,
        }
        started = time.perf_counter()
        try:
            raw_match = _invoke_tool(match_tool, match_args)
        except TypeError:
            legacy_args = {"image_ref": image_ref, "query": query}
            try:
                raw_match = _invoke_tool(match_tool, legacy_args)
            except Exception as exc:
                raw_match = {
                    "status": "error",
                    "summary": "Attached-image keyframe match raised an exception.",
                    "error": {"message": str(exc)},
                }
        except Exception as exc:
            raw_match = {
                "status": "error",
                "summary": "Attached-image keyframe match raised an exception.",
                "error": {"message": str(exc)},
            }
        evidence = [_evidence("match_attached_image_to_keyframes", raw_match)]
        evidence_data = evidence[0].setdefault("data", {})
        if isinstance(evidence_data, dict):
            evidence_data.update(
                {
                    "query": query,
                    "image_ref": image_ref,
                    "image_focus": focus,
                    "elapsed_sec": round(max(0.0, time.perf_counter() - started), 3),
                }
            )
        keyframe_id = _resolved_keyframe_id_from_tool_data(_tool_data(raw_match))
        if keyframe_id is None or not _tool_status_ok(raw_match):
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": evidence,
                "failure_reason": _tool_summary(raw_match)
                or "Attached-image keyframe matching did not produce a recommended destination.",
            }
        anchor: NavigableAnchor = {
            "anchor_type": "keyframe",
            "keyframe_id": int(keyframe_id),
            "source": "match_attached_image_to_keyframes",
        }
        return {
            "status": "resolved",
            "draft": draft,
            "target_ref": {
                **target_ref,
                "image_focus": focus,
            },
            "evidence": evidence,
            "anchor": anchor,
        }

    def _semantic_keyframe_anchor_from_background(self) -> Optional[NavigableAnchor]:
        record = self.background_result
        if not isinstance(record, dict):
            return None
        if str(record.get("status") or "").strip().lower() != "completed":
            return None
        keyframe_id = _resolved_keyframe_id_from_tool_data(record)
        if keyframe_id is None:
            return None
        return {
            "anchor_type": "keyframe",
            "keyframe_id": int(keyframe_id),
            "source": "background_preanalysis",
        }

    def _resolve_semantic_object(
        self,
        draft: TargetDraft,
        target_ref: TargetRef,
        *,
        current_task: dict[str, Any],
    ) -> ResolutionResult:
        source = target_ref["source"]
        has_inputs = bool(target_ref.get("inputs_from"))
        if source == "attached_image":
            return self._resolve_attached_image_object_staging(
                draft,
                target_ref,
                current_task=current_task,
            )
        reusable_evidence: list[Evidence] = []
        reusable = self._reusable_anchor_for_target(target_ref, current_task)
        if reusable is not None:
            evidence = [_anchor_evidence(reusable)]
            if str(reusable.get("mobility") or "").strip() == "dynamic":
                return {
                    "status": "needs_observation",
                    "draft": draft,
                    "target_ref": target_ref,
                    "evidence": evidence,
                    "required_next_step": {
                        "step_type": "needs_live_localization_after_arrival",
                        "reason": "Session object anchor is dynamic and must be localized live before navigation.",
                    },
                }
            navigable = anchor_to_navigable(reusable)
            if navigable is not None and navigable.get("anchor_type") == "position":
                return {
                    "status": "resolved",
                    "draft": draft,
                    "target_ref": target_ref,
                    "evidence": evidence,
                    "anchor": navigable,
                }
            if navigable is not None and navigable.get("anchor_type") == "keyframe":
                if source in {"arrived_scene", "upstream_result"} and has_inputs:
                    reusable_evidence = evidence
                else:
                    return {
                        "status": "needs_observation",
                        "draft": draft,
                        "target_ref": target_ref,
                        "evidence": evidence,
                        "required_next_step": {
                            "step_type": "needs_live_localization_after_arrival",
                            "reason": "Session anchor identifies a staging keyframe/view; object still needs live localization after arrival.",
                        },
                    }
        if source in {"arrived_scene", "upstream_result"} and not has_inputs:
            step_type, reason = next_step_for_unsupported_object(source, has_inputs=has_inputs)
            return {
                "status": "needs_observation",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "required_next_step": {"step_type": step_type, "reason": reason},
            }
        if source not in {"current_view", "arrived_scene", "upstream_result"}:
            step_type, reason = next_step_for_unsupported_object(source, has_inputs=has_inputs)
            return {
                "status": "needs_observation",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "required_next_step": {"step_type": step_type, "reason": reason},
            }
        description = str(target_ref.get("description") or "").strip()
        if not description:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "semantic_object target is missing object_description.",
            }
        approach_tool = find_tool_by_name(self.tools, "approach_object_in_current_view")
        if approach_tool is None:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "Object localization tool approach_object_in_current_view is unavailable.",
            }
        args: dict[str, Any] = {
            "target_description": description,
            "target_source": source,
        }
        if target_ref.get("stop_distance_m") is not None:
            args["stop_distance_m"] = float(target_ref["stop_distance_m"])
        approach_started = time.perf_counter()
        try:
            raw_approach = _invoke_tool(approach_tool, args)
        except Exception as exc:
            raw_approach = {
                "status": "error",
                "summary": "Object localization raised an exception.",
                "error": {"message": str(exc)},
            }
        evidence = reusable_evidence + [_evidence("approach_object_in_current_view", raw_approach)]
        last_evidence = evidence[-1] if evidence else {}
        evidence_data = last_evidence.setdefault("data", {}) if isinstance(last_evidence, dict) else {}
        if isinstance(evidence_data, dict):
            evidence_data["elapsed_sec"] = round(max(0.0, time.perf_counter() - approach_started), 3)
        position_destination = _extract_position_destination(raw_approach)
        budgeted_live_step = _budgeted_live_required_step(raw_approach, target_ref=target_ref)
        if position_destination is None and budgeted_live_step is not None:
            return {
                "status": "needs_observation",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": evidence,
                "required_next_step": budgeted_live_step,
            }
        if position_destination is None or not _tool_status_ok(raw_approach):
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": evidence,
                "failure_reason": _tool_summary(raw_approach)
                or "Semantic object grounding did not produce a position destination.",
            }
        anchor: NavigableAnchor = {
            "anchor_type": "position",
            "position": position_destination["position"],
            "yaw_deg": float(position_destination.get("yaw_deg") or 0.0),
            "source": "approach_object_in_current_view",
        }
        return {
            "status": "resolved",
            "draft": draft,
            "target_ref": target_ref,
            "evidence": evidence,
            "anchor": anchor,
        }

    def _resolve_attached_image_object_staging(
        self,
        draft: TargetDraft,
        target_ref: TargetRef,
        *,
        current_task: dict[str, Any],
    ) -> ResolutionResult:
        description = str(target_ref.get("description") or "").strip()
        if not description:
            return {
                "status": "failed",
                "draft": draft,
                "target_ref": target_ref,
                "evidence": [],
                "failure_reason": "attached-image semantic_object target is missing object_description.",
            }
        staging_ref: TargetRef = {
            **target_ref,
            "kind": "object",
            "source": "attached_image",
            "description": description,
            "query": description,
            "image_focus": "object",
            "target_kind": str(target_ref.get("target_kind") or "object"),
        }
        result = self._resolve_attached_image_keyframe(
            draft,
            staging_ref,
            current_task=current_task,
        )
        if result.get("status") == "resolved" and isinstance(result.get("anchor"), dict):
            result["anchor"] = {
                **result["anchor"],
                "source": "attached_image_object_staging",
            }
            result["required_next_step"] = {
                "step_type": "navigate_to_staging_keyframe_then_live_localize",
                "reason": (
                    "Attached-image object targets are staged through a matched keyframe; "
                    "the object position must be localized from the arrived/current view."
                ),
            }
        return result

    def _reusable_anchor_for_target(
        self,
        target_ref: TargetRef,
        current_task: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        runtime_context = get_runtime_tool_context()
        runtime_control = runtime_context.get("shared_runtime_control")
        if not isinstance(runtime_control, dict):
            return None
        tasks = runtime_context.get("tasks")
        task_map = tasks if isinstance(tasks, dict) else {}
        return find_reusable_anchor(target_ref, current_task, task_map, runtime_control)
