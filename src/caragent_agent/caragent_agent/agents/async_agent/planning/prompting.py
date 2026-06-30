"""Prompt-loading and planning-gate helpers for the async agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class PlanningRequirementDecision(TypedDict, total=False):
    """Structured planning-gate output for direct runtime tasks."""

    requires_planning: bool
    task_type: str
    reason: str


def load_agent_prompts() -> Dict[str, Any]:
    """Load async-agent prompt templates from the shared YAML registry."""

    base_path = Path(__file__).resolve().parent.parent.parent.parent
    try:
        with open(base_path / "prompts" / "agent_prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


AGENT_PROMPTS = load_agent_prompts()


def _normalize_user_input(user_input: str) -> str:
    """Return a normalized lowercase user-input string."""

    return str(user_input or "").strip().lower()


def _direct_llm_action_decision() -> PlanningRequirementDecision:
    """Return metadata for one direct runtime task."""

    return {
        "requires_planning": False,
        "task_type": "llm_action",
        "reason": "lightweight_runtime_interaction",
    }


def _planning_required_decision() -> PlanningRequirementDecision:
    """Return the canonical decision for requests that need planning."""

    return {"requires_planning": True, "reason": "requires_task_plan"}


def _rule_planning_decision(user_input: str) -> Optional[PlanningRequirementDecision]:
    """Return a rules-first structured decision when the request is obvious."""

    normalized = _normalize_user_input(user_input)
    if not normalized:
        return _direct_llm_action_decision()

    planning_markers = (
        "navigate",
        "go to",
        "go back through",
        "guide me to",
        "travel to",
        "drive to",
        "head to",
        "move to",
        "proceed to",
        "return to",
        "repeat this route",
        "repeat that route",
        "navigate to those places",
        "change the plan",
        "change the route",
        "change the remaining plan",
        "change the remaining route",
        "change its destination",
        "change the destination",
        "replan",
        "replace the plan",
        "replace the route",
        "cancel the plan",
        "stop the current plan",
        "insert ",
        "skip ",
        "remove ",
        "after that",
        "then ",
        "if ",
        "when ",
        "unless ",
    )
    if any(marker in normalized for marker in planning_markers):
        return _planning_required_decision()

    direct_query_markers = (
        "where are you",
        "where are you now",
        "where is the robot now",
        "where is the caragent now",
        "current position",
        "your current position",
        "robot position",
        "caragent position",
        "current state",
        "your state",
        "what do you see",
        "current view",
        "current image",
        "how far are you",
        "distance from your current position",
        "distance from the current position",
    )
    if any(marker in normalized for marker in direct_query_markers):
        return _direct_llm_action_decision()

    runtime_memory_markers = (
        "this route",
        "that route",
        "current route",
        "previous route",
        "route distance",
        "total distance",
        "recap",
        "summarize",
        "summary",
        "what did we visit",
        "what have we visited",
        "where we went",
        "where did we go",
        "visited",
        "destinations",
        "actually reach",
        "actually reached",
        "already reached",
        "what edits",
        "edits i have made",
        "edits have i made",
        "plan edit",
        "previous trip",
        "previous tour",
        "last trip",
        "last tour",
        "task history",
        "navigation history",
        "memory",
    )
    if any(marker in normalized for marker in runtime_memory_markers):
        return _direct_llm_action_decision()

    return None


def _rule_requires_planning(user_input: str) -> bool | None:
    """Return a rules-first planning decision when the request is obvious."""

    decision = _rule_planning_decision(user_input)
    if decision is None:
        return None
    return bool(decision.get("requires_planning"))


def _normalize_direct_task_metadata(
    raw: dict[str, Any],
) -> PlanningRequirementDecision:
    """Keep model-provided direct-task metadata inside the supported contract."""

    reason = str(raw.get("reason") or "").strip()
    return {
        "requires_planning": False,
        "task_type": "llm_action",
        "reason": reason or "lightweight_runtime_interaction",
    }


def _parse_planning_requirement_response(content: str) -> PlanningRequirementDecision:
    """Parse the planning-gate JSON contract, with tolerant text handling."""

    text = str(content or "").strip()
    upper_text = text.upper()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            requires = bool(
                parsed.get("requires_planning")
                or str(parsed.get("decision") or "").strip().upper()
                == "REQUIRES PLANNING"
            )
            if requires:
                reason = str(parsed.get("reason") or "").strip()
                decision = _planning_required_decision()
                if reason:
                    decision["reason"] = reason
                return decision
            return _normalize_direct_task_metadata(parsed)
    except Exception:
        pass

    if "REQUIRES PLANNING" in upper_text and "DOES NOT REQUIRE PLANNING" not in upper_text:
        return _planning_required_decision()
    if "DOES NOT REQUIRE PLANNING" in upper_text:
        return _direct_llm_action_decision()
    return _planning_required_decision()


def classify_planning_requirement(
    user_input: str,
    llm: BaseChatModel,
    *,
    agent_prompts: Dict[str, Any] | None = None,
) -> PlanningRequirementDecision:
    """Decide whether input needs planning and return direct-task metadata."""

    rule_decision = _rule_planning_decision(user_input)
    if rule_decision is not None:
        return rule_decision

    prompts = agent_prompts or AGENT_PROMPTS
    prompt_template = prompts.get("is_require_planning_prompt") if prompts else None
    if not prompt_template:
        return _direct_llm_action_decision()

    prompt = prompt_template.format(user_input=user_input)
    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_planning_requirement_response(str(response.content))


def is_require_planning(
    user_input: str,
    llm: BaseChatModel,
    *,
    agent_prompts: Dict[str, Any] | None = None,
) -> bool:
    """Decide whether the user input requires a new plan, using rules first."""

    decision = classify_planning_requirement(
        user_input,
        llm,
        agent_prompts=agent_prompts,
    )
    return bool(decision.get("requires_planning"))


__all__ = [
    "AGENT_PROMPTS",
    "PlanningRequirementDecision",
    "classify_planning_requirement",
    "is_require_planning",
    "load_agent_prompts",
]
