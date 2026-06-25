"""Static audit for LLM-facing prompt contract drift.

This check intentionally focuses on prompt text and runtime guidance text that
is shown to models. Internal compatibility field names may still exist in code.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _find_agent_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        workspace_candidate = (
            parent
            / "ros2"
            / "caragent_ws"
            / "src"
            / "caragent_agent"
            / "caragent_agent"
        )
        if (workspace_candidate / "prompts" / "agent_prompts.yaml").exists():
            return workspace_candidate
        board_candidate = parent / "src" / "caragent_agent" / "caragent_agent"
        if (board_candidate / "prompts" / "agent_prompts.yaml").exists():
            return board_candidate
        package_candidate = parent / "caragent_agent"
        if (package_candidate / "prompts" / "agent_prompts.yaml").exists():
            return package_candidate
    raise FileNotFoundError("Could not locate caragent_agent package root")


REPO_AGENT_ROOT = _find_agent_root()
PROMPT_PATH = REPO_AGENT_ROOT / "prompts" / "agent_prompts.yaml"
CONTEXT_PATH = REPO_AGENT_ROOT / "agents" / "async_agent" / "execution" / "context.py"


FORBIDDEN_PROMPT_PHRASES = (
    "destination resolver",
    "resolver task",
    "broad resolver",
    "object-level resolver",
    "object-level destination resolver",
    "finish with a json destination",
    "json destination",
)


def _load_prompt_text() -> str:
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    parts = []
    for key in ("react_agent_system", "plan_system", "react_system", "background_system"):
        value = data.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _load_runtime_guidance_text() -> str:
    return CONTEXT_PATH.read_text(encoding="utf-8")


def main() -> int:
    prompt_text = _load_prompt_text().lower()
    context_text = _load_runtime_guidance_text().lower()
    failures: list[dict[str, str]] = []

    for phrase in FORBIDDEN_PROMPT_PHRASES:
        if phrase in prompt_text:
            failures.append({"source": str(PROMPT_PATH), "phrase": phrase})
        if phrase in context_text:
            failures.append({"source": str(CONTEXT_PATH), "phrase": phrase})

    result = {"status": "ok" if not failures else "failed", "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
