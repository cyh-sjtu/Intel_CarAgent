"""Public interface wiring tools, LLM, and LangGraph agent for the async agent runtime."""

import ast
import inspect
import json
import math
import os
import re
import select
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, create_model

from caragent_agent.agents.async_agent.async_agent_graph import create_async_agent
from caragent_agent.agents.async_agent.arrival_verification import (
    build_arrival_verification,
)
from caragent_agent.agents.async_agent.execution.support import (
    derive_headline_turn_response,
    normalize_turn_response_items,
)
from caragent_agent.agents.async_agent.orchestration.runtime import (
    build_navigation_arrival_event,
    new_structured_id,
    now_iso,
    parse_navigation_arrival_position,
)
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.execution.tool_call_budget import (
    maybe_block_repeated_tool_call,
    maybe_add_keyframe_match_budget_hint,
)
from caragent_agent.agents.async_agent.memory.run_memory import AsyncAgentRunMemory
from caragent_agent.agents.async_agent.runtime.resource_scheduler import resolve_runtime_profile
from caragent_agent.agents.base.base_agent_interface import BaseAgent, ToolOrchestrate
from caragent_agent.agents.tools.analysis.image_analyzer import (
    CurrentImageAnalyzerTool,
    ImageAnalyzerTool,
    MultiImageAnalyzerTool,
)
from caragent_agent.agents.tools.analysis.attached_image_tools import (
    AttachedImageAnalyzerTool,
    AttachedImageKeyframeMatcherTool,
    AttachedImageObjectResolverTool,
    HistoricalKeyframeObjectPreanalysisTool,
)
from caragent_agent.agents.tools.current.get_current_state import Get_Current_State_Tool
from caragent_agent.agents.tools.current.capture_current_view import CaptureCurrentViewTool
from caragent_agent.agents.tools.info.get_nodes_info import GetKeyFrameNodesInfoTool
from caragent_agent.agents.tools.memory import QueryMemoryTool
from caragent_agent.agents.tools.navigation.navigator import NavigationTool, NavigationToPositionTool
from caragent_agent.agents.tools.objects.approach_object import ApproachObjectInCurrentViewTool
from caragent_agent.agents.tools.search.keyword_search import KeywordSearchTool
from caragent_agent.agents.tools.search.requirement_search import RequirementSearchTool
from caragent_agent.config.config import config, ensure_api_key_env
from caragent_agent.third_party.from_langgraph.react_agent import create_react_agent
from caragent_agent.utils.conversation_logger import ConversationLogger
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import get_react_agent_system_prompt


class AsyncAgent(BaseAgent):
    """Facade that configures tools, LLM backend, and delegates messages to the async LangGraph."""

    def __init__(
        self,
        scene_memory,
        is_navigation_mode: bool = True,  # enable navigation tool or not
        controller_type: str = "nav2",
        controller=None,
        enable_logging: bool = True,
        use_multi_agents: bool = True,  # True for enable Multi-Agents framework (orchestrator, plan, executer, background workers, False for simple ReAct Agent)
        num_background_workers: int | None = None,
        record: bool = False,
    ):
        self.is_navigation_mode = is_navigation_mode
        self.controller_type = controller_type
        self.controller = controller
        self.use_multi_agents = use_multi_agents
        self.runtime_profile = resolve_runtime_profile(config)
        self.num_background_workers = self._resolve_background_worker_count(
            num_background_workers
        )
        self.record = record
        super().__init__("AsyncAgent", scene_memory, enable_logging)

        # set api key for Agent core LLM (the LLM used in nodes or tools are separately set)
        self.llm = self._build_core_llm()
        self.background_llm = self._build_background_llm()
        self.checkpointer = InMemorySaver()

        if self.enable_logging:
            log_dir = config.get("log_dir", "logs")
            self.logger = ConversationLogger(log_dir=log_dir)
            self.logger.setup_library_logging()

            if hasattr(self, "controller") and self.controller:
                if hasattr(self.controller, "set_logger"):
                    self.controller.set_logger(self.logger, record=self.record)

                if hasattr(self.controller, "get_timestamp"):
                    try:
                        self.logger.set_time_provider(
                            lambda: str(self.controller.get_timestamp())
                        )
                    except Exception as e:
                        print(f"Failed to set timestamp provider in init: {e}")

        self.run_memory = AsyncAgentRunMemory(
            session_id=getattr(self.logger, "session_name", None) if hasattr(self, "logger") and self.logger else None,
            session_dir=self.logger.get_session_dir() if hasattr(self, "logger") and self.logger else None,
            metadata={
                "agent_name": self.name,
                "controller_type": self.controller_type,
                "is_navigation_mode": self.is_navigation_mode,
                "use_multi_agents": self.use_multi_agents,
                "num_background_workers": self.num_background_workers,
                "logging_enabled": self.enable_logging,
            },
        )
        self._turn_locks: dict[str, threading.RLock] = {}
        self._controller_arrival_watchdogs: dict[str, dict[str, Any]] = {}
        self._controller_arrival_watchdogs_lock = threading.RLock()
        self._controller_arrival_turn_listeners: list[Any] = []
        self._bind_runtime_references_to_tools()

        self._create_langgraph_agent()
        self._start_session_initial_pose_recorder()

    def _resolve_background_worker_count(self, explicit_workers: int | None) -> int:
        """Use explicit worker count when provided, otherwise runtime profile default."""

        if explicit_workers is not None:
            try:
                return max(0, int(explicit_workers))
            except Exception:
                return 0

        try:
            return max(
                0,
                int(self.runtime_profile.get("background_workers_default", 0)),
            )
        except Exception:
            return 0

    def _start_session_initial_pose_recorder(self) -> None:
        """Record the first valid map-frame robot pose as the session route start."""

        if not self.is_navigation_mode:
            return
        if not getattr(self, "controller", None) or not getattr(self, "run_memory", None):
            return
        get_state = getattr(self.controller, "get_current_state", None)
        if not callable(get_state):
            return

        thread = threading.Thread(
            target=self._record_session_initial_pose_when_available,
            name="caragent-session-initial-pose",
            daemon=True,
        )
        thread.start()

    def _record_session_initial_pose_when_available(self) -> None:
        """Poll briefly after startup until localization exposes a stable map pose."""

        deadline = time.monotonic() + 120.0
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                state = self.controller.get_current_state()
            except Exception as exc:
                state = {"source": "unavailable", "error": str(exc)}

            if isinstance(state, dict):
                source = str(state.get("source") or "").strip()
                position = state.get("position")
                if source in {"tf", "simulation"} and self._is_valid_pose_position(position):
                    details = {
                        "controller_status": state.get("status"),
                        "state_source": source,
                        "record_attempt": attempt,
                    }
                    if state.get("yaw_deg") is not None:
                        details["yaw_deg"] = state.get("yaw_deg")
                    recorded = self.run_memory.record_session_initial_pose(
                        position=[
                            float(position[0]),
                            float(position[1]),
                            float(position[2]) if len(position) >= 3 else 0.0,
                        ],
                        orientation=state.get("orientation"),
                        source=source,
                        details=details,
                    )
                    if recorded and hasattr(self, "logger") and self.logger:
                        try:
                            self.logger.log_foreground(
                                "RunMemory: Session initial pose recorded: "
                                f"[{float(position[0]):.3f}, {float(position[1]):.3f}, "
                                f"{float(position[2]) if len(position) >= 3 else 0.0:.3f}] "
                                f"source={source}"
                            )
                        except Exception:
                            pass
                    return

            time.sleep(1.0)

        if hasattr(self, "logger") and self.logger:
            try:
                self.logger.log_foreground(
                    "RunMemory: Session initial pose was not recorded because no map-frame pose became available."
                )
            except Exception:
                pass

    @staticmethod
    def _is_valid_pose_position(position: Any) -> bool:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return False
        try:
            values = [
                float(position[0]),
                float(position[1]),
                float(position[2]) if len(position) >= 3 else 0.0,
            ]
        except Exception:
            return False
        return all(math.isfinite(value) for value in values)

    def _qwen_enable_thinking_for_role(self, role: str | None) -> bool:
        """Return whether DashScope/Qwen thinking mode should be enabled.

        DashScope thinking-mode tool loops require provider reasoning_content to
        be passed back verbatim on later turns. Generic LangChain/OpenAI message
        conversion does not preserve that reliably, so async-agent roles disable
        Qwen thinking by default unless explicitly overridden in config.
        """

        role_overrides = config.get("llm_qwen_enable_thinking_by_role")
        if isinstance(role_overrides, dict) and role in role_overrides:
            return bool(role_overrides[role])
        return bool(config.get("llm_qwen_enable_thinking", False))

    def _deepseek_enable_thinking_for_role(self, role: str | None) -> bool:
        """Return whether DeepSeek reasoning/thinking should be enabled.

        DeepSeek models with reasoning (deepseek-v4-pro, etc.) return
        ``reasoning_content`` that must be passed back verbatim on every
        subsequent request.  LangChain message conversion does not preserve
        that field across ReAct tool-calling turns, so async-agent roles
        disable DeepSeek thinking by default unless explicitly overridden.
        """

        role_overrides = config.get("llm_deepseek_enable_thinking_by_role")
        if isinstance(role_overrides, dict) and role in role_overrides:
            return bool(role_overrides[role])
        return bool(config.get("llm_deepseek_enable_thinking", False))

    def _build_chat_openai(
        self,
        model_name: str,
        *,
        role: str | None = None,
    ) -> ChatOpenAI:
        """Build a ChatOpenAI instance from a concrete model identifier.

        Provider (base URL + API key) is detected from the model name prefix:
        qwen* → DashScope, deepseek* → DeepSeek API.
        """

        normalized = str(model_name or "").strip().lower()
        if not normalized:
            raise ValueError("LLM model name must be a non-empty string.")

        if normalized.startswith("qwen"):
            api_key = ensure_api_key_env("qwen")
            if not api_key or not str(api_key).strip():
                raise ValueError(
                    "DASHSCOPE_API_KEY is missing; set the environment variable "
                    "or add api_keys.dashscope to ignored local_config.yaml/llm_api.yaml."
                )
            return ChatOpenAI(
                api_key=str(api_key).strip(),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=model_name,
                temperature=0.0,
                extra_body={
                    "enable_thinking": self._qwen_enable_thinking_for_role(role)
                },
            )

        if normalized.startswith("deepseek"):
            api_key = ensure_api_key_env("deepseek")
            if not api_key or not str(api_key).strip():
                raise ValueError(
                    "DEEPSEEK_API_KEY is missing; set the environment variable "
                    "or add api_keys.deepseek to ignored local_config.yaml/llm_api.yaml."
                )
            extra: dict[str, Any] = {}
            thinking_enabled = self._deepseek_enable_thinking_for_role(role)
            if not thinking_enabled:
                extra["thinking"] = {"type": "disabled"}
            return ChatOpenAI(
                api_key=str(api_key).strip(),
                base_url="https://api.deepseek.com/v1",
                model=model_name,
                temperature=0.0,
                extra_body=extra if extra else None,
            )

        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Expected a Qwen or DeepSeek model alias."
        )

    def build_llm(self, role: str) -> ChatOpenAI:
        """Build a chat model for an agent role from llm_routing configuration.

        When ``llm_routing`` is populated every role uses its declared model.
        Otherwise the system uses the single-model config keys
        (``agent_core_llm_model`` / ``agent_background_llm_model``).

        Supported roles: ``orchestrate``, ``planner``, ``executor``, ``background``.
        """

        _ROLE_TO_MODEL_KEY: dict[str, str] = {
            "orchestrate": "agent_core_llm_model",
            "planner": "agent_core_llm_model",
            "executor": "agent_core_llm_model",
            "background": "agent_background_llm_model",
        }

        routing = config.get("llm_routing")
        if isinstance(routing, dict) and routing.get(role):
            return self._build_chat_openai(str(routing[role]).strip(), role=role)

        model_key = _ROLE_TO_MODEL_KEY.get(role)
        if model_key:
            model_name = str(config.get(model_key) or "").strip()
        else:
            model_name = ""

        if not model_name:
            model_name = str(routing.get("default") if isinstance(routing, dict) else "") or "qwen3.6-plus"

        return self._build_chat_openai(model_name, role=role)

    def _build_core_llm(self):
        """Build the primary async-agent chat model (orchestrate + plan + execute)."""

        return self.build_llm("orchestrate")

    def _build_background_llm(self):
        """Build the background-agent model, falling back to the core model when unavailable."""

        try:
            return self.build_llm("background")
        except Exception as exc:
            print(
                f"Background LLM unavailable; falling back to core model: {exc}"
            )
            return self.llm

    def _setup_tools(self):
        """Register available tools and bind navigation/controller tools when enabled."""
        self.register_tool(KeywordSearchTool())
        self.register_tool(RequirementSearchTool())
        self.register_tool(GetKeyFrameNodesInfoTool())
        self.register_tool(ImageAnalyzerTool())
        self.register_tool(AttachedImageAnalyzerTool())
        self.register_tool(AttachedImageKeyframeMatcherTool())
        self.register_tool(AttachedImageObjectResolverTool())
        self.register_tool(HistoricalKeyframeObjectPreanalysisTool())
        self.register_tool(QueryMemoryTool())
        # self.register_tool(MultiImageAnalyzerTool())

        if self.is_navigation_mode:
            nav_tool = NavigationTool(self.controller)
            self.register_tool(nav_tool)
            self.controller = nav_tool.controller
            self.register_tool(NavigationToPositionTool(self.controller))

            self.register_tool(Get_Current_State_Tool(self.controller))
            self.register_tool(CaptureCurrentViewTool(self.controller))
            self.register_tool(CurrentImageAnalyzerTool(self.controller))
            self.register_tool(ApproachObjectInCurrentViewTool(self.controller))

    def _bind_runtime_references_to_tools(self) -> None:
        """Attach runtime objects such as run memory to all registered tools."""

        for tool in self.get_all_tools():
            tool.scene_memory = self.scene_memory
            tool.run_memory = getattr(self, "run_memory", None)

    def _create_langgraph_agent(self):
        """Instantiate either the multi-node async agent or a simple ReAct agent."""
        tools = self._get_langchain_tools()

        if self.use_multi_agents:
            background_tools = self._get_background_langchain_tools()

            self.agent = create_async_agent(
                tools=tools,
                orchestrate_llm=self.build_llm("orchestrate"),
                planner_llm=self.build_llm("planner"),
                executor_llm=self.build_llm("executor"),
                background_llm=self.build_llm("background"),
                background_tools=background_tools,
                num_background_workers=self.num_background_workers,
                checkpointer=self.checkpointer,
                logger=self.logger if hasattr(self, "logger") else None,
                run_memory=getattr(self, "run_memory", None),
            )
        else:
            self.agent = create_react_agent(
                model=self.llm,
                tools=tools,
                prompt=get_react_agent_system_prompt(),
                checkpointer=self.checkpointer,
                logger=self.logger.log_foreground if hasattr(self, "logger") and self.logger else None,
            )

    def _get_langchain_tools(self) -> List:
        """Wrap registered tools into LangChain StructuredTool instances for the orchestrator."""
        langchain_tools = []

        for tool in self.get_all_tools():
            sig = inspect.signature(tool.execute)
            type_hints = get_type_hints(tool.execute)

            fields = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue

                param_type = type_hints.get(param_name, str)
                schema_param_type = self._get_input_schema_param_type(param_type)

                if param.default != inspect.Parameter.empty:
                    default_value = param.default
                else:
                    default_value = ...

                fields[param_name] = (
                    schema_param_type,
                    Field(default=default_value, description=f"Parameter {param_name}"),
                )

            InputSchema = create_model(f"{tool.name}Input", **fields)

            def _wrapped_execute(
                _tool=tool,
                _type_hints=type_hints,
                **kwargs,
            ):
                normalized_kwargs = self._normalize_tool_kwargs(kwargs, _type_hints)
                repeated_result = maybe_block_repeated_tool_call(
                    _tool.name,
                    normalized_kwargs,
                )
                if repeated_result is not None:
                    return repeated_result
                result = _tool.execute(**normalized_kwargs)
                return maybe_add_keyframe_match_budget_hint(_tool.name, result)

            structured_tool = StructuredTool.from_function(
                func=_wrapped_execute,
                name=tool.name,
                description=tool.description,
                args_schema=InputSchema,
                return_direct=self._tool_should_return_direct(tool),
            )
            object.__setattr__(
                structured_tool,
                "capability_tags",
                getattr(tool, "capability_tags", ()),
            )

            langchain_tools.append(structured_tool)

        return langchain_tools

    def _tool_should_return_direct(self, tool: Any) -> bool:
        """Stop the ReAct loop immediately after side-effectful navigation tools run."""

        return getattr(tool, "name", "") in {
            "go_to_keyframe",
            "go_to_position",
            "submit_task_result",
        }

    def _get_background_langchain_tools(self) -> List:
        """Filter and wrap tools that are safe to run in background workers. The tools for background worker exclude navigation and current state tools, which are not suitable for background processing."""

        BACKGROUND_ALLOWED_TOOLS = {
            "KeywordSearchTool",
            "RequirementSearchTool",
            "GetKeyFrameNodesInfoTool",
            "QueryMemoryTool",
            "HistoricalKeyframeObjectPreanalysisTool",
        }

        langchain_tools = []

        for tool in self.get_all_tools():
            if type(tool).__name__ not in BACKGROUND_ALLOWED_TOOLS:
                continue

            sig = inspect.signature(tool.execute)
            type_hints = get_type_hints(tool.execute)

            fields = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue

                param_type = type_hints.get(param_name, str)
                schema_param_type = self._get_input_schema_param_type(param_type)

                if param.default != inspect.Parameter.empty:
                    default_value = param.default
                else:
                    default_value = ...

                fields[param_name] = (
                    schema_param_type,
                    Field(default=default_value, description=f"Parameter {param_name}"),
                )

            InputSchema = create_model(f"{tool.name}Input", **fields)

            def _wrapped_execute(
                _tool=tool,
                _type_hints=type_hints,
                **kwargs,
            ):
                normalized_kwargs = self._normalize_tool_kwargs(kwargs, _type_hints)
                with UnifiedLLMClient.request_priority("background"):
                    return _tool.execute(**normalized_kwargs)

            structured_tool = StructuredTool.from_function(
                func=_wrapped_execute,
                name=tool.name,
                description=tool.description,
                args_schema=InputSchema,
            )
            object.__setattr__(
                structured_tool,
                "capability_tags",
                getattr(tool, "capability_tags", ()),
            )

            langchain_tools.append(structured_tool)

        return langchain_tools

    def _get_input_schema_param_type(self, param_type: Any) -> Any:
        """Relax list-like tool inputs so schema validation can accept list strings from LLMs."""

        if self._type_accepts_sequence_input(param_type):
            return Union[param_type, str]
        return param_type

    def _type_accepts_sequence_input(self, param_type: Any) -> bool:
        """Return True when one parameter annotation expects a list-like value."""

        origin = get_origin(param_type)
        if origin is list:
            return True
        if origin is None:
            return False
        return any(
            self._type_accepts_sequence_input(arg)
            for arg in get_args(param_type)
            if arg is not type(None)
        )

    def _normalize_tool_kwargs(
        self,
        kwargs: Dict[str, Any],
        type_hints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Coerce common LLM-emitted argument shapes into the tool's expected runtime values."""

        normalized_kwargs = dict(kwargs)
        normalized_kwargs.pop("kwargs", None)
        for param_name, param_type in type_hints.items():
            if param_name == "return" or param_name not in normalized_kwargs:
                continue
            if not self._type_accepts_sequence_input(param_type):
                continue
            normalized_kwargs[param_name] = self._coerce_sequence_argument(
                normalized_kwargs.get(param_name)
            )
        return normalized_kwargs

    def _coerce_sequence_argument(self, value: Any) -> Any:
        """Convert stringified list inputs like \"[1, 2, 3]\" into real Python lists."""

        if value is None or isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            if isinstance(parsed, tuple):
                return list(parsed)
            if isinstance(parsed, list):
                return parsed

        if "," in stripped and not any(token in stripped for token in "[](){}"):
            return [
                item.strip()
                for item in stripped.split(",")
                if item.strip()
            ]

        return value

    def _poll_controller_message(self) -> str:
        """Return the latest controller message when available."""
        controller = getattr(self, "controller", None)
        if controller is None or not hasattr(controller, "check_for_new_messages"):
            return ""

        message = controller.check_for_new_messages()
        return str(message) if message else ""

    def _controller_arrived_status_message(self) -> str:
        """Synthesize an arrival message when controller status already reached arrived."""

        controller = getattr(self, "controller", None)
        if controller is None or not hasattr(controller, "get_status"):
            return ""
        try:
            status = str(controller.get_status() or "").strip().lower()
        except Exception:
            return ""
        if status != "arrived":
            return ""

        position = None
        if hasattr(controller, "get_current_state"):
            try:
                state = controller.get_current_state()
                if isinstance(state, dict):
                    position = state.get("position")
            except Exception:
                position = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            try:
                z = float(position[2]) if len(position) >= 3 else 0.0
                return "Arrived at destination [{x:.3f}, {y:.3f}, {z:.3f}]".format(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=z,
                )
            except (TypeError, ValueError):
                pass
        return "Arrived at destination."

    def _controller_message_is_arrival(self, message: str) -> bool:
        """Return True when a raw controller message reports arrival."""

        normalized = str(message or "").lower()
        return (
            "arrived at destination" in normalized
            or "arrived at the navigation goal" in normalized
            or "navigation goal reached" in normalized
        )

    def _controller_arrival_matches_active_navigation(
        self,
        message: str,
        state: dict[str, Any],
        *,
        tolerance_m: float | None = None,
    ) -> bool:
        """Reject stale controller arrival messages from a previous navigation leg."""

        if tolerance_m is None:
            nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
            try:
                tolerance_m = float(nav_cfg.get("controller_arrival_match_tolerance_m", 2.0))
            except Exception:
                tolerance_m = 2.0
        tolerance_m = max(0.5, float(tolerance_m))

        if not isinstance(state, dict):
            return False
        active_navigation = state.get("active_navigation")
        if not isinstance(active_navigation, dict):
            return False
        active_plan_id = str(active_navigation.get("plan_id") or "").strip()
        current_plan_id = str(state.get("current_plan_id") or "").strip()
        if active_plan_id and current_plan_id and active_plan_id != current_plan_id:
            return False
        try:
            active_task_id = int(active_navigation.get("task_id"))
        except Exception:
            active_task_id = None
        task = (state.get("tasks") or {}).get(active_task_id) if active_task_id is not None else None
        if not (
            isinstance(task, dict)
            and str(task.get("status") or "").strip().lower() == "waiting"
            and task.get("wait_for_event") == "navigation_arrived"
        ):
            return False

        reported_position = parse_navigation_arrival_position(message)
        destination = active_navigation.get("destination_position")
        if reported_position is None or not isinstance(destination, (list, tuple)):
            return True
        if len(reported_position) < 2 or len(destination) < 2:
            return True

        try:
            dx = float(reported_position[0]) - float(destination[0])
            dy = float(reported_position[1]) - float(destination[1])
        except Exception:
            return True

        return (dx * dx + dy * dy) ** 0.5 <= tolerance_m

    def _resolve_navigation_arrival_task_id(
        self,
        state: dict[str, Any],
    ) -> int | None:
        """Resolve the in-flight navigation task that should consume arrival."""

        if not isinstance(state, dict):
            return None
        tasks = state.get("tasks")
        if not isinstance(tasks, dict):
            tasks = {}

        active_navigation = state.get("active_navigation")
        if isinstance(active_navigation, dict):
            try:
                task_id = int(active_navigation.get("task_id"))
            except Exception:
                task_id = None
            if task_id is not None and task_id in tasks:
                task = tasks[task_id]
                if (
                    isinstance(task, dict)
                    and str(task.get("status") or "").strip().lower() == "waiting"
                    and task.get("wait_for_event") == "navigation_arrived"
                ):
                    return task_id

        current_task_id = state.get("current_task_id")
        try:
            current_task_id = int(current_task_id)
        except Exception:
            current_task_id = None
        if current_task_id is not None and current_task_id in tasks:
            task = tasks[current_task_id]
            if (
                isinstance(task, dict)
                and str(task.get("status") or "").strip().lower() == "waiting"
                and task.get("wait_for_event") == "navigation_arrived"
            ):
                return current_task_id

        waiting_task_ids: list[int] = []
        for raw_task_id, task in tasks.items():
            if not isinstance(task, dict):
                continue
            if (
                str(task.get("status") or "").strip().lower() == "waiting"
                and task.get("wait_for_event") == "navigation_arrived"
            ):
                try:
                    waiting_task_ids.append(int(raw_task_id))
                except Exception:
                    continue
        if len(waiting_task_ids) == 1:
            return waiting_task_ids[0]
        return None

    def _build_controller_arrival_event(
        self,
        message: str,
        *,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a physical navigation-arrival event without language ingest."""

        content = str(message or "").strip()
        active_navigation = state.get("active_navigation") if isinstance(state, dict) else None

        def attach_arrival_verification(event: dict[str, Any]) -> dict[str, Any]:
            if event.get("type") != "navigation_arrived":
                return event
            payload = event.setdefault("payload", {})
            if not isinstance(payload, dict):
                return event
            tasks = state.get("tasks") if isinstance(state, dict) else {}
            if not isinstance(tasks, dict):
                return event
            task_id = event.get("task_id")
            arrived_task = tasks.get(task_id) if task_id is not None else None
            if not isinstance(arrived_task, dict):
                return event
            try:
                verification = build_arrival_verification(
                    arrived_task=arrived_task,
                    tasks=tasks,
                    tools=self._get_langchain_tools(),
                    event_payload=payload,
                )
            except Exception as exc:
                verification = {
                    "type": "arrival_verification",
                    "target_label": str(arrived_task.get("description") or "目标").strip(),
                    "target_type": "unknown",
                    "seen": None,
                    "confidence": "error",
                    "reason": str(exc),
                    "message": "",
                }
            if verification:
                payload["arrival_verification"] = verification
                if self.enable_logging and self.logger:
                    self.logger.log_foreground(
                        "arrival_verification: "
                        + json.dumps(verification, ensure_ascii=False, default=str)
                    )
            return event

        def attach_arrival_capture(event: dict[str, Any]) -> dict[str, Any]:
            payload = event.setdefault("payload", {})
            if not isinstance(payload, dict):
                return event
            if event.get("type") != "navigation_arrived":
                return event
            agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
            if not bool(agent_cfg.get("capture_view_on_navigation_arrival", True)):
                return attach_arrival_verification(event)
            image_format = str(agent_cfg.get("arrival_capture_image_format") or "jpg").strip() or "jpg"
            note_parts = ["navigation_arrived"]
            if payload.get("destination_description"):
                note_parts.append(str(payload.get("destination_description")))
            if event.get("task_id") is not None:
                note_parts.append(f"task_id={event.get('task_id')}")
            try:
                capture_result = CaptureCurrentViewTool(self.controller).execute(
                    note="; ".join(note_parts),
                    image_format=image_format,
                )
            except Exception as exc:
                payload["arrival_capture_error"] = {
                    "code": "arrival_capture_exception",
                    "message": str(exc),
                }
                if self.enable_logging and self.logger:
                    self.logger.log_foreground(
                        "arrival_current_view_capture_failed: "
                        + json.dumps(payload["arrival_capture_error"], ensure_ascii=False, default=str)
                    )
                return attach_arrival_verification(event)
            data = (
                capture_result.get("data")
                if isinstance(capture_result, dict) and isinstance(capture_result.get("data"), dict)
                else {}
            )
            status = str(capture_result.get("status") or "").strip().lower() if isinstance(capture_result, dict) else ""
            if status in {"ok", "success", "succeeded"} and data.get("path"):
                payload["arrival_image_ref"] = {
                    key: data.get(key)
                    for key in (
                        "image_ref_id",
                        "path",
                        "source",
                        "note",
                        "image_format",
                        "width",
                        "height",
                        "image_source",
                    )
                    if data.get(key) not in (None, "", [], {})
                }
                if self.enable_logging and self.logger:
                    self.logger.log_foreground(
                        "arrival_current_view_captured: "
                        + json.dumps(payload["arrival_image_ref"], ensure_ascii=False, default=str)
                    )
            else:
                payload["arrival_capture_error"] = {
                    "code": (
                        (capture_result.get("error") or {}).get("code")
                        if isinstance(capture_result, dict) and isinstance(capture_result.get("error"), dict)
                        else "arrival_capture_failed"
                    ),
                    "summary": capture_result.get("summary") if isinstance(capture_result, dict) else str(capture_result),
                }
                if self.enable_logging and self.logger:
                    self.logger.log_foreground(
                        "arrival_current_view_capture_failed: "
                        + json.dumps(payload["arrival_capture_error"], ensure_ascii=False, default=str)
                    )
            return attach_arrival_verification(event)

        if isinstance(active_navigation, dict):
            nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
            try:
                arrival_match_tolerance = float(
                    nav_cfg.get("controller_arrival_match_tolerance_m", 2.0)
                )
            except Exception:
                arrival_match_tolerance = 2.0
            event = build_navigation_arrival_event(
                ToolMessage(content=content, tool_call_id=new_structured_id("controller_message")),
                messages=list(state.get("messages") or []),
                tasks=dict(state.get("tasks") or {}),
                current_task_id=state.get("current_task_id"),
                current_plan_id=state.get("current_plan_id"),
                active_navigation=active_navigation,
                match_tolerance_meters=max(0.5, arrival_match_tolerance),
            )
            event["message_id"] = new_structured_id("controller_message")
            payload = event.setdefault("payload", {})
            if isinstance(payload, dict):
                payload["content"] = content
                payload["reported_position"] = parse_navigation_arrival_position(content)
                payload["turn_response_type"] = "result"
                if event.get("type") == "navigation_arrived":
                    payload["summary"] = "Controller reported navigation arrival."
            return attach_arrival_capture(event)

        payload: dict[str, Any] = {
            "summary": "Controller reported navigation arrival.",
            "content": content,
            "reported_position": parse_navigation_arrival_position(content),
            "turn_response_type": "result",
            "unmatched_reason": "missing_active_navigation",
        }
        task_id = self._resolve_navigation_arrival_task_id(state)
        if task_id is not None:
            payload["destination_description"] = str(
                ((state.get("tasks") or {}).get(task_id) or {}).get("description")
                or ""
            ).strip()

        event: dict[str, Any] = {
            "event_id": new_structured_id("event"),
            "type": "navigation_arrival_unmatched",
            "source": "system",
            "created_at": now_iso(),
            "message_id": new_structured_id("controller_message"),
            "payload": payload,
        }
        if task_id is not None:
            event["task_id"] = task_id
        return attach_arrival_capture(event)

    def _dispatch_controller_arrival_event(
        self,
        message: str,
        thread_id: str,
    ) -> bool:
        """Inject one physical arrival event and trigger orchestration."""

        result = self.run_controller_arrival_turn(message, thread_id)
        return bool(result.get("dispatched"))

    def add_controller_arrival_turn_listener(self, listener: Any) -> None:
        """Register a callback for completed background controller-arrival turns."""

        if not callable(listener):
            return
        listeners = getattr(self, "_controller_arrival_turn_listeners", None)
        if not isinstance(listeners, list):
            listeners = []
            self._controller_arrival_turn_listeners = listeners
        if listener not in listeners:
            listeners.append(listener)

    def _notify_controller_arrival_turn_listeners(
        self,
        turn_result: dict[str, Any],
    ) -> None:
        """Notify optional UI/session observers after physical arrival turns."""

        listeners = getattr(self, "_controller_arrival_turn_listeners", None)
        if not isinstance(listeners, list):
            return
        for listener in list(listeners):
            try:
                listener(dict(turn_result))
            except Exception:
                pass

    def _notify_controller_arrival_turn_update_listeners(
        self,
        update: dict[str, Any],
    ) -> None:
        """Notify UI/session observers about an in-flight controller turn update."""

        listeners = getattr(self, "_controller_arrival_turn_listeners", None)
        if not isinstance(listeners, list):
            return
        payload = dict(update)
        payload["controller_arrival_update"] = True
        for listener in list(listeners):
            try:
                listener(dict(payload))
            except Exception:
                pass

    def _empty_turn_result(
        self,
        thread_id: str,
        *,
        role: str = "system",
        message: str = "",
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the standard empty-result dict with optional field overrides."""

        return {
            "thread_id": thread_id,
            "role": role,
            "message": message,
            "response_items": [],
            "turn_response_type": "none",
            "turn_response_text": "",
            "state": self.get_thread_state(thread_id),
            "answer_candidates": [],
            "streamed_ai_messages": [],
            "visited_nodes": [],
            "step_trace": [],
            "saw_plan_node": False,
            "saw_navigation_activity": False,
            "dispatched": False,
            **overrides,
        }

    def run_controller_arrival_turn(
        self,
        message: str,
        thread_id: str,
        *,
        on_update: Any = None,
        notify_listeners: bool = True,
    ) -> dict[str, Any]:
        """Run one controller-arrival turn without routing through language ingest."""

        content = str(message or "").strip()
        if not content or not self._controller_message_is_arrival(content):
            return self._empty_turn_result(thread_id, message=content)
        if not hasattr(self.agent, "update_state"):
            return self._empty_turn_result(thread_id, message=content)

        config = self._build_agent_config(thread_id)
        role = "system"
        answer_candidates: list[str] = []
        streamed_turn_responses: list[dict[str, Any]] = []
        streamed_ai_messages: list[str] = []
        visited_nodes: list[str] = []
        step_trace: list[dict[str, Any]] = []
        saw_plan_node = False
        saw_navigation_activity = False

        if hasattr(self, "run_memory") and self.run_memory:
            self.run_memory.record_turn_start(
                thread_id=thread_id,
                role=role,
                message=content,
            )

        with self._get_turn_lock(thread_id):
            try:
                state = self.get_thread_state(thread_id)
                baseline_turn_responses = self._extract_turn_response_items(state)
                event = self._build_controller_arrival_event(content, state=state)
                self.agent.update_state(config, {"events": [event], "turn_response_items": []}, as_node = "ingest")
                if hasattr(self.agent, "stream"):
                    for chunk in self.agent.stream({}, config, stream_mode="updates"):
                        for node_name, node_state in chunk.items():
                            if node_state is None:
                                continue
                            def controller_update_callback(update: dict[str, Any]) -> None:
                                if callable(on_update):
                                    try:
                                        on_update(update)
                                    except Exception:
                                        pass
                                if notify_listeners:
                                    update_payload = {
                                        **update,
                                        "thread_id": thread_id,
                                        "role": role,
                                        "message": content,
                                        "dispatched": True,
                                    }
                                    self._notify_controller_arrival_turn_update_listeners(
                                        update_payload
                                    )

                            spn, sna = self._process_stream_node_output(
                                node_name, node_state,
                                thread_id=thread_id,
                                on_update=controller_update_callback,
                                visited_nodes=visited_nodes,
                                step_trace=step_trace,
                                streamed_turn_responses=streamed_turn_responses,
                                answer_candidates=answer_candidates,
                                streamed_ai_messages=streamed_ai_messages,
                                baseline_turn_responses=baseline_turn_responses,
                            )
                            if spn:
                                saw_plan_node = True
                            if sna:
                                saw_navigation_activity = True
                elif hasattr(self.agent, "invoke"):
                    self.agent.invoke({}, config)
            except Exception as exc:
                if self.enable_logging and self.logger:
                    self.logger.log_error(
                        error_msg=(
                            "Controller arrival dispatch failed: "
                            f"{str(exc)}"
                        ),
                        error_type="controller_arrival_dispatch",
                        traceback=traceback.format_exc(),
                    )
                state_snapshot = self.get_thread_state(thread_id)
                return self._empty_turn_result(
                    thread_id,
                    role=role,
                    message=content,
                    turn_response_type="error",
                    turn_response_text=str(exc),
                    state=state_snapshot,
                    visited_nodes=visited_nodes,
                    step_trace=step_trace,
                    saw_plan_node=saw_plan_node,
                    saw_navigation_activity=saw_navigation_activity,
                )

        state_snapshot = self.get_thread_state(thread_id)
        if self._state_has_waiting_navigation(state_snapshot):
            self._ensure_controller_arrival_watchdog(thread_id)
        else:
            self._stop_controller_arrival_watchdog(thread_id)
        turn_response_type, turn_response_text, response_items = self._resolve_turn_response(
            role=role,
            state_snapshot=state_snapshot,
            streamed_turn_responses=streamed_turn_responses,
            answer_candidates=answer_candidates,
            saw_plan_node=saw_plan_node,
            saw_navigation_activity=saw_navigation_activity,
            baseline_turn_responses=baseline_turn_responses,
        )
        turn_result = {
            "thread_id": thread_id,
            "role": role,
            "message": content,
            "response_items": response_items,
            "turn_response_type": turn_response_type,
            "turn_response_text": turn_response_text,
            "state": state_snapshot,
            "answer_candidates": answer_candidates,
            "streamed_ai_messages": streamed_ai_messages,
            "visited_nodes": visited_nodes,
            "step_trace": step_trace,
            "saw_plan_node": saw_plan_node,
            "saw_navigation_activity": saw_navigation_activity,
            "dispatched": True,
        }
        if hasattr(self, "run_memory") and self.run_memory:
            self.run_memory.record_turn_result(turn_result)
        if notify_listeners:
            self._notify_controller_arrival_turn_listeners(turn_result)
        return turn_result

    def _get_turn_lock(self, thread_id: str) -> threading.RLock:
        """Return the per-thread lock used to serialize graph turns."""

        locks = getattr(self, "_turn_locks", None)
        if not isinstance(locks, dict):
            locks = {}
            self._turn_locks = locks
        lock = locks.get(thread_id)
        if isinstance(lock, threading.RLock().__class__):
            return lock
        lock = threading.RLock()
        locks[thread_id] = lock
        return lock

    def _state_has_waiting_navigation(self, state: dict[str, Any]) -> bool:
        """Return True while the persisted graph state is waiting for arrival."""

        if not isinstance(state, dict):
            return False
        active_navigation = state.get("active_navigation")
        if isinstance(active_navigation, dict) and active_navigation.get("task_id") is not None:
            return True
        pending_navigation = state.get("pending_navigation")
        if isinstance(pending_navigation, dict) and pending_navigation.get("task_id") is not None:
            return True
        tasks = state.get("tasks")
        if not isinstance(tasks, dict):
            return False
        for task in tasks.values():
            if not isinstance(task, dict):
                continue
            if (
                str(task.get("status") or "").strip().lower() == "waiting"
                and task.get("wait_for_event") == "navigation_arrived"
            ):
                return True
        return False

    def _stop_controller_arrival_watchdog(self, thread_id: str) -> None:
        """Stop the persistent controller-arrival watchdog for one graph thread."""

        lock = getattr(self, "_controller_arrival_watchdogs_lock", None)
        if lock is None:
            return
        with lock:
            watchdogs = getattr(self, "_controller_arrival_watchdogs", {})
            record = watchdogs.pop(thread_id, None)
        if isinstance(record, dict):
            stop_event = record.get("stop_event")
            if isinstance(stop_event, threading.Event):
                stop_event.set()

    def _ensure_controller_arrival_watchdog(
        self,
        thread_id: str,
        *,
        poll_interval: float = 0.1,
    ) -> None:
        """Keep polling controller arrivals while this thread waits for navigation."""

        controller = getattr(self, "controller", None)
        if controller is None or not hasattr(controller, "check_for_new_messages"):
            return
        if not hasattr(self.agent, "update_state"):
            return

        registry_lock = getattr(self, "_controller_arrival_watchdogs_lock", None)
        if registry_lock is None:
            registry_lock = threading.RLock()
            self._controller_arrival_watchdogs_lock = registry_lock

        watchdogs = getattr(self, "_controller_arrival_watchdogs", None)
        if not isinstance(watchdogs, dict):
            watchdogs = {}
            self._controller_arrival_watchdogs = watchdogs

        with registry_lock:
            existing = watchdogs.get(thread_id)
            if isinstance(existing, dict):
                thread = existing.get("thread")
                if isinstance(thread, threading.Thread) and thread.is_alive():
                    return

            stop_event = threading.Event()
            last_arrival_message: list[str | None] = [None]

            def poll_loop() -> None:
                while not stop_event.is_set():
                    try:
                        state = self.get_thread_state(thread_id)
                        if not self._state_has_waiting_navigation(state):
                            break
                        message = self._poll_controller_message()
                        synthesized_from_status = False
                        if not message:
                            message = self._controller_arrived_status_message()
                            synthesized_from_status = bool(message)
                        if (
                            message
                            and message != last_arrival_message[0]
                            and self._controller_message_is_arrival(message)
                        ):
                            if not self._controller_arrival_matches_active_navigation(message, state):
                                last_arrival_message[0] = message
                                if self.enable_logging and self.logger:
                                    self.logger.log_foreground(
                                        "Controller Arrival Watchdog: ignored stale arrival message for current navigation: {message}".format(
                                            message=message
                                        )
                                    )
                                time.sleep(poll_interval)
                                continue
                            if self._dispatch_controller_arrival_event(message, thread_id):
                                last_arrival_message[0] = message
                                if self.enable_logging and self.logger:
                                    source = "controller status" if synthesized_from_status else "controller message"
                                    self.logger.log_foreground(
                                        "Controller Arrival Watchdog: dispatched physical arrival from {source}: {message}".format(
                                            source=source,
                                            message=message
                                        )
                                    )
                                post_dispatch_state = self.get_thread_state(thread_id)
                                if self._state_has_waiting_navigation(post_dispatch_state):
                                    time.sleep(poll_interval)
                                    continue
                                break
                        if not message:
                            last_arrival_message[0] = None
                        time.sleep(poll_interval)
                    except Exception:
                        time.sleep(poll_interval)
                stop_event.set()
                with registry_lock:
                    current = watchdogs.get(thread_id)
                    if isinstance(current, dict) and current.get("stop_event") is stop_event:
                        watchdogs.pop(thread_id, None)

            thread = threading.Thread(
                target=poll_loop,
                name=f"controller-arrival-watchdog-{thread_id}",
                daemon=True,
            )
            watchdogs[thread_id] = {
                "thread": thread,
                "stop_event": stop_event,
            }
            thread.start()

    def _stdin_ready(self, timeout: float = 0.1) -> bool:
        """Check whether stdin currently has buffered input ready to read."""
        try:
            return bool(select.select([sys.stdin], [], [], timeout)[0])
        except (OSError, ValueError):
            return False

    def _read_user_input_message(
        self,
        *,
        settle_timeout: float = 0.4,
        max_lines: int = 64,
        max_collect_seconds: float = 5.0,
    ) -> str:
        """Read one user message, draining continuation lines from pasted input."""
        first_line = sys.stdin.readline()
        if first_line == "":
            return ""

        lines = [first_line.rstrip("\r\n")]
        collect_deadline = time.monotonic() + max_collect_seconds

        for _ in range(max_lines - 1):
            remaining_time = collect_deadline - time.monotonic()
            if remaining_time <= 0:
                break

            # Keep reading until the input stream stays quiet for a short window.
            if not self._stdin_ready(timeout=min(settle_timeout, remaining_time)):
                break

            next_line = sys.stdin.readline()
            if next_line == "":
                break

            lines.append(next_line.rstrip("\r\n"))
            collect_deadline = time.monotonic() + max_collect_seconds

        # Ignore trailing blank lines while preserving intentional interior breaks.
        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines).strip()

    def _read_stdin_line(self) -> str:
        """Read a single stdin line without triggering paste aggregation."""
        line = sys.stdin.readline()
        if line == "":
            return ""
        return line.rstrip("\r\n")

    def _is_internal_workflow_message(self, content: str) -> bool:
        """Return True for framework-internal progress messages that should not be shown as final answers."""

        stripped = content.strip()
        lowered = stripped.lower()
        return bool(
            re.fullmatch(r"task plan generated with \d+ steps\.?", lowered)
            or lowered == "all tasks have been completed successfully!"
        )

    def _select_user_facing_answer(self, candidates: list[str]) -> str:
        """Choose the latest non-internal AI answer from streamed updates."""

        filtered = [
            candidate
            for candidate in candidates
            if candidate.strip() and not self._is_internal_workflow_message(candidate)
        ]
        if not filtered:
            return ""
        return filtered[-1]

    def _normalize_turn_response_type(self, value: Any) -> str:
        """Normalize reply-type values into the supported turn contract."""

        normalized = str(value or "").strip().lower()
        if normalized in {"result", "progress", "error", "none"}:
            return normalized
        return ""

    def _extract_turn_response(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        """Extract one turn reply from state or node payload."""

        response_items = normalize_turn_response_items(payload.get("turn_response_items"))
        if response_items:
            response_type, response_text, _ = derive_headline_turn_response(
                response_items
            )
            return response_type, response_text or ""

        response_text = str(
            payload.get("turn_response_text")
            or payload.get("user_facing_response")
            or ""
        ).strip()
        response_type = self._normalize_turn_response_type(
            payload.get("turn_response_type")
        )
        if response_text and not response_type:
            response_type = "result"
        if not response_text:
            response_type = response_type or "none"
        return response_type, response_text

    def _extract_turn_response_items(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract all response items from one payload."""

        response_items = normalize_turn_response_items(payload.get("turn_response_items"))
        if response_items:
            return [dict(item) for item in response_items]

        response_type, response_text = self._extract_turn_response(payload)
        if not response_text:
            return []
        return [
            {
                "response_type": response_type or "result",
                "response_text": response_text,
                "response_id": payload.get("turn_response_id")
                or payload.get("user_facing_response_id"),
            }
        ]

    def _filter_turn_response_delta(
        self,
        response_items: Any,
        baseline_items: Any,
    ) -> list[dict[str, Any]]:
        """Return response items that were newly produced after turn start."""

        normalized_items = normalize_turn_response_items(response_items)
        normalized_baseline = normalize_turn_response_items(baseline_items)
        if not normalized_items or not normalized_baseline:
            return [dict(item) for item in normalized_items]

        baseline_ids = {
            str(item.get("response_id") or "").strip()
            for item in normalized_baseline
            if str(item.get("response_id") or "").strip()
        }
        baseline_keys = {
            (
                str(item.get("response_type") or "").strip(),
                str(item.get("response_text") or "").strip(),
            )
            for item in normalized_baseline
        }
        delta_items: list[dict[str, Any]] = []
        for item in normalized_items:
            response_id = str(item.get("response_id") or "").strip()
            response_key = (
                str(item.get("response_type") or "").strip(),
                str(item.get("response_text") or "").strip(),
            )
            if response_id and response_id in baseline_ids:
                continue
            if response_key in baseline_keys:
                continue
            delta_items.append(dict(item))
        return delta_items

    def _resolve_turn_response(
        self,
        *,
        role: str,
        state_snapshot: dict[str, Any],
        streamed_turn_responses: list[dict[str, Any]],
        answer_candidates: list[str],
        saw_plan_node: bool,
        saw_navigation_activity: bool,
        baseline_turn_responses: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Resolve the authoritative headline plus the full ordered response stream."""

        normalized_stream_items = normalize_turn_response_items(streamed_turn_responses)
        if normalized_stream_items:
            response_type, response_text, _ = derive_headline_turn_response(
                normalized_stream_items
            )
            return (
                response_type,
                response_text or "",
                [dict(item) for item in normalized_stream_items],
            )

        state_response_items = self._extract_turn_response_items(state_snapshot)
        state_response_items = self._filter_turn_response_delta(
            state_response_items,
            baseline_turn_responses or [],
        )
        if state_response_items:
            response_type, response_text, _ = derive_headline_turn_response(
                normalize_turn_response_items(state_response_items)
            )
            return response_type, response_text or "", state_response_items

        if role != "user":
            return "none", "", []

        if saw_plan_node or saw_navigation_activity:
            return "none", "", []

        fallback_answer = self._select_user_facing_answer(answer_candidates)
        if fallback_answer:
            fallback_items = [
                {
                    "response_type": "result",
                    "response_text": fallback_answer,
                }
            ]
            return "result", fallback_answer, fallback_items
        return "none", "", []

    def _message_has_navigation_activity(self, message: Any) -> bool:
        """Return True when a streamed message indicates navigation is in progress."""

        if isinstance(message, ToolMessage):
            return getattr(message, "name", "") in {"go_to_keyframe", "go_to_position"}

        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            return any(
                tool_call.get("name") in {"go_to_keyframe", "go_to_position"}
                for tool_call in tool_calls
                if isinstance(tool_call, dict)
            )

        return False

    def _process_stream_node_output(
        self,
        node_name: str,
        node_state: dict[str, Any],
        *,
        thread_id: str,
        on_update: Any,
        visited_nodes: list[str],
        step_trace: list[dict[str, Any]],
        streamed_turn_responses: list[dict[str, Any]],
        answer_candidates: list[str],
        streamed_ai_messages: list[str],
        baseline_turn_responses: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, bool]:
        """Process one node state from a stream chunk, updating accumulators in place.

        Returns updated (saw_plan_node, saw_navigation_activity).
        """

        saw_plan_node = False
        saw_navigation_activity = False
        visited_nodes.append(str(node_name))
        if isinstance(node_state, dict):
            step_summary = self._summarize_stream_update(
                str(node_name), node_state
            )
            if "turn_response_items" in step_summary:
                delta_items = self._filter_turn_response_delta(
                    step_summary.get("turn_response_items"),
                    baseline_turn_responses or [],
                )
                if delta_items:
                    step_summary["turn_response_items"] = delta_items
                else:
                    step_summary.pop("turn_response_items", None)
            step_trace.append(step_summary)
            if hasattr(self, "run_memory") and self.run_memory:
                self.run_memory.record_stream_update(
                    thread_id=thread_id,
                    node_name=str(node_name),
                    step_summary=step_summary,
                )
            if callable(on_update):
                try:
                    on_update(
                        {
                            "thread_id": thread_id,
                            "node_name": str(node_name),
                            "node_state": node_state,
                            "step_summary": step_summary,
                            "visited_nodes": list(visited_nodes),
                            "step_trace": list(step_trace),
                            "state": self.get_thread_state(thread_id),
                        }
                    )
                except Exception:
                    pass
        if node_name == "plan":
            saw_plan_node = True

        response_items = self._extract_turn_response_items(node_state)
        response_items = self._filter_turn_response_delta(
            response_items,
            baseline_turn_responses or [],
        )
        if response_items:
            seen_response_ids = {
                str(item.get("response_id") or "").strip()
                for item in streamed_turn_responses
                if str(item.get("response_id") or "").strip()
            }
            seen_response_keys = {
                (
                    str(item.get("response_type") or "").strip(),
                    str(item.get("response_text") or "").strip(),
                )
                for item in streamed_turn_responses
            }
            for item in response_items:
                response_id = str(item.get("response_id") or "").strip()
                response_key = (
                    str(item.get("response_type") or "").strip(),
                    str(item.get("response_text") or "").strip(),
                )
                if response_id and response_id in seen_response_ids:
                    continue
                if response_key in seen_response_keys:
                    continue
                streamed_turn_responses.append(item)
                if response_id:
                    seen_response_ids.add(response_id)
                seen_response_keys.add(response_key)

        if "messages" not in node_state:
            return saw_plan_node, saw_navigation_activity

        for msg in node_state["messages"]:
            if self._message_has_navigation_activity(msg):
                saw_navigation_activity = True
            if msg.content and msg.content.strip() and isinstance(msg, AIMessage):
                content = msg.content.strip()
                answer_candidates.append(content)
                streamed_ai_messages.append(content)

        return saw_plan_node, saw_navigation_activity

    def _summarize_stream_update(
        self,
        node_name: str,
        node_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a compact per-node update summary for UI debugging."""

        summary: dict[str, Any] = {
            "node": str(node_name),
        }

        current_task_id = node_state.get("current_task_id")
        if current_task_id is not None:
            summary["current_task_id"] = current_task_id

        current_plan_id = node_state.get("current_plan_id")
        if current_plan_id is not None:
            summary["current_plan_id"] = current_plan_id

        next_action = node_state.get("next_action")
        if isinstance(next_action, dict) and next_action.get("type"):
            summary["next_action"] = next_action

        response_type, response_text = self._extract_turn_response(node_state)
        if response_text:
            summary["turn_response_type"] = response_type
            summary["turn_response_text"] = response_text
            summary["user_facing_response"] = response_text
        response_items = self._extract_turn_response_items(node_state)
        if response_items:
            summary["turn_response_items"] = response_items

        latest_event = None
        events = node_state.get("events")
        if isinstance(events, list) and events:
            candidate = events[-1]
            if isinstance(candidate, dict):
                payload = candidate.get("payload", {}) or {}
                latest_event = {
                    "type": candidate.get("type"),
                    "task_id": candidate.get("task_id"),
                    "summary": payload.get("summary") or payload.get("content"),
                }
                summary["latest_event"] = latest_event

        tasks = node_state.get("tasks")
        if isinstance(tasks, dict) and tasks:
            focus_task_id = current_task_id
            if focus_task_id is None and isinstance(latest_event, dict):
                focus_task_id = latest_event.get("task_id")

            if focus_task_id in tasks:
                focus_task = tasks[focus_task_id]
                if isinstance(focus_task, dict):
                    summary["focus_task"] = {
                        "task_id": focus_task_id,
                        "status": focus_task.get("status"),
                        "description": focus_task.get("description"),
                    }

        return summary

    def _compose_multiline_message(self) -> str | None:
        """Read a multi-line user message until an explicit submit/cancel command."""
        print("Entered multi-line compose mode.")
        print("Type your message line by line.")
        print("Use `/send` to submit or `/cancel` to discard.\n")

        lines: list[str] = []
        while True:
            print("... ", end="", flush=True)
            line = self._read_stdin_line()
            stripped_line = line.strip()

            if stripped_line == "/send":
                message = "\n".join(lines).strip()
                if not message:
                    print("Multi-line input is empty. Nothing sent.\n")
                    return ""
                return message

            if stripped_line == "/cancel":
                print("Multi-line input cancelled.\n")
                return None

            if stripped_line == "/exit":
                return "/exit"

            lines.append(line)

    def _build_agent_config(self, thread_id: str) -> dict[str, Any]:
        """Build a reusable LangGraph invocation config for one thread."""

        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}

    def get_thread_state(self, thread_id: str) -> dict[str, Any]:
        """Fetch the latest persisted state for one LangGraph thread."""

        if not hasattr(self.agent, "get_state"):
            return {}

        try:
            snapshot = self.agent.get_state(self._build_agent_config(thread_id))
        except Exception:
            return {}

        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            return values
        if isinstance(snapshot, dict):
            return snapshot
        return {}

    def run_message_turn(
        self,
        message: str,
        thread_id: str,
        role: str,
        *,
        on_update: Any = None,
        original_message: str | None = None,
    ) -> dict[str, Any]:
        """Run one message turn and return structured results for UI or CLI callers."""

        config_dict = self._build_agent_config(thread_id)
        answer_candidates: list[str] = []
        streamed_turn_responses: list[dict[str, Any]] = []
        streamed_ai_messages: list[str] = []
        visited_nodes: list[str] = []
        step_trace: list[dict[str, Any]] = []
        saw_plan_node = False
        saw_navigation_activity = False
        if hasattr(self, "run_memory") and self.run_memory:
            self.run_memory.record_turn_start(
                thread_id=thread_id,
                role=role,
                message=message,
            )

        with self._get_turn_lock(thread_id):
            baseline_turn_responses = self._extract_turn_response_items(
                self.get_thread_state(thread_id)
            )
            message_payload: dict[str, Any] = {"role": role, "content": message}
            clean_original_message = str(original_message or "").strip()
            if clean_original_message and clean_original_message != str(message or "").strip():
                message_payload["additional_kwargs"] = {
                    "original_content": clean_original_message,
                    "agent_content": str(message or ""),
                }
            for chunk in self.agent.stream(
                {"messages": [message_payload]},
                config_dict,
                stream_mode="updates",
            ):
                for node_name, node_state in chunk.items():
                    if node_state is None:
                        continue
                    spn, sna = self._process_stream_node_output(
                        node_name, node_state,
                        thread_id=thread_id,
                        on_update=on_update,
                        visited_nodes=visited_nodes,
                        step_trace=step_trace,
                        streamed_turn_responses=streamed_turn_responses,
                        answer_candidates=answer_candidates,
                        streamed_ai_messages=streamed_ai_messages,
                        baseline_turn_responses=baseline_turn_responses,
                    )
                    if spn:
                        saw_plan_node = True
                    if sna:
                        saw_navigation_activity = True

        state_snapshot = self.get_thread_state(thread_id)
        if self._state_has_waiting_navigation(state_snapshot):
            self._ensure_controller_arrival_watchdog(thread_id)
        else:
            self._stop_controller_arrival_watchdog(thread_id)
        turn_response_type, turn_response_text, response_items = self._resolve_turn_response(
            role=role,
            state_snapshot=state_snapshot,
            streamed_turn_responses=streamed_turn_responses,
            answer_candidates=answer_candidates,
            saw_plan_node=saw_plan_node,
            saw_navigation_activity=saw_navigation_activity,
            baseline_turn_responses=baseline_turn_responses,
        )
        turn_result = {
            "thread_id": thread_id,
            "role": role,
            "message": message,
            "response_items": response_items,
            "turn_response_type": turn_response_type,
            "turn_response_text": turn_response_text,
            "state": state_snapshot,
            "answer_candidates": answer_candidates,
            "streamed_ai_messages": streamed_ai_messages,
            "visited_nodes": visited_nodes,
            "step_trace": step_trace,
            "saw_plan_node": saw_plan_node,
            "saw_navigation_activity": saw_navigation_activity,
        }
        if hasattr(self, "run_memory") and self.run_memory:
            self.run_memory.record_turn_result(turn_result)
        return turn_result

    def process_message(
        self,
        message: str,
        thread_id: str,
        role: str
    ):
        """Stream a single user/system message through the agent and print the latest AI reply."""

        try:
            turn_result = self.run_message_turn(message, thread_id, role)
            turn_response_text = str(
                turn_result.get("turn_response_text") or ""
            ).strip()
            if turn_response_text:
                print(f"{Colors.ANSWER}Final Answer:{Colors.RESET}")
                print(f"{Colors.ANSWER} {turn_response_text} {Colors.RESET}\n")
            return turn_response_text

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Agent processing failed: {str(e)}"
            print(f"{Colors.ERROR}Error: {error_msg}{Colors.RESET}")

            if self.enable_logging and self.logger:
                self.logger.log_error(
                    error_msg=error_msg,
                    error_type="agent_processing",
                    traceback=traceback.format_exc(),
                )
            return ""

    def interactive_session(self):
        """Simple CLI loop for manual interaction and live controller/system message handling."""
        print(f"🤖 {self.name} is ready! Type 'exit' to quit.\n")
        print("Single-line input sends immediately. Use `/multi` for multi-line compose mode.\n")

        if self.enable_logging and self.logger:
            pass

        thread_id = "interactive_session"
        latest_controller_msg = ""
        is_new_controller_msg = False
        while True:
            try:
                current_controller_msg = self._poll_controller_message()
                if not current_controller_msg:
                    latest_controller_msg = ""
                if current_controller_msg != latest_controller_msg:
                    is_new_controller_msg = True
                    latest_controller_msg = current_controller_msg
                else:
                    is_new_controller_msg = False

                if is_new_controller_msg or self._stdin_ready(timeout=0.1):
                    if is_new_controller_msg:
                        message = latest_controller_msg
                        is_new_controller_msg = False
                        role = "system"
                        if message.strip():
                            print(f"{Colors.USER}Controller: {message}{Colors.RESET}\n")
                    else:
                        role = "user"
                        message = self._read_user_input_message()
                        stripped_message = message.strip()

                        if stripped_message == "/multi":
                            message = self._compose_multiline_message()
                            if message is None:
                                continue
                            if message == "":
                                continue

                        print(f"{Colors.USER}User: {message}{Colors.RESET}\n")

                    if message.lower() in ["exit", "quit", "bye", "/exit"]:
                        if self.enable_logging and self.logger:
                            print(
                                f"\n Session saved to: {self.logger.get_session_dir()}"
                            )
                        print("Goodbye!")
                        break

                    if not message.strip():
                        continue

                    self.process_message(
                        message,
                        thread_id,
                        role
                    )

            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"{Colors.ERROR}Error: Unexpected error: {str(e)}{Colors.RESET}")


__all__ = ["AsyncAgent"]
