"""Optional dependency stubs for local offline audit scripts.

These stubs are only for import-time compatibility on development laptops that
do not have the ROS/agent LLM stack installed. API audits should still run in an
environment with real dependencies and credentials.
"""

from __future__ import annotations

import sys
import types


def install_offline_import_stubs() -> None:
    if "langchain_core" not in sys.modules:
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")

    if "langchain_core.language_models" not in sys.modules:
        language_models = types.ModuleType("langchain_core.language_models")

        class BaseChatModel:  # pragma: no cover - import shim only
            pass

        class BaseLLM:  # pragma: no cover
            pass

        language_models.BaseChatModel = BaseChatModel
        language_models.BaseLLM = BaseLLM
        sys.modules["langchain_core.language_models"] = language_models

    if "langchain_core.messages" not in sys.modules:
        messages = types.ModuleType("langchain_core.messages")

        class BaseMessage:  # pragma: no cover
            def __init__(self, content=None, **kwargs):
                self.content = content

        class HumanMessage(BaseMessage):  # pragma: no cover
            pass

        class SystemMessage(BaseMessage):  # pragma: no cover
            pass

        class AIMessage(BaseMessage):  # pragma: no cover
            pass

        class ToolMessage(BaseMessage):  # pragma: no cover
            def __init__(self, content=None, name=None, **kwargs):
                super().__init__(content=content, **kwargs)
                self.name = name

        messages.BaseMessage = BaseMessage
        messages.HumanMessage = HumanMessage
        messages.SystemMessage = SystemMessage
        messages.AIMessage = AIMessage
        messages.ToolMessage = ToolMessage
        sys.modules["langchain_core.messages"] = messages

    if "langchain_core.tools" not in sys.modules:
        tools = types.ModuleType("langchain_core.tools")

        class BaseTool:  # pragma: no cover
            pass

        class StructuredTool(BaseTool):  # pragma: no cover
            @classmethod
            def from_function(cls, *args, **kwargs):
                return cls()

        tools.BaseTool = BaseTool
        tools.StructuredTool = StructuredTool
        sys.modules["langchain_core.tools"] = tools

    if "langchain_core.callbacks" not in sys.modules:
        callbacks = types.ModuleType("langchain_core.callbacks")

        class AsyncCallbackHandler:  # pragma: no cover
            pass

        callbacks.AsyncCallbackHandler = AsyncCallbackHandler
        sys.modules["langchain_core.callbacks"] = callbacks

    if "langgraph.graph" not in sys.modules:
        sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
        graph = types.ModuleType("langgraph.graph")
        graph.END = "__end__"
        sys.modules["langgraph.graph"] = graph

    if "langgraph.graph.message" not in sys.modules:
        graph_message = types.ModuleType("langgraph.graph.message")

        def add_messages(left, right):  # pragma: no cover
            return (left or []) + (right or [])

        graph_message.add_messages = add_messages
        sys.modules["langgraph.graph.message"] = graph_message

    if "langgraph.prebuilt.tool_node" not in sys.modules:
        prebuilt = types.ModuleType("langgraph.prebuilt")
        tool_node = types.ModuleType("langgraph.prebuilt.tool_node")

        class ToolNode:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def invoke(self, *args, **kwargs):
                raise RuntimeError("ToolNode is unavailable in offline audit stubs")

        tool_node.ToolNode = ToolNode
        prebuilt.tool_node = tool_node
        sys.modules["langgraph.prebuilt"] = prebuilt
        sys.modules["langgraph.prebuilt.tool_node"] = tool_node

    if "caragent_agent.third_party.from_langgraph.react_agent" not in sys.modules:
        react_agent = types.ModuleType("caragent_agent.third_party.from_langgraph.react_agent")

        def create_react_agent(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("create_react_agent is unavailable in offline audit stubs")

        react_agent.create_react_agent = create_react_agent
        sys.modules["caragent_agent.third_party.from_langgraph.react_agent"] = react_agent

    if "langchain_openai" not in sys.modules:
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                pass

        langchain_openai.ChatOpenAI = ChatOpenAI
        sys.modules["langchain_openai"] = langchain_openai


__all__ = ["install_offline_import_stubs"]
