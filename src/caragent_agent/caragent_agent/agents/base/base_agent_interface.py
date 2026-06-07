"""Abstract agent base interfaces plus tool registration utilities."""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from caragent_agent.agents.tools.base.tool_base import ToolBase, tool_registry
from caragent_agent.impression_graph.scene_memory import SceneMemory
from caragent_agent.utils.conversation_logger import ConversationLogger


class BaseAgent(ABC):
    """Abstract base class for agents that use scene memory and tools."""
    
    def __init__(self, name: str, scene_memory: SceneMemory, enable_logging: bool = True):
        self.name = name
        self.scene_memory = scene_memory
        self._tools: Dict[str, ToolBase] = {}
        self.enable_logging = enable_logging
        self.logger: Optional[ConversationLogger] = None
        self._setup_tools()
        
    @abstractmethod
    def _setup_tools(self):
        """Register tools the agent can call."""
        pass
    
    def set_logger(self, logger: ConversationLogger):
        """Attach a conversation logger instance."""
        self.logger = logger
        
    def get_logger(self) -> Optional[ConversationLogger]:
        """Return the attached conversation logger if set."""
        return self.logger
        
    def register_tool(self, tool: ToolBase):
        """Register a tool and inject scene memory reference."""
        tool.scene_memory = self.scene_memory
        tool.run_memory = getattr(self, "run_memory", None)
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[ToolBase]:
        """Fetch a tool by name if registered."""
        return self._tools.get(name)
        
    def get_all_tools(self) -> List[ToolBase]:
        """Return all registered tools."""
        return list(self._tools.values())
        
    def get_tool_schemas(self) -> List[Dict]:
        """Return tool schemas for LLM function/tool calling."""
        return [tool.get_tool_schema() for tool in self._tools.values()]
        
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool by name with provided kwargs."""
        if tool_name not in self._tools:
            return f"Tool '{tool_name}' not found"
            
        try:
            tool = self._tools[tool_name]
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
            
    @abstractmethod
    def process_message(self, message: str, thread_id: str, role: str) -> str:
        """Process a user message and return the reply."""
        pass


class ToolOrchestrate:
    """Utility to batch-register and manage tools."""
    
    def __init__(self):
        self.tools: Dict[str, ToolBase] = {}
        
    def register_tools_from_modules(self, modules: List[str]):
        """Import and register tools from a list of module paths."""
        for module_path in modules:
            self._import_and_register_tools(module_path)
            
    def _import_and_register_tools(self, module_path: str):
        """Import a module and register any ToolBase subclasses found."""
        try:
            module = __import__(module_path, fromlist=[''])

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (isinstance(attr, type) and 
                    issubclass(attr, ToolBase) and 
                    attr != ToolBase):

                    tool_instance = attr()
                    self.register_tool(tool_instance)
                    
        except ImportError as e:
            print(f"Failed to import module {module_path}: {e}")
            
    def register_tool(self, tool: ToolBase):
        """Register a single tool and add it to the shared registry."""
        self.tools[tool.name] = tool
        tool_registry.register(tool)
        
    def get_tools_by_category(self, category: str) -> List[ToolBase]:
        """Return tools; category filtering can be expanded later."""
        return list(self.tools.values())
        
    def create_tool_set(self, tool_names: List[str]) -> List[ToolBase]:
        """Return a list of tools matching the provided names."""
        tools = []
        for name in tool_names:
            if name in self.tools:
                tools.append(self.tools[name])
        return tools
