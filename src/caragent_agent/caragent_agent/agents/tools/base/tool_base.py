import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional

"""Tool base classes and registration utilities."""

ToolStatus = Literal["ok", "partial", "blocked", "error"]


def _normalize_jsonable(value: Any) -> Any:
    """Convert common runtime values into JSON-safe Python primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_jsonable(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return _normalize_jsonable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _normalize_jsonable(vars(value))
        except Exception:
            pass

    return str(value)


class ToolBase(ABC):
    """Abstract base class for tools.

    Subclasses must implement 'execute'. The base class provides a
    'get_tool_schema' helper that introspects 'execute' signature
    to produce a simple JSON-schema-like description for LLM integration.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        *,
        capability_tags: Optional[Sequence[str]] = None,
    ):
        self.name = name
        self.description = description
        self._scene_memory = None
        self._run_memory = None
        self.capability_tags = tuple(
            str(tag).strip()
            for tag in (capability_tags or ())
            if str(tag).strip()
        )
        self.tool_contract_version = "tool_result_v1"
        
    @property
    def scene_memory(self):
        return self._scene_memory
        
    @scene_memory.setter  
    def scene_memory(self, value):
        self._scene_memory = value

    @property
    def run_memory(self):
        """Return the optional session-level run-memory reference."""

        return self._run_memory

    @run_memory.setter
    def run_memory(self, value):
        """Attach one session-level run-memory reference to the tool."""

        self._run_memory = value
        
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool behavior. Should be overridden by subclasses.

    Use keyword arguments to make tools compatible with structured tool
    invocation frameworks.
        """
        raise NotImplementedError()
        
    def validate_inputs(self, **kwargs) -> bool:
        """Optional input validation hook. Return True when inputs are valid."""
        return True

    @staticmethod
    def to_jsonable(value: Any) -> Any:
        """Normalize arbitrary runtime values into JSON-safe content."""

        return _normalize_jsonable(value)

    def build_tool_result(
        self,
        *,
        status: ToolStatus,
        summary: str,
        data: Any = None,
        error: Optional[dict[str, Any] | str] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one normalized structured tool-result payload."""

        normalized_error: Any = None
        if isinstance(error, str):
            normalized_error = {"message": error}
        elif error is not None:
            normalized_error = self.to_jsonable(error)

        normalized_provenance = self.to_jsonable(provenance or {})
        if not isinstance(normalized_provenance, dict):
            normalized_provenance = {"source_type": "unknown"}
        normalized_provenance.setdefault("source_type", "unknown")
        normalized_provenance.setdefault("tool_name", self.name)
        normalized_provenance.setdefault("contract_version", self.tool_contract_version)

        return {
            "status": status,
            "summary": str(summary or "").strip(),
            "data": self.to_jsonable(data),
            "error": normalized_error,
            "provenance": normalized_provenance,
        }

    def ok(
        self,
        summary: str,
        *,
        data: Any = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one successful tool-result payload."""

        return self.build_tool_result(
            status="ok",
            summary=summary,
            data=data,
            provenance=provenance,
        )

    def partial(
        self,
        summary: str,
        *,
        data: Any = None,
        error: Optional[dict[str, Any] | str] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one partially successful tool-result payload."""

        return self.build_tool_result(
            status="partial",
            summary=summary,
            data=data,
            error=error,
            provenance=provenance,
        )

    def blocked(
        self,
        summary: str,
        *,
        data: Any = None,
        error: Optional[dict[str, Any] | str] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one blocked tool-result payload."""

        return self.build_tool_result(
            status="blocked",
            summary=summary,
            data=data,
            error=error,
            provenance=provenance,
        )

    def error_result(
        self,
        summary: str,
        *,
        data: Any = None,
        error: Optional[dict[str, Any] | str] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one failed tool-result payload."""

        return self.build_tool_result(
            status="error",
            summary=summary,
            data=data,
            error=error,
            provenance=provenance,
        )
        
    def get_tool_schema(self) -> Dict:
        """Build a simplified tool schema from the `execute` signature.

    Returns a dict with 'name', 'description', and a 'parameters' block
    compatible with basic LLM structured-call formats.
        """
        sig = inspect.signature(self.execute)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'kwargs':
                continue
                
            param_info = {
                "type": self._get_param_type(param),
                "description": f"Parameter {param_name}"
            }
            
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
                
            parameters[param_name] = param_info
            
        return {
            "name": self.name,
            "description": self.description,
            "x-capability-tags": list(self.capability_tags),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": [name for name, param in sig.parameters.items() 
                           if param.default == inspect.Parameter.empty and name != 'kwargs']
            }
        }
        
    def _get_param_type(self, param) -> str:
        """Map a function parameter annotation to a simple type string.

        Recognizes list/dict generics and basic builtin types.
        """
        if param.annotation == inspect.Parameter.empty:
            return "string"
            
        annotation = param.annotation
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is list:
                return "array"
            elif annotation.__origin__ is dict:
                return "object"
                
        if annotation in (int, float):
            return "number"
        elif annotation is bool:
            return "boolean"
        else:
            return "string"


def tool(name: str, description: str):
    """Decorator to mark a plain function as a tool with metadata.

    Attaches `_tool_name`, `_tool_description`, and `_is_tool` attributes
    to the wrapped function so callers can detect tool functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        # 添加工具元数据
        wrapper._tool_name = name
        wrapper._tool_description = description
        wrapper._is_tool = True
        
        return wrapper
    return decorator


class ToolRegistry:
    """Simple in-memory registry for ToolBase instances.

    Provides lookup by name and helpers to list tools and their schemas.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolBase] = {}
        
    def register(self, tool: ToolBase):
        """Register a ToolBase instance under its `name`."""
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[ToolBase]:
        """Return the tool registered under `name`, or None if missing."""
        return self._tools.get(name)
        
    def get_all_tools(self) -> List[ToolBase]:
        """Return a list of all registered ToolBase instances."""
        return list(self._tools.values())
        
    def get_tool_schemas(self) -> List[Dict]:
        """Return schema descriptions for all registered tools."""
        return [tool.get_tool_schema() for tool in self._tools.values()]


# Module-level registry instance for convenience.
tool_registry = ToolRegistry()
