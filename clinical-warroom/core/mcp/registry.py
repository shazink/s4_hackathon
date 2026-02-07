"""
Clinical War Room - MCP Registry

Tool registration and discovery.
All tools MUST be registered before use.
"""

from typing import Dict, Callable, Optional, Any
from dataclasses import dataclass
from core.mcp.schemas import ToolDefinition, ToolInput, ToolOutput, ToolCategory
from core.exceptions import ToolNotFoundError


@dataclass
class RegisteredTool:
    """A tool registered in the MCP registry."""
    definition: ToolDefinition
    handler: Callable[[ToolInput], ToolOutput]
    enabled: bool = True


class MCPRegistry:
    """
    Central registry for all MCP tools.
    
    Design principles:
    - Tools MUST be registered before use
    - Tools are identified by unique names
    - Tools can be enabled/disabled at runtime
    - Registry provides discovery and validation
    """
    
    _instance: Optional["MCPRegistry"] = None
    
    def __new__(cls) -> "MCPRegistry":
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, RegisteredTool] = {}
        return cls._instance
    
    def register(
        self, 
        definition: ToolDefinition, 
        handler: Callable[[ToolInput], ToolOutput]
    ) -> None:
        """
        Register a tool with its handler.
        
        Args:
            definition: Tool schema and metadata
            handler: Function that executes the tool
        """
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        
        self._tools[definition.name] = RegisteredTool(
            definition=definition,
            handler=handler,
            enabled=True
        )
    
    def get(self, name: str) -> RegisteredTool:
        """
        Get a registered tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            RegisteredTool instance
            
        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> list:
        """
        List all registered tools.
        
        Args:
            category: Optional filter by category
            
        Returns:
            List of tool definitions
        """
        tools = []
        for tool in self._tools.values():
            if category is None or tool.definition.category == category:
                tools.append(tool.definition.to_dict())
        return tools
    
    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def enable(self, name: str) -> None:
        """Enable a tool."""
        tool = self.get(name)
        tool.enabled = True
    
    def disable(self, name: str) -> None:
        """Disable a tool (it remains registered but won't execute)."""
        tool = self.get(name)
        tool.enabled = False
    
    def clear(self) -> None:
        """Clear all registered tools (for testing)."""
        self._tools.clear()


# Global registry instance
registry = MCPRegistry()


def register_tool(definition: ToolDefinition):
    """
    Decorator to register a tool handler.
    
    Usage:
        @register_tool(MY_TOOL_SCHEMA)
        def my_tool_handler(input: ToolInput) -> ToolOutput:
            ...
    """
    def decorator(handler: Callable[[ToolInput], ToolOutput]):
        registry.register(definition, handler)
        return handler
    return decorator
