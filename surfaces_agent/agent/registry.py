# surfaces_agent/agent/registry.py
import inspect
from typing import Callable, Dict, Any, Type
from pydantic import BaseModel

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}

    def register(self, name: str, description: str, schema: Type[BaseModel], func: Callable):
        """Registers a deterministic scientific module."""
        self._tools[name] = {
            "description": description,
            "schema": schema,
            "func": func
        }

    def get_llm_tools(self) -> list[dict]:
        """Converts registered tools into the JSON schema required by LLMs."""
        llm_tools = []
        for name, tool_data in self._tools.items():
            schema = tool_data["schema"].schema()
            
            # OpenAI/Gemini standard function calling format
            llm_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_data["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                }
            })
        return llm_tools

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes the mapped Python function with validated arguments."""
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."
        
        tool = self._tools[name]
        try:
            # Validate LLM output against the Pydantic schema
            validated_args = tool["schema"](**arguments)
            # Execute the deterministic scientific code
            result = tool["func"](**validated_args.model_dump())
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"