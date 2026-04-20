from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BaseTool:
    """Minimal BaseTool shim for agentic tool wrappers."""

    name: str = "base_tool"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def to_json_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": "", "parameters": {"type": "object", "properties": {}}}

