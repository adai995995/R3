from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from gem.tools.base_tool import BaseTool
from gem.utils.sandbox import run_python


@dataclass
class PythonCodeTool(BaseTool):
    name: str = "python_code"

    def __call__(self, code: str, timeout_s: float = 5.0) -> Dict[str, Any]:
        return run_python(code=code, timeout_s=timeout_s)

