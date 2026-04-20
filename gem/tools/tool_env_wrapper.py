from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolEnvWrapper:
    """Minimal shim wrapper.

    ROLL uses this symbol mainly for type/import compatibility.
    """

    env: Any

