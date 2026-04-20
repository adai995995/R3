from __future__ import annotations

from enum import Enum


class TerminalState(str, Enum):
    NON_TERMINAL = "non_terminal"
    TERMINAL = "terminal"


# Backward-compatible alias used by some env wrappers.
TERMINAL_STATE = TerminalState

