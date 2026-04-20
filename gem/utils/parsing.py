from __future__ import annotations

import re
from typing import Optional


def extract_last_boxed_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\\{([^}]*)\\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_code_from_model(text: str) -> str:
    # Best-effort: prefer fenced code blocks; else return raw text.
    m = re.search(r"```(?:python)?\\s*([\\s\\S]*?)\\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

