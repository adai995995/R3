# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/axon-rl/gem/blob/main/gem/tools/search_tool.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

import requests

from gem.tools.base_tool import BaseTool

_DEFAULT_TIMEOUT = 5.0

_SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)


@dataclass
class SearchTool(BaseTool):
    """HTTP dense retrieval client (msgpack body), compatible with GEM retrieval_server."""

    tool_type: str = field(default="search", repr=False)
    name: str = "search"
    num_workers: int = 1
    search_url: Optional[str] = None
    topk: int = 3
    timeout: float = _DEFAULT_TIMEOUT
    api_format: str = "msgpack"

    def __post_init__(self) -> None:
        self._search_url_resolved = self.search_url is not None

    def _parse_action(self, action: str) -> Tuple[str, str, bool]:
        match = _SEARCH_PATTERN.search(action)
        if not match:
            return "", "", False
        parsed_query = match.group(1).strip()
        parsed_action = action[: match.end()]
        return parsed_query, parsed_action, True

    def _resolve_search_url(self) -> str:
        if not self._search_url_resolved:
            self.search_url = self.search_url or os.environ.get("SEARCH_URL")
            self._search_url_resolved = True
        if not self.search_url:
            raise ValueError("search_url must be provided for SearchTool (config or SEARCH_URL).")
        return self.search_url

    def _search(self, query: str) -> str:
        url = self._resolve_search_url()
        try:
            if self.api_format == "search_r1_json":
                payload = {"queries": [query], "topk": self.topk, "return_scores": True}
                response = requests.post(url, json=payload, timeout=self.timeout)
            else:
                import msgspec
                payload = {"query": query, "topk": self.topk, "return_scores": True}
                response = requests.post(
                    url,
                    data=msgspec.msgpack.encode(payload),
                    timeout=self.timeout,
                )
            response.raise_for_status()
            if self.api_format == "search_r1_json":
                decoded = response.json()
                # Search-R1 returns a batch-shaped result: [[doc, ...]].
                if decoded.get("result") and isinstance(decoded["result"][0], list):
                    decoded["result"] = decoded["result"][0]
            else:
                import msgspec
                decoded = msgspec.msgpack.decode(response.content)
            result = decoded.get("result", [])
            return self._passages2string(result)
        except Exception as e:
            return f"[SearchTool Error: {e}]"

    def _passages2string(self, result: list) -> str:
        lines: list[str] = []
        for idx, doc_item in enumerate(result):
            content = doc_item["document"]["contents"]
            parts = content.split("\n")
            title = parts[0] if parts else ""
            text = "\n".join(parts[1:]) if len(parts) > 1 else ""
            lines.append(f"Doc {idx + 1}(Title: {title}) {text}\n")
        return "".join(lines)

    def instruction_string(self) -> str:
        return (
            "You have access to a search engine to help answer questions.\n\n"
            "Additional instructions:\n"
            "- If your reasoning shows you lack some knowledge, explain what you need to find.\n"
            "- Issue a search query using:\n"
            "<search>\n"
            "your query here\n"
            "</search>\n"
            "- The environment returns passages from the search engine.\n"
            "- Repeat reasoning and search as needed.\n"
            "- When ready, give your final answer in plain text."
        )

    def execute_action(self, action: str) -> Tuple[bool, bool, str, str]:
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
            return False, True, "", ""

        search_result = self._search(parsed_query)
        has_error = "[SearchTool Error:" in search_result
        return True, has_error, search_result, parsed_action
