"""Loads tool-calling test suites from JSON files in config/tool_suites."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCase(BaseModel):
    id: str
    category: str
    prompt: str
    tools: list[dict]
    # Accepted calls: [{"name": ..., "arguments": {param: [accepted values]}}].
    # Empty means the model should not call anything.
    expected: list[dict] = Field(default_factory=list)


class ToolSuite(BaseModel):
    id: str
    title: str
    source: str = ""
    categories: dict[str, str] = Field(default_factory=dict)
    cases: list[ToolCase]


def load_suites(directory: Path) -> dict[str, ToolSuite]:
    suites: dict[str, ToolSuite] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            suite = ToolSuite.model_validate(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping tool suite %s: %s", path.name, exc)
            continue
        suites[suite.id] = suite
    return suites
