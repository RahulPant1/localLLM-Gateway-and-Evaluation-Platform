"""Loads agent test suites from JSON files in config/agent_suites."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You are an assistant that completes tasks for the user by calling the tools you are given. "
    "Today is {today}. Look things up with the tools instead of guessing. "
    "When the task is done, reply with a short final answer for the user."
)


class Scenario(BaseModel):
    id: str
    category: str
    title: str
    prompt: str
    # Which simulated apps the model gets tools for: files, mail, api.
    apps: list[str]
    # Changes to the shared world for this scenario only.
    world: dict = Field(default_factory=dict)
    checks: list[dict]
    max_steps: int = 10


class AgentSuite(BaseModel):
    id: str
    title: str
    description: str = ""
    today: str = "2026-11-17"
    system: str = DEFAULT_SYSTEM
    categories: dict[str, str] = Field(default_factory=dict)
    # The files, mailbox and API data every scenario starts from.
    world: dict
    scenarios: list[Scenario]

    @property
    def system_prompt(self) -> str:
        return self.system.format(today=self.today)


def load_agent_suites(directory: Path) -> dict[str, AgentSuite]:
    suites: dict[str, AgentSuite] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            suite = AgentSuite.model_validate(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping agent suite %s: %s", path.name, exc)
            continue
        suites[suite.id] = suite
    return suites
