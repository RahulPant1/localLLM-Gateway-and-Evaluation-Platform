"""Connectors give an agent tools backed by real apps.

Each connector offers OpenAI-format tool definitions and runs calls to them.
All of them are read-only: they can look things up but cannot send, write,
delete or change anything. What a tool returns is data from outside the
gateway - an email, a file, a web page - and may contain text written to
mislead the model, so the agent's instructions say to treat it as data.
"""

import json
from dataclasses import dataclass

from llm_gateway.agents.world import ToolOutcome

MAX_RESULT_CHARS = 6000


@dataclass
class ConnectorStatus:
    name: str
    label: str
    available: bool
    # What it is connected to, or how to set it up.
    detail: str


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def error(message: str) -> ToolOutcome:
    return ToolOutcome(dump({"error": message}), error=True)


def invalid(message: str) -> ToolOutcome:
    return ToolOutcome(dump({"error": message}), invalid=True)


def capped(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
