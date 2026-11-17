"""Scores a model's tool calls against the accepted answers, with no judge.

The rules follow BFCL's "AST" check: the function name must match, every
required argument must be present with an accepted value, no argument the
function does not define may appear, and a category with no fitting function
passes only if the model calls nothing. Numbers compare by value (5 equals
5.0) but a number sent as text, or the reverse, is a type error, as in BFCL.
"""

import itertools
import json
import re
from dataclasses import dataclass

from llm_gateway.providers.base import ToolCall

# BFCL ignores case and these characters when comparing text.
_IGNORED = re.compile(r"[ ,./\-_*^]")


@dataclass
class Verdict:
    passed: bool
    reason: str


def _text(value: str) -> str:
    return _IGNORED.sub("", value).lower().replace("'", '"')


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _equal(actual, accepted) -> bool:
    """One accepted value against what the model sent."""
    if isinstance(accepted, dict):
        if not isinstance(actual, dict) or not set(actual) <= set(accepted):
            return False
        for key, options in accepted.items():
            if key not in actual:
                if not (isinstance(options, list) and "" in options):
                    return False
            elif not _value_ok(actual[key], options if isinstance(options, list) else [options]):
                return False
        return True
    if isinstance(accepted, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(accepted)
            and all(_equal(a, b) for a, b in zip(actual, accepted, strict=True))
        )
    if isinstance(accepted, bool) or isinstance(actual, bool):
        return actual is accepted
    if _is_number(accepted):
        return _is_number(actual) and float(actual) == float(accepted)
    if isinstance(accepted, str):
        return isinstance(actual, str) and _text(actual) == _text(accepted)
    return actual == accepted


def _value_ok(actual, alternatives: list) -> bool:
    return any(_equal(actual, option) for option in alternatives if option != "")


def _call_problem(call: ToolCall, expected: dict, known: dict[str, set[str]]) -> str | None:
    """Why this call is not the expected one, or None when it matches."""
    if call.malformed:
        return f"{call.name}: arguments were not a JSON object"
    if call.name != expected["name"]:
        return f"called {call.name}, expected {expected['name']}"
    allowed = known.get(call.name, set())
    for key in call.arguments:
        if allowed and key not in allowed:
            return f"{call.name}: unknown argument '{key}'"
    for key, options in expected["arguments"].items():
        if key not in call.arguments:
            if "" not in options:
                return f"{call.name}: missing argument '{key}'"
        elif not _value_ok(call.arguments[key], options):
            shown = [o for o in options if o != ""]
            return f"{call.name}: '{key}' was {call.arguments[key]!r}, expected {shown!r}"
    return None


def score(expected: list[dict], calls: list[ToolCall], tools: list[dict]) -> Verdict:
    """`expected` is the case's accepted calls; empty means no call is right."""
    if not expected:
        if calls:
            return Verdict(False, f"called {calls[0].name} although no function fits")
        return Verdict(True, "correctly made no call")
    if not calls:
        return Verdict(False, "made no tool call")
    if len(calls) != len(expected):
        return Verdict(False, f"made {len(calls)} calls, expected {len(expected)}")
    known = {
        t["function"]["name"]: set((t["function"].get("parameters") or {}).get("properties", {}))
        for t in tools
    }
    # Calls may come in any order: try each way of pairing them up.
    first_problem = None
    for order in itertools.permutations(range(len(calls))):
        problems = [
            _call_problem(calls[i], exp, known) for i, exp in zip(order, expected, strict=True)
        ]
        if not any(problems):
            return Verdict(True, "all calls match")
        if first_problem is None:
            first_problem = next(p for p in problems if p)
    return Verdict(False, first_problem or "calls do not match")


_TAGGED = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_values(text: str):
    """Every JSON object or list found in free text."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if text[index] in "{[":
            try:
                value, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            yield value
            index = end
        else:
            index += 1


def extract_text_calls(content: str) -> list[ToolCall]:
    """Calls a model wrote into its answer as JSON instead of using the
    provider's tool-call field, e.g. {"name": "f", "arguments": {...}}."""
    calls = []
    sources = _TAGGED.findall(content) + _FENCE.findall(content) or [content]
    for source in sources:
        for value in _json_values(source):
            for item in value if isinstance(value, list) else [value]:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                arguments = item.get("arguments", item.get("parameters", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                calls.append(
                    ToolCall(
                        name=item["name"],
                        arguments=arguments if isinstance(arguments, dict) else {},
                        malformed=not isinstance(arguments, dict),
                    )
                )
        if calls:
            break
    return calls
