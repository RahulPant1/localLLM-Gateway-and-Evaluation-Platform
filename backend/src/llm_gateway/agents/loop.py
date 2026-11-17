"""The agent loop shared by the simulated tests and real-app runs.

The model gets the conversation and the tools; each tool call it makes is
run by `call_tool` and the result goes back to it; this repeats until the
model answers without calling anything, or a limit is reached.
"""

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from llm_gateway.agents.world import ToolOutcome
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import ModelInfo
from llm_gateway.tooluse.checker import extract_text_calls

MAX_TOKENS = 1024
MAX_TOKENS_THINKING = 4096
# The same call with the same arguments this many times in a row ends the run.
REPEAT_LIMIT = 3

ToolCaller = Callable[[str, dict], ToolOutcome | Awaitable[ToolOutcome]]


@dataclass
class LoopResult:
    answer: str = ""
    # answered | step_limit | loop | cut_off
    stop: str = "step_limit"
    trace: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    invalid_calls: int = 0
    text_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0


async def run_loop(
    queue: ExecutionQueue,
    model: ModelInfo,
    messages: list[dict],
    tools: list[dict],
    call_tool: ToolCaller,
    *,
    think: bool,
    max_steps: int = 10,
    on_step: Callable[[LoopResult], None] | None = None,
) -> LoopResult:
    """Run the loop. `messages` is extended in place. `on_step` is called after
    every step with the result so far, so a screen can show progress."""
    offered = {t["function"]["name"] for t in tools}
    result = LoopResult()
    last_call, repeats = None, 0
    started = time.perf_counter()
    for step in range(max_steps):
        turn, queue_wait_ms, execution_ms = await queue.run_tool_turn(
            messages,
            tools,
            model,
            think=think,
            max_tokens=MAX_TOKENS_THINKING if model.supports_thinking else MAX_TOKENS,
            temperature=0,
        )
        result.queue_wait_ms += queue_wait_ms
        result.prompt_tokens += turn.prompt_tokens or 0
        result.output_tokens += turn.output_tokens or 0
        calls, as_text = turn.tool_calls, False
        if not calls:
            found = extract_text_calls(turn.content)
            if found and all(c.name in offered for c in found):
                calls, as_text = found, True
        entry: dict = {
            "step": step + 1,
            "content": "" if as_text else turn.content,
            "thought": bool(turn.thinking),
            "calls": [],
            "ms": execution_ms,
            "calls_as_text": as_text,
        }
        result.trace.append(entry)
        if not calls:
            result.answer = turn.content
            result.stop = "cut_off" if turn.done_reason == "length" else "answered"
            break
        if as_text:
            result.text_calls += len(calls)
        messages.append(
            {
                "role": "assistant",
                "content": entry["content"],
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ],
            }
        )
        looping = False
        for call in calls:
            outcome = call_tool(call.name, call.arguments)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            result.tool_calls += 1
            result.invalid_calls += outcome.invalid
            entry["calls"].append(
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": outcome.text,
                    "invalid": outcome.invalid,
                    "error": outcome.error,
                }
            )
            messages.append({"role": "tool", "content": outcome.text, "tool_name": call.name})
            key = (call.name, repr(sorted(call.arguments.items())))
            repeats = repeats + 1 if key == last_call else 1
            last_call = key
            looping = looping or repeats >= REPEAT_LIMIT
        if on_step:
            on_step(result)
        if looping:
            result.stop = "loop"
            break
    result.execution_ms = round((time.perf_counter() - started) * 1000 - result.queue_wait_ms, 1)
    result.queue_wait_ms = round(result.queue_wait_ms, 1)
    return result
