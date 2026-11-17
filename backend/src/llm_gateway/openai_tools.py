"""Function calling for the OpenAI-compatible endpoint.

A request with `tools`, or a conversation that already contains tool calls
and tool results, takes this path instead of the plain text one. The client
runs the tools itself; the gateway only passes the function definitions to
the model and returns the calls the model wants made, in OpenAI's format.

- The model must support tool calling. With model "auto", the router decides
  as usual and, if its choice cannot call tools, the next suitable model that
  can is used instead (the routing explanation says so).
- Some models write a call as JSON in their answer instead of using the
  tool-call field. When every such call names a function the client offered,
  it is returned as a real tool call and flagged in the `gateway` field.
- Streaming is supported, but the turn is generated in full first: tool-call
  arguments are only usable once complete, and Ollama returns calls whole.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field

from llm_gateway.providers.base import ChatMessage, ModelInfo, ProviderError, ToolCall, ToolTurn
from llm_gateway.routing.types import RoutingDraft
from llm_gateway.runner import THINKING_EXHAUSTED
from llm_gateway.schemas import PromptInput
from llm_gateway.tooluse.checker import extract_text_calls


class ToolRequestError(Exception):
    """A request the gateway cannot serve. `status` and `code` go to the client."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class ToolOutcome:
    request_id: str
    model: str
    turn: ToolTurn
    calls: list[ToolCall]
    content: str
    # The calls came as JSON text in the answer and were converted.
    recovered: bool
    calls_ids: list[str] = field(default_factory=list)


def uses_tools(body) -> bool:
    return bool(body.tools) or any(
        m.role == "tool" or (m.role == "assistant" and m.tool_calls) for m in body.messages
    )


def _arguments(raw) -> dict:
    """OpenAI sends arguments as a JSON string; Ollama wants an object."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def to_provider_messages(messages) -> list[dict]:
    """OpenAI chat messages in the shape Ollama's /api/chat expects."""
    out: list[dict] = []
    names_by_id: dict[str, str] = {}
    for m in messages:
        if m.role in ("system", "developer"):
            out.append({"role": "system", "content": m.text()})
        elif m.role == "user":
            out.append({"role": "user", "content": m.text()})
        elif m.role == "assistant":
            entry: dict = {"role": "assistant", "content": m.text()}
            if m.tool_calls:
                entry["tool_calls"] = []
                for call in m.tool_calls:
                    function = call.get("function") or {}
                    names_by_id[call.get("id", "")] = function.get("name", "")
                    entry["tool_calls"].append(
                        {
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": _arguments(function.get("arguments")),
                            }
                        }
                    )
            out.append(entry)
        else:  # tool result
            entry = {"role": "tool", "content": m.text()}
            name = m.name or names_by_id.get(m.tool_call_id or "")
            if name:
                entry["tool_name"] = name
            out.append(entry)
    return out


def summarise_for_history(messages) -> tuple[str, str | None, list, str | None]:
    """What the gateway stores: (prompt, system, text-only history, source material).

    The prompt is the last user message. Tool calls and results after it are
    kept as source material, so History and a judge see what the model saw.
    """
    system = "\n\n".join(m.text() for m in messages if m.role in ("system", "developer")) or None
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=None)
    if last_user is None or not messages[last_user].text().strip():
        raise ToolRequestError(
            400, "The conversation needs a user message with text.", "unsupported_messages"
        )
    history = [
        ChatMessage(role=m.role, content=m.text())
        for m in messages[:last_user]
        if m.role in ("user", "assistant") and m.text().strip()
    ]
    trail = []
    for m in messages[last_user + 1 :]:
        if m.role == "assistant":
            for call in m.tool_calls or []:
                function = call.get("function") or {}
                trail.append(f"Called {function.get('name')}({function.get('arguments', '')})")
            if m.text().strip():
                trail.append(f"Assistant: {m.text()}")
        elif m.role == "tool":
            trail.append(f"Result{f' of {m.name}' if m.name else ''}: {m.text()}")
    source = "Tool calls and results so far:\n" + "\n".join(trail) if trail else None
    return messages[last_user].text(), system, history, source


async def choose_model(state, body, prompt: str, source: str | None, system: str | None):
    """(model info, routing draft or None). Raises ToolRequestError."""
    installed = {m.name: m for m in await state.catalog.list()}
    can_call = {name for name, m in installed.items() if "tools" in m.capabilities}
    if body.model not in ("auto", "gateway-auto"):
        info = installed.get(body.model)
        if info is None:
            raise ToolRequestError(
                404,
                f"Model '{body.model}' is not available. Use 'auto' or a name from /v1/models.",
                "model_not_found",
            )
        if body.model not in can_call:
            raise ToolRequestError(
                400,
                f"{body.model} does not support tool calling. Use 'auto' or one of: "
                + ", ".join(sorted(can_call)),
                "tools_not_supported",
            )
        return info, None
    draft: RoutingDraft = await state.router.route(prompt, source, system)
    order = [
        draft.selected_model,
        draft.fallback_model,
        *draft.candidates,
        *state.router.registry.last_resort,
        *sorted(can_call),
    ]
    chosen = next((name for name in order if name in can_call), None)
    if chosen is None:
        raise ToolRequestError(
            503, "No installed model supports tool calling.", "no_model_available"
        )
    if chosen != draft.selected_model:
        reason = (
            f" The request uses tools and {draft.selected_model} cannot call them,"
            f" so {chosen} was used."
            if draft.selected_model
            else f" {chosen} was used because the request uses tools."
        )
        draft = draft.model_copy(
            update={"selected_model": chosen, "explanation": draft.explanation + reason}
        )
    return installed[chosen], draft


def _readable(content: str, calls: list[ToolCall]) -> str:
    """The stored answer: the text, then each call, so History shows what happened."""
    lines = [content.strip()] if content.strip() else []
    lines += [f"[tool call] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in calls]
    return "\n".join(lines)


async def run_tool_turn(state, body, think: bool) -> ToolOutcome:
    """Store the request, run one turn and record the outcome. Raises ToolRequestError."""
    prompt, system, history, source = summarise_for_history(body.messages)
    info: ModelInfo
    try:
        info, draft = await choose_model(state, body, prompt, source, system)
    except ProviderError as exc:
        raise ToolRequestError(503, str(exc), exc.code) from exc
    store, queue = state.store, state.queue
    record = store.create_request(
        PromptInput(prompt=prompt, system=system, source_material=source),
        [] if draft else [info.name],
        routing_mode="automatic" if draft else "manual",
        think=think,
        max_tokens=body.max_completion_tokens or body.max_tokens,
        history=history,
        source="api",
        temperature=body.temperature,
    )
    if draft:
        store.add_decision(record.id, draft)
        execution = store.add_execution(record.id, info.name)
    else:
        execution = record.executions[0]

    tools = [] if body.tool_choice == "none" else list(body.tools or [])
    store.mark_running(execution.id)
    try:
        turn, queue_wait_ms, execution_ms = await queue.run_tool_turn(
            to_provider_messages(body.messages),
            tools,
            info,
            think=think,
            max_tokens=body.max_completion_tokens or body.max_tokens,
            temperature=body.temperature,
        )
    except asyncio.CancelledError:
        store.fail_execution(execution.id, "interrupted", "The client disconnected")
        raise
    except ProviderError as exc:
        store.fail_execution(execution.id, exc.code, str(exc))
        raise ToolRequestError(
            503 if exc.code in ("provider_unavailable", "low_memory") else 502, str(exc), exc.code
        ) from exc

    calls, content, recovered = turn.tool_calls, turn.content, False
    offered = {t.get("function", {}).get("name") for t in tools}
    if not calls and tools:
        found = extract_text_calls(turn.content)
        if found and all(c.name in offered and not c.malformed for c in found):
            calls, content, recovered = found, "", True

    metrics = turn.model_dump(exclude={"model", "content", "thinking", "tool_calls"})
    metrics["tool_calls"] = [c.model_dump(exclude={"malformed"}) for c in calls]
    metrics["tool_calls_recovered_from_text"] = recovered
    if not calls and not content.strip() and turn.thinking and turn.done_reason == "length":
        store.fail_execution(
            execution.id,
            THINKING_EXHAUSTED,
            "Ran out of room while thinking: no answer and no tool call.",
            thinking=turn.thinking,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            metrics=metrics,
        )
        raise ToolRequestError(502, "The model spent all its tokens thinking.", THINKING_EXHAUSTED)
    store.complete_execution(
        execution.id,
        response=_readable(content, calls),
        thinking=turn.thinking,
        queue_wait_ms=queue_wait_ms,
        execution_ms=execution_ms,
        metrics=metrics,
        warnings=["output_cut_off"] if turn.done_reason == "length" else [],
    )
    return ToolOutcome(
        request_id=record.id,
        model=info.name,
        turn=turn,
        calls=calls,
        content=content,
        recovered=recovered,
        calls_ids=[f"call_{uuid.uuid4().hex[:24]}" for _ in calls],
    )


def openai_tool_calls(outcome: ToolOutcome, *, with_index: bool = False) -> list[dict]:
    out = []
    for i, (call, call_id) in enumerate(zip(outcome.calls, outcome.calls_ids, strict=True)):
        entry: dict = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        if with_index:
            entry = {"index": i, **entry}
        out.append(entry)
    return out


def finish_reason(outcome: ToolOutcome) -> str:
    if outcome.calls:
        return "tool_calls"
    return "length" if outcome.turn.done_reason == "length" else "stop"


def usage(outcome: ToolOutcome) -> dict:
    prompt = outcome.turn.prompt_tokens or 0
    output = outcome.turn.output_tokens or 0
    return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}
