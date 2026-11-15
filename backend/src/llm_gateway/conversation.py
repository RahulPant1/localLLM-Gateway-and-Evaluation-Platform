"""Follow-up chat: rebuilding a conversation and estimating context use.

A conversation is not a separate table. Each follow-up is an ordinary request
whose `parent_execution_id` names the answer it continues from, so every turn
keeps its own timings, status and evaluations.
"""

from llm_gateway.providers.base import ChatMessage
from llm_gateway.schemas import ContextUsage, Execution, RequestRecord, Thread, ThreadTurn
from llm_gateway.store import Store

CHARS_PER_TOKEN = 4
# Local models have small context windows and Ollama drops old text silently
# once they fill, so the app warns early and stops before that happens.
WARNING_FRACTION = 0.7
FULL_FRACTION = 0.9

Turn = tuple[RequestRecord, Execution]


def compose_prompt(request: RequestRecord) -> str:
    """The text sent to the model: the prompt, followed by any source material."""
    if not request.source_material:
        return request.prompt
    return f"{request.prompt}\n\nSource material:\n{request.source_material}"


def history_messages(turns: list[Turn]) -> list[ChatMessage]:
    """Earlier turns as chat messages. Thinking text is never sent back to the model."""
    messages = []
    for request, execution in turns:
        messages.append(ChatMessage(role="user", content=compose_prompt(request)))
        messages.append(ChatMessage(role="assistant", content=execution.response or ""))
    return messages


def full_thread(store: Store, execution_id: str) -> list[Turn]:
    """The whole conversation through this answer: everything before it, then
    the follow-ups after it (the newest branch, if it was continued twice)."""
    turns = store.ancestors(execution_id)
    while turns:
        following = store.newest_follow_up(turns[-1][1].id)
        if following is None:
            break
        turns.append(following)
    return turns


def context_usage(turns: list[Turn]) -> ContextUsage:
    """Estimate the tokens the next message would start with.

    The last completed turn reports exactly how many tokens its input used.
    Its answer is added on top; thinking tokens are excluded because thinking
    is not sent back.
    """
    completed = [t for t in turns if t[1].status == "completed"]
    limit = None
    used = 0.0
    if completed:
        _, last = completed[-1]
        metrics = last.metrics or {}
        limit = metrics.get("num_ctx")
        prompt_tokens = metrics.get("prompt_tokens")
        output_tokens = metrics.get("output_tokens") or 0
        answer_chars = len(last.response or "")
        thinking_chars = len(last.thinking or "")
        if prompt_tokens is not None:
            total_chars = answer_chars + thinking_chars
            answer_share = answer_chars / total_chars if total_chars else 0
            used = prompt_tokens + output_tokens * answer_share
        else:
            used = sum(len(m.content) for m in history_messages(completed)) / CHARS_PER_TOKEN
    fraction = round(used / limit, 3) if limit else None
    state = "ok"
    if fraction is not None and fraction >= FULL_FRACTION:
        state = "full"
    elif fraction is not None and fraction >= WARNING_FRACTION:
        state = "warning"
    return ContextUsage(used_tokens=round(used), limit_tokens=limit, fraction=fraction, state=state)


def build_thread(turns: list[Turn]) -> Thread:
    root_request, root_execution = turns[0]
    last_execution = turns[-1][1]
    in_progress = last_execution.status in ("queued", "running")
    completed = [e for _, e in turns if e.status == "completed"]
    return Thread(
        model=root_execution.model,
        system=root_request.system,
        turns=[
            ThreadTurn(
                request_id=request.id,
                prompt=request.prompt,
                source_material=request.source_material,
                created_at=request.created_at,
                execution=execution,
            )
            for request, execution in turns
        ],
        context=context_usage(turns),
        # After a failed turn, a new message continues from the last good answer.
        reply_to=None if in_progress or not completed else completed[-1].id,
    )
