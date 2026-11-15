"""OpenAI-compatible endpoint, so other tools can use the gateway.

Editors, chat front ends and scripts that speak the OpenAI chat format point
at http://localhost:8000/v1. The model name "auto" lets the router choose.
Every call is stored like a request made in the app, so it appears in
History and can be evaluated.

Supported: text messages, streaming, max_tokens, temperature,
reasoning_effort. Ignored: tools, images, stop, top_p, response_format, n.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.providers.base import ChatMessage, ModelNotFound, ProviderError
from llm_gateway.schemas import Execution, PromptInput, RequestRecord

router = APIRouter(prefix="/v1")

# Model names that mean "let the router choose".
AUTO_MODELS = {"auto", "gateway-auto"}
KEEP_ALIVE_SECONDS = 10.0
# HTTP status for a failed execution's error code.
STATUS_FOR_CODE = {
    "model_not_found": 404,
    "provider_unavailable": 503,
    "generation_timeout": 504,
    "generation_failed": 502,
}


class Part(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    text: str | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[Part] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if any(part.type != "text" for part in self.content or []):
            raise ValueError("Only text content is supported; images and files are not.")
        return "\n".join(part.text or "" for part in self.content or [])


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "auto"
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: str | None = None


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error"):
    body = {"error": {"message": message, "type": kind, "code": code, "param": None}}
    return JSONResponse(status_code=status, content=body)


def split_messages(messages: list[Message]) -> tuple[str | None, list[ChatMessage], str]:
    """Turn an OpenAI message list into (system prompt, earlier turns, new prompt)."""
    system = [m.text() for m in messages if m.role in ("system", "developer")]
    turns = [m for m in messages if m.role in ("user", "assistant")]
    if any(m.role == "tool" for m in messages):
        raise ValueError("Tool messages are not supported.")
    if not turns or turns[-1].role != "user" or not turns[-1].text().strip():
        raise ValueError("The last message must be a user message with text.")
    history = [ChatMessage(role=m.role, content=m.text()) for m in turns[:-1]]
    return "\n\n".join(s for s in system if s.strip()) or None, history, turns[-1].text()


def _usage(execution: Execution) -> dict:
    metrics = execution.metrics or {}
    prompt = int(metrics.get("prompt_tokens") or 0)
    output = int(metrics.get("output_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}


def _finish_reason(execution: Execution) -> str:
    return "length" if (execution.metrics or {}).get("done_reason") == "length" else "stop"


def _gateway_info(record: RequestRecord) -> dict:
    """Extra, non-OpenAI field: where to find this call in the gateway."""
    info: dict = {"request_id": record.id}
    if record.routing:
        info["routing"] = record.routing.model_dump(
            include={"category", "complexity", "confidence", "decided_by", "fallback_used"}
        )
    return info


def _outcome(record: RequestRecord | None) -> tuple[Execution | None, tuple[int, str, str] | None]:
    """The execution that answered, or (status, message, code) for a failure."""
    if record is None:
        return None, (500, "The request was lost.", "internal_error")
    if not record.executions:
        reason = record.routing.explanation if record.routing else "Routing did not complete."
        return None, (503, f"No model could be chosen. {reason}", "no_model_available")
    # The last execution is the fallback model when the first one failed.
    execution = record.executions[-1]
    if execution.status != "completed":
        code = execution.error_code or "generation_failed"
        message = execution.error_detail or "The model did not produce an answer."
        return None, (STATUS_FOR_CODE.get(code, 500), message, code)
    return execution, None


@router.get("/models")
async def list_models(request: Request) -> dict:
    """ "auto" followed by every installed model, in the OpenAI list format."""
    try:
        installed = await request.app.state.catalog.list()
    except ProviderError as exc:
        return _error(503, str(exc), exc.code, "server_error")
    names = ["auto", *(m.name for m in installed)]
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "local-llm-gateway"}
            for name in names
        ],
    }


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    state = request.app.state
    store, runner = state.store, state.runner
    try:
        system, history, prompt = split_messages(body.messages)
    except ValueError as exc:
        return _error(400, str(exc), "unsupported_messages")

    automatic = body.model in AUTO_MODELS
    if not automatic:
        try:
            await state.catalog.get(body.model)
        except ModelNotFound as exc:
            return _error(404, f"{exc}. Use 'auto' or a name from /v1/models.", exc.code)
        except ProviderError as exc:
            return _error(STATUS_FOR_CODE.get(exc.code, 500), str(exc), exc.code, "server_error")

    think = state.settings.api_think
    if body.reasoning_effort:
        think = body.reasoning_effort.lower() in ("medium", "high", "xhigh", "max")

    record = store.create_request(
        PromptInput(prompt=prompt, system=system),
        [] if automatic else [body.model],
        routing_mode="automatic" if automatic else "manual",
        think=think,
        max_tokens=body.max_completion_tokens or body.max_tokens,
        history=history,
        source="api",
        temperature=body.temperature,
    )
    events = runner.subscribe(record.id)
    task = await runner.schedule([record])
    completion_id = f"chatcmpl-{record.id}"
    created = int(time.time())

    if not body.stream:
        try:
            while (await events.get())[0] != "end":
                pass
        finally:
            runner.unsubscribe(record.id)
        final = store.get_request(record.id)
        execution, failure = _outcome(final)
        if failure:
            status, message, code = failure
            return _error(status, message, code, "server_error")
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": execution.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": execution.response or ""},
                    "finish_reason": _finish_reason(execution),
                }
            ],
            "usage": _usage(execution),
            "gateway": _gateway_info(final),
        }

    include_usage = bool(body.stream_options and body.stream_options.include_usage)

    async def stream() -> AsyncIterator[str]:
        model = body.model
        finished = False

        def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                **extra,
            }
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            yield chunk({"role": "assistant", "content": ""})
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), KEEP_ALIVE_SECONDS)
                except TimeoutError:
                    # Waiting in the queue or for a model to load: keep the connection open.
                    yield ": keep-alive\n\n"
                    continue
                if event[0] == "start":
                    model = event[1]
                elif event[0] == "delta":
                    key = "content" if event[1] == "content" else "reasoning_content"
                    yield chunk({key: event[2]})
                else:
                    break
            final = store.get_request(record.id)
            execution, failure = _outcome(final)
            finished = True
            if failure:
                _, message, code = failure
                error = {"message": message, "type": "server_error", "code": code}
                yield f"data: {json.dumps({'error': error})}\n\n"
            else:
                yield chunk({}, _finish_reason(execution), gateway=_gateway_info(final))
                if include_usage:
                    usage = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": _usage(execution),
                    }
                    yield f"data: {json.dumps(usage)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            runner.unsubscribe(record.id)
            if not finished:
                # The client went away: stop generating and free the GPU.
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
