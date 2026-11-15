"""Endpoints for requests, batches, export and evaluations."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from llm_gateway.catalog import ModelCatalog
from llm_gateway.conversation import build_thread, context_usage, full_thread
from llm_gateway.runner import Runner
from llm_gateway.schemas import (
    DIMENSIONS,
    RUBRIC_VERSION,
    BatchCreate,
    BatchDetail,
    BatchExport,
    BatchSummary,
    EvaluationImport,
    ExportItem,
    FollowUpCreate,
    PromptInput,
    RequestCreate,
    RequestRecord,
    Thread,
)
from llm_gateway.store import Store

router = APIRouter(prefix="/api")

EXPORT_INSTRUCTIONS = (
    "Score each item on its own, against its prompt (and source material if present). "
    "Give every dimension a score from 0 to 10. Set groundedness to null when there is no "
    "source material. Return the results to POST /api/evaluations as "
    '{"evaluations": [{"execution_id", "judge", "scores": {...}, "summary", "strengths", '
    '"weaknesses", "recommendations", "critical_issues"}]}. Do not supply an overall score; '
    "the gateway computes it."
)


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_runner(request: Request) -> Runner:
    return request.app.state.runner


def get_catalog(request: Request) -> ModelCatalog:
    return request.app.state.catalog


StoreDep = Annotated[Store, Depends(get_store)]
RunnerDep = Annotated[Runner, Depends(get_runner)]
CatalogDep = Annotated[ModelCatalog, Depends(get_catalog)]


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


@router.post("/requests", status_code=201)
async def create_request(body: RequestCreate, store: StoreDep, runner: RunnerDep) -> RequestRecord:
    """Save a prompt and run it on the chosen model(s) in the background.

    Returns at once with executions in status "queued"; poll
    GET /api/requests/{id} for progress. An unavailable model is recorded as a
    failed execution, not rejected, so the attempt stays in history.
    """
    record = store.create_request(
        body, _unique(body.models), think=body.think, max_tokens=body.max_tokens
    )
    await runner.schedule([record])
    return record


@router.get("/requests")
def list_requests(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_batches: bool = False,
) -> list[RequestRecord]:
    """History, newest first. Batch requests are listed under their batch by default."""
    return store.list_requests(limit=limit, offset=offset, include_batches=include_batches)


@router.get("/requests/{request_id}")
def get_request(request_id: str, store: StoreDep) -> RequestRecord:
    record = store.get_request(request_id)
    if record is None:
        raise HTTPException(404, "Request not found")
    return record


@router.get("/executions/{execution_id}/thread")
def get_thread(execution_id: str, store: StoreDep) -> Thread:
    """The conversation this answer belongs to, with an estimate of context use."""
    turns = full_thread(store, execution_id)
    if not turns:
        raise HTTPException(404, "Execution not found")
    return build_thread(turns)


@router.post("/executions/{execution_id}/follow-ups", status_code=201)
async def create_follow_up(
    execution_id: str, body: FollowUpCreate, store: StoreDep, runner: RunnerDep
) -> RequestRecord:
    """Continue the conversation from a completed answer, on the same model.

    The model receives every earlier prompt and answer in the conversation.
    Refused once the context window is nearly full, because Ollama would
    silently drop the oldest text.
    """
    turns = store.ancestors(execution_id)
    if not turns:
        raise HTTPException(404, "Execution not found")
    root_request, _ = turns[0]
    _, parent = turns[-1]
    if parent.status != "completed":
        raise HTTPException(
            409,
            {"error": "parent_not_completed", "detail": "Only a completed answer can be continued"},
        )
    usage = context_usage(turns)
    if usage.state == "full":
        raise HTTPException(
            409,
            {
                "error": "context_full",
                "detail": f"The conversation uses about {usage.used_tokens} of the model's "
                f"{usage.limit_tokens} context tokens. Start a new request to continue.",
            },
        )
    record = store.create_request(
        PromptInput(prompt=body.prompt, system=root_request.system),
        [parent.model],
        routing_mode=root_request.routing_mode,
        think=root_request.think if body.think is None else body.think,
        max_tokens=root_request.max_tokens,
        parent_execution_id=parent.id,
    )
    await runner.schedule([record])
    return record


@router.post("/batches", status_code=201)
async def create_batch(
    body: BatchCreate, store: StoreDep, runner: RunnerDep, catalog: CatalogDep
) -> BatchSummary:
    """Run many prompts against many models and keep every output.

    With `models` omitted, every installed model is used (this needs Ollama
    to be reachable).
    """
    prompts = [PromptInput(prompt=p) if isinstance(p, str) else p for p in body.prompts]
    models = _unique(body.models) if body.models else [m.name for m in await catalog.list()]
    if not models:
        raise HTTPException(409, "No models are installed")
    name = (body.name or "").strip() or f"Batch {datetime.now():%Y-%m-%d %H:%M}"
    batch_id = store.create_batch(
        name, prompts, models, think=body.think, max_tokens=body.max_tokens
    )
    await runner.schedule(store.batch_requests(batch_id))
    return store.get_batch(batch_id)


@router.get("/batches")
def list_batches(store: StoreDep) -> list[BatchSummary]:
    return store.list_batches()


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, store: StoreDep) -> BatchDetail:
    summary = store.get_batch(batch_id)
    if summary is None:
        raise HTTPException(404, "Batch not found")
    return BatchDetail(**summary.model_dump(), requests=store.batch_requests(batch_id))


@router.get("/batches/{batch_id}/export")
def export_batch(
    batch_id: str,
    store: StoreDep,
    response: Response,
    blind: bool = True,
    download: bool = False,
) -> BatchExport:
    """Completed outputs of a batch, shaped for an external judge.

    Blind (the default) leaves out model names and orders each prompt's
    outputs by their random id, so the judge cannot tell which model wrote
    what. Thinking text is never included.
    """
    summary = store.get_batch(batch_id)
    if summary is None:
        raise HTTPException(404, "Batch not found")
    items = []
    for request in store.batch_requests(batch_id):
        completed = [e for e in request.executions if e.status == "completed"]
        if blind:
            completed.sort(key=lambda e: e.id)
        for execution in completed:
            items.append(
                ExportItem(
                    execution_id=execution.id,
                    prompt=request.prompt,
                    system=request.system,
                    source_material=request.source_material,
                    response=execution.response or "",
                    model=None if blind else execution.model,
                )
            )
    if download:
        response.headers["Content-Disposition"] = (
            f'attachment; filename="batch-{batch_id[:8]}-{"blind" if blind else "named"}.json"'
        )
    return BatchExport(
        batch_id=batch_id,
        batch_name=summary.name,
        blind=blind,
        rubric_version=RUBRIC_VERSION,
        dimensions=list(DIMENSIONS),
        scale="0-10",
        instructions=EXPORT_INSTRUCTIONS,
        items=items,
    )


@router.post("/evaluations", status_code=201)
def import_evaluations(body: EvaluationImport, store: StoreDep) -> dict:
    """Store evaluation results produced by any judge (a model, Claude Code, a person).

    The overall score is computed here from the dimension scores. Existing
    evaluations are never overwritten; each import adds new records.
    """
    missing = store.missing_executions([e.execution_id for e in body.evaluations])
    if missing:
        raise HTTPException(404, {"error": "unknown_executions", "execution_ids": missing})
    return {"stored": len(store.add_evaluations(body.evaluations))}
