"""Endpoints for requests, batches, export and evaluations."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from llm_gateway.catalog import ModelCatalog
from llm_gateway.conversation import build_thread, context_usage, full_thread
from llm_gateway.evaluation import (
    build_export,
    build_items,
    render_judge_package,
    summarise_batch,
)
from llm_gateway.overview import Overview, build_overview
from llm_gateway.routing.router import Router
from llm_gateway.routing.types import RoutingDraft
from llm_gateway.rubric import Rubric
from llm_gateway.runner import Runner
from llm_gateway.schemas import (
    BatchCreate,
    BatchDetail,
    BatchExport,
    BatchSummary,
    EvaluationImport,
    FollowUpCreate,
    JudgePackage,
    JudgePackageRequest,
    ModelScores,
    PromptInput,
    RequestCreate,
    RequestRecord,
    RoutePreview,
    Thread,
)
from llm_gateway.store import Store
from llm_gateway.taxonomy import DOMAINS, TASK_TYPES, Label

router = APIRouter(prefix="/api")


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_runner(request: Request) -> Runner:
    return request.app.state.runner


def get_catalog(request: Request) -> ModelCatalog:
    return request.app.state.catalog


def get_router(request: Request) -> Router:
    return request.app.state.router


StoreDep = Annotated[Store, Depends(get_store)]
RouterDep = Annotated[Router, Depends(get_router)]
RunnerDep = Annotated[Runner, Depends(get_runner)]
CatalogDep = Annotated[ModelCatalog, Depends(get_catalog)]


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


@router.post("/requests", status_code=201)
async def create_request(body: RequestCreate, store: StoreDep, runner: RunnerDep) -> RequestRecord:
    """Save a prompt and run it in the background.

    With `models` given, it runs on those models (manual). With `models`
    omitted, the router picks one (automatic): the record first has no
    executions and `routing: null`, then gains the decision and an execution.
    Poll GET /api/requests/{id} for progress. An unavailable model is recorded
    as a failed execution, not rejected, so the attempt stays in history.
    """
    automatic = body.models is None
    record = store.create_request(
        body,
        [] if automatic else _unique(body.models),
        routing_mode="automatic" if automatic else "manual",
        think=body.think,
        max_tokens=body.max_tokens,
    )
    await runner.schedule([record])
    return record


@router.post("/route")
async def preview_route(body: RoutePreview, routing: RouterDep) -> RoutingDraft:
    """Show which model the router would pick, without running or storing anything."""
    return await routing.route(body.prompt, body.source_material, body.system, use_llm=body.use_llm)


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
    # Also record which model the router would have picked for each prompt, so
    # its choices can later be compared with the evaluation scores.
    await runner.schedule(store.batch_requests(batch_id), shadow_route=True)
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
    request: Request,
    store: StoreDep,
    response: Response,
    blind: bool = True,
    download: bool = False,
    exclude_judge: str | None = None,
) -> BatchExport:
    """Completed outputs of a batch for an external judge, with the rubric and
    judge instructions. Blind by default: no model names. Thinking text is
    never included. `exclude_judge` leaves out outputs that judge already scored.
    """
    export = build_export(
        store,
        batch_id,
        rubric=request.app.state.rubric,
        instructions=request.app.state.judge_instructions,
        blind=blind,
        exclude_judge=exclude_judge,
    )
    if export is None:
        raise HTTPException(404, "Batch not found")
    if download:
        response.headers["Content-Disposition"] = (
            f'attachment; filename="batch-{batch_id[:8]}-{"blind" if blind else "named"}.json"'
        )
    return export


@router.post("/judge-package")
def judge_package(body: JudgePackageRequest, request: Request, store: StoreDep) -> JudgePackage:
    """The judge instructions, rubric and chosen outputs as one text, ready to
    paste into Claude Code or any chat. The reply goes to POST /api/evaluations.

    Takes any mix of single requests (from history) and whole batches.
    """
    requests = []
    missing = []
    for request_id in dict.fromkeys(body.request_ids):
        record = store.get_request(request_id)
        if record is None:
            missing.append(request_id)
        else:
            requests.append(record)
    batch_names = []
    for batch_id in dict.fromkeys(body.batch_ids):
        summary = store.get_batch(batch_id)
        if summary is None:
            missing.append(batch_id)
        else:
            batch_names.append(summary.name)
            requests.extend(store.batch_requests(batch_id))
    if missing:
        raise HTTPException(404, {"error": "not_found", "ids": missing})
    if not requests:
        raise HTTPException(422, "Choose at least one request or batch")

    singles = len(requests) - sum(1 for r in requests if r.batch_id)
    parts = [f'batch "{name}"' for name in batch_names]
    if singles:
        parts.append(f"{singles} history run{'s' if singles != 1 else ''}")
    rubric = request.app.state.rubric
    items = build_items(store, requests, blind=body.blind, exclude_judge=body.exclude_judge)
    return JudgePackage(
        text=render_judge_package(
            request.app.state.judge_instructions, rubric, items, " and ".join(parts)
        ),
        item_count=len(items),
        rubric_version=rubric.version,
        blind=body.blind,
    )


@router.get("/batches/{batch_id}/scores")
def batch_scores(batch_id: str, store: StoreDep) -> list[ModelScores]:
    """Mean evaluation scores per model and judge for a batch."""
    if store.get_batch(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    return summarise_batch(store, batch_id)


@router.get("/overview")
def overview(
    request: Request,
    store: StoreDep,
    days: Annotated[int | None, Query(ge=1, le=3650)] = None,
) -> Overview:
    """Routing, performance and evaluation totals. `days` omitted = all time."""
    since = datetime.now(UTC) - timedelta(days=days) if days else None
    config = request.app.state.settings
    pricing = None
    if config.cloud_input_per_mtok or config.cloud_output_per_mtok:
        pricing = (
            config.cloud_compare_model,
            config.cloud_input_per_mtok,
            config.cloud_output_per_mtok,
        )
    return build_overview(store.requests_since(since), days=days, since=since, pricing=pricing)


@router.get("/rubric")
def get_rubric(request: Request) -> Rubric:
    return request.app.state.rubric


@router.get("/task-types")
def task_types() -> dict[str, list[Label]]:
    """The fixed task types and domains a judge chooses from."""
    return {"task_types": TASK_TYPES, "domains": DOMAINS}


@router.get("/judge-prompt")
def get_judge_prompt(request: Request) -> dict[str, str]:
    """The instructions every judge is given (config/judge_prompt.md)."""
    return {"instructions": request.app.state.judge_instructions}


@router.post("/evaluations", status_code=201)
def import_evaluations(body: EvaluationImport, request: Request, store: StoreDep) -> dict:
    """Store evaluation results produced by any judge (a model, Claude Code, a person).

    The overall score is computed here from the dimension scores and the
    rubric weights. Existing evaluations are never overwritten; each import
    adds new records.
    """
    missing = store.missing_executions([e.execution_id for e in body.evaluations])
    if missing:
        raise HTTPException(404, {"error": "unknown_executions", "execution_ids": missing})
    rubric = request.app.state.rubric
    stored = store.add_evaluations(
        body.evaluations, weights=rubric.weights, rubric_version=rubric.version
    )
    return {"stored": len(stored)}
