"""Endpoints for multi-step agent runs."""

from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from llm_gateway.agents.runner import AgentRunner
from llm_gateway.agents.suites import AgentSuite
from llm_gateway.catalog import ModelCatalog
from llm_gateway.providers.base import ProviderError
from llm_gateway.store import Store

router = APIRouter(prefix="/api/agents")
KIND = "agent"


def _store(request: Request) -> Store:
    return request.app.state.store


def _runner(request: Request) -> AgentRunner:
    return request.app.state.agent_runner


def _suites(request: Request) -> dict[str, AgentSuite]:
    return request.app.state.agent_suites


StoreDep = Annotated[Store, Depends(_store)]
RunnerDep = Annotated[AgentRunner, Depends(_runner)]
SuitesDep = Annotated[dict[str, AgentSuite], Depends(_suites)]


class SuiteInfo(BaseModel):
    id: str
    title: str
    description: str
    scenario_count: int
    categories: dict[str, dict]


class RunCreate(BaseModel):
    suite: str
    models: list[str] = Field(min_length=1)
    think: bool = False
    name: str | None = None


class Tally(BaseModel):
    passed: int = 0
    failed: int = 0
    # The scenario did not run to the end (timeout, low memory, interrupted).
    errors: int = 0
    waiting: int = 0


class ModelTally(Tally):
    model: str
    # Mean share of checks passed, 0 to 1, over finished scenarios.
    check_score: float | None = None
    by_category: dict[str, Tally] = Field(default_factory=dict)
    mean_steps: float | None = None
    mean_tool_calls: float | None = None
    invalid_calls: int = 0
    # Calls the model wrote as JSON text instead of real tool calls.
    text_calls: int = 0
    mean_ms: float | None = None


class RunSummary(BaseModel):
    id: str
    suite: str
    name: str
    models: list[str]
    think: bool
    created_at: str
    total: int
    finished: int
    running: bool
    per_model: list[ModelTally]


class ResultRow(BaseModel):
    id: str
    case_id: str
    category: str
    model: str
    status: str
    passed: bool | None
    score: float | None
    title: str
    prompt: str
    apps: list[str]
    checks: list[dict]
    trace: list[dict]
    effects: dict
    answer: str | None
    stop_reason: str | None
    steps: int | None
    tool_calls: int | None
    invalid_calls: int | None
    text_calls: int | None
    execution_ms: float | None
    error_code: str | None
    error_detail: str | None


class RunDetail(RunSummary):
    results: list[ResultRow]


def _mean(values: list[float], digits: int = 1) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def summarise(run: dict, results: list[dict], running: bool) -> RunSummary:
    tallies = {m: ModelTally(model=m) for m in run["models"]}
    done: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        tally = tallies.get(r["model"])
        if tally is None:
            continue
        bucket = tally.by_category.setdefault(r["category"], Tally())
        field = (
            "waiting"
            if r["status"] in ("queued", "running")
            else "errors"
            if r["status"] == "failed"
            else "passed"
            if r["passed"]
            else "failed"
        )
        for t in (tally, bucket):
            setattr(t, field, getattr(t, field) + 1)
        if r["status"] == "completed":
            done[r["model"]].append(r)
    for model, rows in done.items():
        t = tallies[model]
        t.check_score = _mean([r["score"] or 0 for r in rows], 3)
        t.mean_steps = _mean([r["steps"] or 0 for r in rows])
        t.mean_tool_calls = _mean([r["tool_calls"] or 0 for r in rows])
        t.invalid_calls = sum(r["invalid_calls"] or 0 for r in rows)
        t.text_calls = sum(r["text_calls"] or 0 for r in rows)
        t.mean_ms = _mean([r["execution_ms"] for r in rows if r["execution_ms"] is not None], 0)
    return RunSummary(
        id=run["id"],
        suite=run["suite"],
        name=run["name"],
        models=run["models"],
        think=run["think"],
        created_at=run["created_at"],
        total=len(results),
        finished=sum(r["status"] in ("completed", "failed") for r in results),
        running=running,
        per_model=list(tallies.values()),
    )


@router.get("/suites")
def list_suites(suites: SuitesDep) -> list[SuiteInfo]:
    out = []
    for suite in suites.values():
        counts: dict[str, int] = defaultdict(int)
        for s in suite.scenarios:
            counts[s.category] += 1
        out.append(
            SuiteInfo(
                id=suite.id,
                title=suite.title,
                description=suite.description,
                scenario_count=len(suite.scenarios),
                categories={
                    name: {"description": suite.categories.get(name, ""), "scenarios": n}
                    for name, n in counts.items()
                },
            )
        )
    return out


@router.post("/runs", status_code=201)
async def create_run(
    body: RunCreate, request: Request, store: StoreDep, runner: RunnerDep, suites: SuitesDep
) -> RunSummary:
    suite = suites.get(body.suite)
    if suite is None:
        raise HTTPException(404, f"No agent suite named '{body.suite}'")
    catalog: ModelCatalog = request.app.state.catalog
    try:
        installed = {m.name: m for m in await catalog.list()}
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    models = list(dict.fromkeys(body.models))
    for name in models:
        if name not in installed:
            raise HTTPException(404, f"Model '{name}' is not installed")
        if "tools" not in installed[name].capabilities:
            raise HTTPException(
                422, f"{name} does not support tool calling, so it cannot run agent tasks"
            )
    run_id = store.create_tool_run(
        suite.id,
        body.name or suite.title,
        models,
        body.think,
        [(s.id, s.category) for s in suite.scenarios],
        kind=KIND,
    )
    runner.schedule(run_id, suite)
    return summarise(store.tool_run(run_id, kind=KIND), store.tool_results(run_id, kind=KIND), True)


@router.get("/runs")
def list_runs(store: StoreDep, runner: RunnerDep) -> list[RunSummary]:
    return [
        summarise(r, store.tool_results(r["id"], kind=KIND), runner.active(r["id"]))
        for r in store.list_tool_runs(kind=KIND)
    ]


@router.get("/runs/{run_id}")
def get_run(run_id: str, store: StoreDep, runner: RunnerDep, suites: SuitesDep) -> RunDetail:
    run = store.tool_run(run_id, kind=KIND)
    if run is None:
        raise HTTPException(404, "No such run")
    results = store.tool_results(run_id, kind=KIND)
    scenarios = {
        s.id: s for s in (suites[run["suite"]].scenarios if run["suite"] in suites else [])
    }
    rows = []
    for r in results:
        s = scenarios.get(r["case_id"])
        rows.append(
            ResultRow(
                **{k: r.get(k) for k in ResultRow.model_fields if k in r},
                title=s.title if s else r["case_id"],
                prompt=s.prompt if s else "",
                apps=s.apps if s else [],
            )
        )
    summary = summarise(run, results, runner.active(run_id))
    return RunDetail(**summary.model_dump(), results=rows)


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str, store: StoreDep, runner: RunnerDep, suites: SuitesDep
) -> RunSummary:
    run = store.tool_run(run_id, kind=KIND)
    if run is None:
        raise HTTPException(404, "No such run")
    if run["suite"] not in suites:
        raise HTTPException(404, f"Suite '{run['suite']}' is no longer available")
    store.requeue_tool_run(run_id, kind=KIND)
    runner.schedule(run_id, suites[run["suite"]])
    return summarise(run, store.tool_results(run_id, kind=KIND), True)
