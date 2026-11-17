"""Endpoints for tool-calling test runs."""

from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from llm_gateway.catalog import ModelCatalog
from llm_gateway.providers.base import ProviderError
from llm_gateway.store import Store
from llm_gateway.tooluse.runner import ToolRunner
from llm_gateway.tooluse.suites import ToolSuite

router = APIRouter(prefix="/api/tool-use")


def _store(request: Request) -> Store:
    return request.app.state.store


def _runner(request: Request) -> ToolRunner:
    return request.app.state.tool_runner


def _suites(request: Request) -> dict[str, ToolSuite]:
    return request.app.state.tool_suites


StoreDep = Annotated[Store, Depends(_store)]
RunnerDep = Annotated[ToolRunner, Depends(_runner)]
SuitesDep = Annotated[dict[str, ToolSuite], Depends(_suites)]


class SuiteInfo(BaseModel):
    id: str
    title: str
    source: str
    case_count: int
    # Category -> what it tests and how many cases it has.
    categories: dict[str, dict]


class RunCreate(BaseModel):
    suite: str
    models: list[str] = Field(min_length=1)
    # Off by default: reasoning makes each call far slower and is not what is measured.
    think: bool = False
    name: str | None = None


class Tally(BaseModel):
    passed: int = 0
    # Of those passed, how many sent the call in the proper tool-call field
    # (the rest wrote it as JSON text in the answer).
    passed_native: int = 0
    wrong: int = 0
    # The case did not run to an answer (timeout, out of memory, interrupted).
    errors: int = 0
    waiting: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.wrong + self.errors + self.waiting


class ModelTally(Tally):
    model: str
    by_category: dict[str, Tally] = Field(default_factory=dict)
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
    reason: str | None
    prompt: str
    tool_names: list[str]
    expected: list[dict]
    calls: list[dict]
    call_format: str | None
    content: str | None
    thinking: str | None
    error_code: str | None
    error_detail: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    execution_ms: float | None


class RunDetail(RunSummary):
    results: list[ResultRow]


def _summarise(run: dict, results: list[dict], runner: ToolRunner) -> RunSummary:
    tallies = {m: ModelTally(model=m) for m in run["models"]}
    times: dict[str, list[float]] = defaultdict(list)
    for r in results:
        tally = tallies[r["model"]]
        bucket = tally.by_category.setdefault(r["category"], Tally())
        field = (
            "waiting"
            if r["status"] in ("queued", "running")
            else "errors"
            if r["status"] == "failed"
            else "passed"
            if r["passed"]
            else "wrong"
        )
        for t in (tally, bucket):
            setattr(t, field, getattr(t, field) + 1)
            if field == "passed" and r["call_format"] != "text":
                t.passed_native += 1
        if r["status"] == "completed" and r["execution_ms"] is not None:
            times[r["model"]].append(r["execution_ms"])
    for model, values in times.items():
        tallies[model].mean_ms = round(sum(values) / len(values), 1)
    finished = sum(1 for r in results if r["status"] in ("completed", "failed"))
    return RunSummary(
        id=run["id"],
        suite=run["suite"],
        name=run["name"],
        models=run["models"],
        think=run["think"],
        created_at=run["created_at"],
        total=len(results),
        finished=finished,
        running=runner.active(run["id"]),
        per_model=list(tallies.values()),
    )


@router.get("/suites")
def list_suites(suites: SuitesDep) -> list[SuiteInfo]:
    out = []
    for suite in suites.values():
        counts: dict[str, int] = defaultdict(int)
        for case in suite.cases:
            counts[case.category] += 1
        out.append(
            SuiteInfo(
                id=suite.id,
                title=suite.title,
                source=suite.source,
                case_count=len(suite.cases),
                categories={
                    name: {"description": suite.categories.get(name, ""), "cases": n}
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
        raise HTTPException(404, f"No tool suite named '{body.suite}'")
    models = list(dict.fromkeys(body.models))
    catalog: ModelCatalog = request.app.state.catalog
    try:
        installed = {m.name: m for m in await catalog.list()}
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    for name in models:
        info = installed.get(name)
        if info is None:
            raise HTTPException(404, f"Model '{name}' is not installed")
        if "tools" not in info.capabilities:
            raise HTTPException(
                422, f"{name} does not support tool calling, so it cannot be tested here"
            )
    run_id = store.create_tool_run(
        suite.id,
        body.name or f"{suite.title}",
        models,
        body.think,
        [(c.id, c.category) for c in suite.cases],
    )
    runner.schedule(run_id, suite)
    run = store.tool_run(run_id)
    return _summarise(run, store.tool_results(run_id), runner)


@router.get("/runs")
def list_runs(store: StoreDep, runner: RunnerDep) -> list[RunSummary]:
    return [_summarise(r, store.tool_results(r["id"]), runner) for r in store.list_tool_runs()]


@router.get("/runs/{run_id}")
def get_run(run_id: str, store: StoreDep, runner: RunnerDep, suites: SuitesDep) -> RunDetail:
    run = store.tool_run(run_id)
    if run is None:
        raise HTTPException(404, "No such run")
    results = store.tool_results(run_id)
    cases = {c.id: c for c in (suites[run["suite"]].cases if run["suite"] in suites else [])}
    rows = []
    for r in results:
        case = cases.get(r["case_id"])
        rows.append(
            ResultRow(
                id=r["id"],
                case_id=r["case_id"],
                category=r["category"],
                model=r["model"],
                status=r["status"],
                passed=r["passed"],
                reason=r["reason"],
                prompt=case.prompt if case else "",
                tool_names=[t["function"]["name"] for t in case.tools] if case else [],
                expected=case.expected if case else [],
                calls=r["calls"],
                call_format=r["call_format"],
                content=r["content"],
                thinking=r["thinking"],
                error_code=r["error_code"],
                error_detail=r["error_detail"],
                prompt_tokens=r["prompt_tokens"],
                output_tokens=r["output_tokens"],
                execution_ms=r["execution_ms"],
            )
        )
    return RunDetail(**_summarise(run, results, runner).model_dump(), results=rows)


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str, store: StoreDep, runner: RunnerDep, suites: SuitesDep
) -> RunSummary:
    run = store.tool_run(run_id)
    if run is None:
        raise HTTPException(404, "No such run")
    if run["suite"] not in suites:
        raise HTTPException(404, f"Suite '{run['suite']}' is no longer available")
    # A run that is still going picks newly queued cases up on its next pass.
    store.requeue_tool_run(run_id)
    runner.schedule(run_id, suites[run["suite"]])
    return _summarise(run, store.tool_results(run_id), runner)
