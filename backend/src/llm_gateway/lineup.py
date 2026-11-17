"""The model lineup: which models are compared, and how much of every test each has done.

Hiding a model takes it out of every screen, export, route and new run, but
deletes nothing: restoring it brings all its results back. Running all tests
for a model adds it to every existing batch and tool-use run, so its results
sit beside the other models' on the same prompts.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from llm_gateway.agents.runner import AgentRunner
from llm_gateway.agents.suites import AgentSuite
from llm_gateway.providers.base import ModelProvider, ProviderError
from llm_gateway.runner import Runner
from llm_gateway.store import Store
from llm_gateway.tooluse.runner import ToolRunner
from llm_gateway.tooluse.suites import ToolSuite

router = APIRouter(prefix="/api/lineup")


def _store(request: Request) -> Store:
    return request.app.state.store


StoreDep = Annotated[Store, Depends(_store)]


class BatchInfo(BaseModel):
    id: str
    name: str
    prompts: int


class ToolRunInfo(BaseModel):
    id: str
    name: str
    suite: str
    cases: int
    # "tool" for function-calling runs, "agent" for multi-step agent runs.
    kind: str = "tool"


class LineupEntry(BaseModel):
    name: str
    installed: bool
    hidden: bool
    tools: bool
    size_bytes: int | None = None
    parameter_size: str | None = None
    capabilities: list[str] = []
    # Batch id -> prompts answered / failed / waiting / judged.
    batches: dict[str, dict] = {}
    # Tool or agent run id -> cases passed / wrong / failed / waiting.
    tool_runs: dict[str, dict] = {}
    # Prompts and tool cases this model has never been run on.
    not_run: int = 0
    # Answers with no judge score yet.
    unjudged: int = 0


class Lineup(BaseModel):
    models: list[LineupEntry]
    batches: list[BatchInfo]
    tool_runs: list[ToolRunInfo]
    # Answers from models in the lineup that still need a judge.
    unjudged: int


class ModelName(BaseModel):
    model: str


class RunAllResult(BaseModel):
    model: str
    queued_answers: int
    queued_tool_cases: int
    queued_agent_scenarios: int = 0
    notes: list[str]


@router.get("")
async def lineup(request: Request, store: StoreDep) -> Lineup:
    provider: ModelProvider = request.app.state.provider
    suites: dict[str, ToolSuite] = request.app.state.tool_suites
    try:
        installed = {m.name: m for m in await provider.list_models()}
    except ProviderError:
        installed = {}
    batches = [
        BatchInfo(id=b.id, name=b.name, prompts=b.request_count) for b in store.list_batches()
    ]
    agent_suites: dict[str, AgentSuite] = request.app.state.agent_suites
    runs = [
        ToolRunInfo(
            id=r["id"],
            name=r["name"],
            suite=r["suite"],
            cases=len(suites[r["suite"]].cases) if r["suite"] in suites else 0,
        )
        for r in store.list_tool_runs()
    ] + [
        ToolRunInfo(
            id=r["id"],
            name=r["name"],
            suite=r["suite"],
            cases=len(agent_suites[r["suite"]].scenarios) if r["suite"] in agent_suites else 0,
            kind="agent",
        )
        for r in store.list_tool_runs(kind="agent")
    ]
    batch_cov = store.batch_coverage()
    tool_cov = store.tool_coverage()
    for model, per_run in store.tool_coverage(kind="agent").items():
        tool_cov.setdefault(model, {}).update(per_run)
    names = sorted(set(installed) | store.models_with_results(), key=str.lower)
    entries = []
    for name in names:
        info = installed.get(name)
        b = batch_cov.get(name, {})
        t = tool_cov.get(name, {})
        tools = info is not None and "tools" in info.capabilities
        not_run = sum(
            batch.prompts
            - sum(b.get(batch.id, {}).get(k, 0) for k in ("answered", "failed", "waiting"))
            for batch in batches
        )
        if tools:
            not_run += sum(run.cases - sum(t.get(run.id, {}).values()) for run in runs)
        entries.append(
            LineupEntry(
                name=name,
                installed=info is not None,
                hidden=name in store.hidden,
                tools=tools,
                size_bytes=info.size_bytes if info else None,
                parameter_size=info.parameter_size if info else None,
                capabilities=info.capabilities if info else [],
                batches=b,
                tool_runs=t,
                not_run=max(not_run, 0),
                unjudged=sum(c["answered"] - c["judged"] for c in b.values()),
            )
        )
    return Lineup(
        models=entries,
        batches=batches,
        tool_runs=runs,
        unjudged=sum(e.unjudged for e in entries if not e.hidden),
    )


@router.post("/hide")
def hide(body: ModelName, store: StoreDep) -> dict:
    """Take a model and its results out of every view. Nothing is deleted."""
    store.set_hidden(body.model, True)
    return {"model": body.model, "hidden": True}


@router.post("/restore")
def restore(body: ModelName, store: StoreDep) -> dict:
    store.set_hidden(body.model, False)
    return {"model": body.model, "hidden": False}


@router.post("/run-all")
async def run_all(body: ModelName, request: Request, store: StoreDep) -> RunAllResult:
    """Run a model on every existing batch prompt and tool-use case it has not done yet.

    Its answers then wait in the Evaluate tab for a judge; once scored, the
    Overview includes the model automatically.
    """
    runner: Runner = request.app.state.runner
    tool_runner: ToolRunner = request.app.state.tool_runner
    suites: dict[str, ToolSuite] = request.app.state.tool_suites
    provider: ModelProvider = request.app.state.provider
    try:
        info = next((m for m in await provider.list_models() if m.name == body.model), None)
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    if info is None:
        raise HTTPException(404, f"{body.model} is not installed in Ollama")
    notes = []
    if body.model in store.hidden:
        store.set_hidden(body.model, False)
        notes.append("The model was hidden; it is back in the lineup.")

    queued_answers = 0
    for batch in store.list_batches():
        if batch.counts["queued"] or batch.counts["running"]:
            notes.append(f"Skipped batch '{batch.name}': it is still running. Try again later.")
            continue
        request_ids = store.add_model_to_batch(batch.id, body.model)
        if request_ids:
            queued_answers += len(request_ids)
            await runner.schedule(store.batch_requests(batch.id), shadow_route=True)

    queued_cases = 0
    if "tools" not in info.capabilities:
        notes.append(
            "Tool-use and agent tests skipped: Ollama reports no tool support for this model."
        )
    else:
        for suite in suites.values():
            run_id = store.latest_tool_run(suite.id) or store.create_tool_run(
                suite.id, suite.title, [], False, []
            )
            added = store.add_model_to_tool_run(
                run_id, body.model, [(c.id, c.category) for c in suite.cases]
            )
            if added:
                queued_cases += added
                tool_runner.schedule(run_id, suite)
    queued_scenarios = 0
    if "tools" in info.capabilities:
        agent_runner: AgentRunner = request.app.state.agent_runner
        for agent_suite in request.app.state.agent_suites.values():
            run_id = store.latest_tool_run(agent_suite.id, kind="agent") or store.create_tool_run(
                agent_suite.id, agent_suite.title, [], False, [], kind="agent"
            )
            added = store.add_model_to_tool_run(
                run_id,
                body.model,
                [(s.id, s.category) for s in agent_suite.scenarios],
                kind="agent",
            )
            if added:
                queued_scenarios += added
                agent_runner.schedule(run_id, agent_suite)
    if not queued_answers and not queued_cases and not queued_scenarios:
        notes.append("Nothing to run: this model has already done every test.")
    return RunAllResult(
        model=body.model,
        queued_answers=queued_answers,
        queued_tool_cases=queued_cases,
        queued_agent_scenarios=queued_scenarios,
        notes=notes,
    )
