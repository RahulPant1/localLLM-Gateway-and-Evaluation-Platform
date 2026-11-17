"""Summary numbers for the Overview screen, computed from stored requests.

Read-only: nothing here changes the database. The whole period is loaded and
counted in Python, which is fast enough for the thousands of rows a local
gateway produces.
"""

from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

from pydantic import BaseModel

from llm_gateway.rubric import DIMENSIONS
from llm_gateway.schemas import Execution, RequestRecord
from llm_gateway.taxonomy import DOMAIN_LABELS, TASK_TYPE_LABELS

# A generation whose model load took less than this found the model already in
# VRAM ("warm"). Measured: warm loads are under 200 ms, swaps take 5 s or more.
WARM_LOAD_MS = 1000.0
NOT_ROUTED = "not_routed"
UNCATEGORISED = "uncategorised"


class Share(BaseModel):
    key: str
    count: int
    share: float


class RoutingOverview(BaseModel):
    # Decisions where the router chose the model that ran.
    live: int
    # Decisions recorded on batch prompts for analysis only.
    shadow: int
    by_model: list[Share]
    by_category: list[Share]
    decided_by: dict[str, int]
    fallback_used: int
    escalations: int
    mean_confidence: float | None


class CloudEstimate(BaseModel):
    """What the same token counts would cost at a cloud model's list price."""

    model: str
    input_per_mtok: float
    output_per_mtok: float
    usd: float


class PerformanceOverview(BaseModel):
    executions: int
    completed: int
    failed: int
    prompt_tokens: int
    output_tokens: int
    # Time the GPU spent generating, and time spent loading models into VRAM.
    generation_seconds: float
    load_seconds: float
    tokens_per_second: float | None
    # Share of completed generations that needed no model swap.
    warm_rate: float | None
    mean_swap_ms: float | None
    mean_queue_wait_ms: float | None
    cloud_estimate: CloudEstimate | None


class GroupRow(BaseModel):
    """One model or one task category."""

    key: str
    executions: int
    completed: int
    failed: int
    median_ms: float | None
    slowest_ms: float | None
    output_tokens: int
    tokens_per_second: float | None
    warm_rate: float | None


class ModelScore(BaseModel):
    model: str
    # Prompts this row is computed from (the shared prompts when comparable).
    evaluated: int
    # Every scored answer from this model, shared or not.
    scored_total: int = 0
    overall: float | None
    dimensions: dict[str, float | None]
    critical_issues: int


class RouterCheck(BaseModel):
    """The router's pick against the judge's scores, on prompts where the
    router's pick and at least one other model were both scored."""

    prompts: int
    picked_top: int
    router_mean: float | None
    # The single model that would have scored best on the same prompts, among
    # models scored on all of them.
    best_fixed_model: str | None
    best_fixed_mean: float | None
    # Mean of the best score on each prompt: the ceiling for any router.
    best_possible_mean: float | None = None


class CellScore(BaseModel):
    mean: float
    answers: int


class TaskRow(BaseModel):
    """Scores for one task type (or domain), per model."""

    key: str
    label: str
    prompts: int
    models: dict[str, CellScore]
    # Highest mean; null when fewer than two models were scored or they tie.
    best_model: str | None


class EvaluationOverview(BaseModel):
    scored: int
    unscored: int
    judges: list[str]
    # Prompts with a score for every model in the comparison. The leaderboard and
    # the task grids use only these, so every model is judged on the same prompts.
    compared_prompts: int = 0
    # Prompts with a score for at least one model.
    scored_prompts: int = 0
    # False when no prompt has a score for every model: rows then use each
    # model's own prompts and are not directly comparable.
    comparable: bool = True
    by_model: list[ModelScore]
    # Rows by the judge's task type and domain; `models` lists every scored model.
    models: list[str]
    by_task: list[TaskRow]
    by_domain: list[TaskRow]
    router_check: RouterCheck


class Overview(BaseModel):
    days: int | None
    since: datetime | None
    requests: int
    routing: RoutingOverview
    performance: PerformanceOverview
    by_model: list[GroupRow]
    by_category: list[GroupRow]
    evaluation: EvaluationOverview


def _mean(values: list[float], digits: int = 2) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def _shares(counts: Counter) -> list[Share]:
    total = sum(counts.values())
    return [
        Share(key=key, count=count, share=round(count / total, 3))
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _metric(execution: Execution, name: str) -> float:
    value = (execution.metrics or {}).get(name)
    return float(value) if isinstance(value, int | float) else 0.0


def _is_warm(execution: Execution) -> bool | None:
    load = (execution.metrics or {}).get("load_ms")
    return None if load is None else load < WARM_LOAD_MS


def _speed(executions: list[Execution]) -> float | None:
    seconds = sum(_metric(e, "generation_ms") for e in executions) / 1000
    tokens = sum(_metric(e, "output_tokens") for e in executions)
    return round(tokens / seconds, 1) if seconds > 0 else None


def _warm_rate(executions: list[Execution]) -> float | None:
    known = [w for w in map(_is_warm, executions) if w is not None]
    return round(sum(known) / len(known), 3) if known else None


def _latest(execution: Execution) -> list:
    """One evaluation per judge: the most recent. A judge scoring an answer
    again replaces its earlier score instead of being counted twice."""
    latest: dict[str, object] = {}
    for evaluation in sorted(execution.evaluations, key=lambda v: v.created_at):
        if evaluation.status == "completed":
            latest[evaluation.judge] = evaluation
    return list(latest.values())


def _score(execution: Execution) -> float | None:
    """The answer's overall score: the mean of each judge's latest score."""
    return _mean([v.overall_score for v in _latest(execution) if v.overall_score is not None])


def _scored_answers(request: RequestRecord) -> dict[str, Execution]:
    """model -> its scored answer to this prompt (the latest attempt that was scored)."""
    found: dict[str, Execution] = {}
    for execution in request.executions:
        if execution.status == "completed" and _score(execution) is not None:
            found[execution.model] = execution
    return found


def _routing(requests: list[RequestRecord]) -> RoutingOverview:
    decisions = [r.routing for r in requests if r.routing is not None]
    return RoutingOverview(
        live=sum(not d.shadow for d in decisions),
        shadow=sum(d.shadow for d in decisions),
        by_model=_shares(Counter(d.selected_model for d in decisions if d.selected_model)),
        by_category=_shares(Counter(d.category for d in decisions)),
        decided_by=dict(Counter(d.decided_by for d in decisions)),
        fallback_used=sum(d.fallback_used for d in decisions),
        escalations=sum(d.escalate for d in decisions),
        mean_confidence=_mean([d.confidence for d in decisions]),
    )


def _performance(
    executions: list[Execution], pricing: tuple[str, float, float] | None
) -> PerformanceOverview:
    completed = [e for e in executions if e.status == "completed"]
    prompt_tokens = int(sum(_metric(e, "prompt_tokens") for e in completed))
    output_tokens = int(sum(_metric(e, "output_tokens") for e in completed))
    swaps = [_metric(e, "load_ms") for e in completed if _is_warm(e) is False]
    estimate = None
    if pricing:
        name, input_price, output_price = pricing
        estimate = CloudEstimate(
            model=name,
            input_per_mtok=input_price,
            output_per_mtok=output_price,
            usd=round((prompt_tokens * input_price + output_tokens * output_price) / 1e6, 4),
        )
    return PerformanceOverview(
        executions=len(executions),
        completed=len(completed),
        failed=sum(e.status == "failed" for e in executions),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        generation_seconds=round(sum(_metric(e, "generation_ms") for e in completed) / 1000, 1),
        load_seconds=round(sum(_metric(e, "load_ms") for e in completed) / 1000, 1),
        tokens_per_second=_speed(completed),
        warm_rate=_warm_rate(completed),
        mean_swap_ms=_mean(swaps, 0),
        mean_queue_wait_ms=_mean(
            [e.queue_wait_ms for e in executions if e.queue_wait_ms is not None], 0
        ),
        cloud_estimate=estimate,
    )


def _groups(grouped: dict[str, list[Execution]]) -> list[GroupRow]:
    rows = []
    for key, executions in grouped.items():
        completed = [e for e in executions if e.status == "completed"]
        times = [e.execution_ms for e in completed if e.execution_ms is not None]
        rows.append(
            GroupRow(
                key=key,
                executions=len(executions),
                completed=len(completed),
                failed=sum(e.status == "failed" for e in executions),
                median_ms=round(median(times)) if times else None,
                slowest_ms=round(max(times)) if times else None,
                output_tokens=int(sum(_metric(e, "output_tokens") for e in completed)),
                tokens_per_second=_speed(completed),
                warm_rate=_warm_rate(completed),
            )
        )
    return sorted(rows, key=lambda row: (-row.executions, row.key))


def _router_check(requests: list[RequestRecord]) -> RouterCheck:
    # One entry per comparable prompt: {model: score}, plus the router's pick.
    compared: list[tuple[str, dict[str, float]]] = []
    for request in requests:
        decision = request.routing
        if decision is None or not decision.selected_model:
            continue
        scores = {m: _score(e) for m, e in _scored_answers(request).items()}
        if len(scores) >= 2 and decision.selected_model in scores:
            compared.append((decision.selected_model, scores))

    picked = [scores[model] for model, scores in compared]
    best_model, best_mean = None, None
    if compared:
        on_all = set.intersection(*(set(scores) for _, scores in compared))
        means = {m: _mean([scores[m] for _, scores in compared]) for m in on_all}
        if means:
            best_model = max(sorted(means), key=lambda m: means[m])
            best_mean = means[best_model]
    return RouterCheck(
        prompts=len(compared),
        picked_top=sum(scores[model] >= max(scores.values()) for model, scores in compared),
        router_mean=_mean(picked),
        best_fixed_model=best_model,
        best_fixed_mean=best_mean,
        best_possible_mean=_mean([max(scores.values()) for _, scores in compared]),
    )


def _label_of(request: RequestRecord, field: str) -> str:
    """The label most judges gave this prompt; the earliest wins a tie."""
    labels = [
        value
        for execution in request.executions
        for evaluation in execution.evaluations
        if (value := getattr(evaluation, field))
    ]
    return Counter(labels).most_common(1)[0][0] if labels else UNCATEGORISED


def _task_rows(requests: list[RequestRecord], field: str, labels: dict[str, str]) -> list[TaskRow]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompts: Counter = Counter()
    for request in requests:
        scored = {m: _score(e) for m, e in _scored_answers(request).items()}
        if not scored:
            continue
        key = _label_of(request, field)
        prompts[key] += 1
        for model, score in scored.items():
            scores[key][model].append(score)

    rows = []
    for key, per_model in scores.items():
        cells = {
            model: CellScore(mean=_mean(values), answers=len(values))
            for model, values in sorted(per_model.items())
        }
        ranked = sorted(cells.items(), key=lambda item: -item[1].mean)
        clear_winner = len(ranked) >= 2 and ranked[0][1].mean > ranked[1][1].mean
        rows.append(
            TaskRow(
                key=key,
                label=labels.get(key, "Not categorised"),
                prompts=prompts[key],
                models=cells,
                best_model=ranked[0][0] if clear_winner else None,
            )
        )
    # Uncategorised last, then by number of prompts.
    return sorted(rows, key=lambda row: (row.key == UNCATEGORISED, -row.prompts, row.label))


def _evaluation(requests: list[RequestRecord]) -> EvaluationOverview:
    judges: set[str] = set()
    scored = unscored = 0
    for request in requests:
        for execution in request.executions:
            if execution.status != "completed":
                continue
            if execution.evaluations:
                scored += 1
            elif request.parent_execution_id is None:
                # Follow-up answers are not sent to judges, so they are not "unscored".
                unscored += 1
            judges.update(v.judge for v in execution.evaluations)

    # Rank models on the same prompts only: a model scored on an easier set,
    # or on more judges, would otherwise look better or worse than it is.
    answers = {r.id: _scored_answers(r) for r in requests}
    scored_requests = [r for r in requests if answers[r.id]]
    models = sorted({m for found in answers.values() for m in found})
    shared = [r for r in scored_requests if set(answers[r.id]) >= set(models)]
    comparable = bool(shared)
    basis = shared if comparable else scored_requests

    totals = Counter(m for r in scored_requests for m in answers[r.id])
    per_model: dict[str, list[Execution]] = defaultdict(list)
    for request in basis:
        for model, execution in answers[request.id].items():
            per_model[model].append(execution)

    rows = []
    for model, executions in per_model.items():
        evaluations = [v for e in executions for v in _latest(e)]
        dimensions = {}
        for name in DIMENSIONS:
            # Per answer first (across judges), then across answers.
            per_answer = [
                _mean([x for v in _latest(e) if (x := getattr(v.scores, name)) is not None])
                for e in executions
            ]
            dimensions[name] = _mean([x for x in per_answer if x is not None])
        rows.append(
            ModelScore(
                model=model,
                evaluated=len(executions),
                scored_total=totals[model],
                overall=_mean([s for e in executions if (s := _score(e)) is not None]),
                dimensions=dimensions,
                critical_issues=sum(len(v.critical_issues) for v in evaluations),
            )
        )
    rows.sort(key=lambda row: (-(row.overall or 0), row.model))
    return EvaluationOverview(
        scored=scored,
        unscored=unscored,
        judges=sorted(judges),
        compared_prompts=len(shared),
        scored_prompts=len(scored_requests),
        comparable=comparable,
        by_model=rows,
        models=models,
        by_task=_task_rows(basis, "task_type", TASK_TYPE_LABELS),
        by_domain=_task_rows(basis, "domain", DOMAIN_LABELS),
        router_check=_router_check(requests),
    )


def build_overview(
    requests: list[RequestRecord],
    *,
    days: int | None = None,
    since: datetime | None = None,
    pricing: tuple[str, float, float] | None = None,
) -> Overview:
    executions = [e for r in requests for e in r.executions]
    per_model: dict[str, list[Execution]] = defaultdict(list)
    per_category: dict[str, list[Execution]] = defaultdict(list)
    for request in requests:
        category = request.routing.category if request.routing else NOT_ROUTED
        for execution in request.executions:
            per_model[execution.model].append(execution)
            per_category[category].append(execution)
    return Overview(
        days=days,
        since=since,
        requests=len(requests),
        routing=_routing(requests),
        performance=_performance(executions, pricing),
        by_model=_groups(per_model),
        by_category=_groups(per_category),
        evaluation=_evaluation(requests),
    )
