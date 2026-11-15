"""Preparing stored outputs for a judge, and summarising the scores that come back.

Used by both the HTTP API and scripts/evals.py, so Claude Code can judge a
batch without the backend running.
"""

import json
from collections import defaultdict

from llm_gateway.conversation import history_messages
from llm_gateway.rubric import Rubric
from llm_gateway.schemas import BatchExport, ExportItem, ModelScores, RequestRecord
from llm_gateway.store import Store


def build_items(
    store: Store,
    requests: list[RequestRecord],
    *,
    blind: bool = True,
    exclude_judge: str | None = None,
) -> list[ExportItem]:
    """Completed outputs of the given requests, shaped for a judge.

    Blind leaves out model names and orders each prompt's outputs by their
    random id, so position does not reveal the model. Thinking text is never
    included. `exclude_judge` drops outputs that judge has already scored.
    """
    items = []
    for request in requests:
        completed = [e for e in request.executions if e.status == "completed"]
        if exclude_judge:
            completed = [
                e for e in completed if all(v.judge != exclude_judge for v in e.evaluations)
            ]
        if blind:
            completed.sort(key=lambda e: e.id)
        history = list(request.history)
        if request.parent_execution_id:
            history = history_messages(store.ancestors(request.parent_execution_id))
        for execution in completed:
            items.append(
                ExportItem(
                    execution_id=execution.id,
                    prompt=request.prompt,
                    system=request.system,
                    source_material=request.source_material,
                    history=history,
                    response=execution.response or "",
                    model=None if blind else execution.model,
                )
            )
    return items


def build_export(
    store: Store,
    batch_id: str,
    *,
    rubric: Rubric,
    instructions: str,
    blind: bool = True,
    exclude_judge: str | None = None,
) -> BatchExport | None:
    """Completed outputs of one batch, with the rubric and judge instructions."""
    summary = store.get_batch(batch_id)
    if summary is None:
        return None
    return BatchExport(
        batch_id=batch_id,
        batch_name=summary.name,
        blind=blind,
        rubric=rubric,
        instructions=instructions,
        items=build_items(
            store, store.batch_requests(batch_id), blind=blind, exclude_judge=exclude_judge
        ),
    )


def render_judge_package(
    instructions: str, rubric: Rubric, items: list[ExportItem], label: str
) -> str:
    """One self-contained text to paste into a chat with a judge: the
    instructions, the rubric, and the items to score."""
    lines = [instructions, "", f"# Rubric (version {rubric.version})", ""]
    lines.append(f"Scale: {rubric.scale}.")
    for dimension in rubric.dimensions:
        lines += ["", f"## {dimension.name} (weight {dimension.weight:g})"]
        if dimension.description:
            lines.append(dimension.description)
        lines += [
            f"- {score_range}: {meaning}" for score_range, meaning in dimension.anchors.items()
        ]
    data = [item.model_dump(exclude_none=True) for item in items]
    for item in data:
        if not item["history"]:
            del item["history"]
    lines += [
        "",
        f"# Items to judge ({len(data)}, from {label})",
        "",
        "```json",
        json.dumps(data, indent=2, ensure_ascii=False),
        "```",
        "",
        f"Judge all {len(data)} items. Reply with only the JSON object described above "
        '("evaluations": [...]), one entry per item, so it can be stored as it is.',
    ]
    return "\n".join(lines)


def summarise_batch(store: Store, batch_id: str) -> list[ModelScores]:
    """Mean scores per model and judge for a batch."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for request in store.batch_requests(batch_id):
        for execution in request.executions:
            for evaluation in execution.evaluations:
                groups[(execution.model, evaluation.judge)].append(evaluation)

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    rows = []
    for (model, judge), evaluations in sorted(groups.items()):
        dimensions = {}
        for name in evaluations[0].scores.model_dump():
            scored = [getattr(v.scores, name) for v in evaluations]
            dimensions[name] = mean([s for s in scored if s is not None])
        rows.append(
            ModelScores(
                model=model,
                judge=judge,
                evaluated=len(evaluations),
                overall=mean([v.overall_score for v in evaluations if v.overall_score is not None]),
                dimensions=dimensions,
                critical_issues=sum(len(v.critical_issues) for v in evaluations),
            )
        )
    return rows
