"""Shapes of stored records and API payloads for requests, batches and evaluations."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_gateway.providers.base import ChatMessage
from llm_gateway.routing.types import RoutingDecision
from llm_gateway.rubric import Rubric
from llm_gateway.taxonomy import DOMAIN_LABELS, TASK_TYPE_LABELS, normalise

ExecutionStatus = Literal["queued", "running", "completed", "failed"]
RoutingMode = Literal["manual", "automatic"]


class DimensionScores(BaseModel):
    """Scores from 0 to 10. Groundedness is null ("not applicable") when the
    request had no source material."""

    correctness: float = Field(ge=0, le=10)
    relevance: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    instruction_adherence: float = Field(ge=0, le=10)
    groundedness: float | None = Field(default=None, ge=0, le=10)


def overall_score(scores: DimensionScores, weights: dict[str, float]) -> float | None:
    """Weighted mean of the dimensions that were scored, using the rubric's weights.

    Computed by the app, never taken from the judge, so it is reproducible.
    """
    scored = {k: v for k, v in scores.model_dump().items() if v is not None}
    total_weight = sum(weights.get(k, 1.0) for k in scored)
    if total_weight <= 0:
        return None
    return round(sum(v * weights.get(k, 1.0) for k, v in scored.items()) / total_weight, 2)


class EvaluationInput(BaseModel):
    execution_id: str
    # Who judged: a model id, "claude-code", "human", ...
    judge: str = Field(min_length=1)
    # Omitted = the rubric version currently in config/rubric.yaml.
    rubric_version: str | None = None
    scores: DimensionScores
    # What kind of task the prompt is, from the fixed lists in taxonomy.py.
    # Null on evaluations stored before these fields existed.
    task_type: str | None = None
    domain: str | None = None
    summary: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendations: list[str] = []
    critical_issues: list[str] = []

    @field_validator("task_type", mode="before")
    @classmethod
    def _known_task_type(cls, value: object) -> str | None:
        return normalise(value, TASK_TYPE_LABELS)

    @field_validator("domain", mode="before")
    @classmethod
    def _known_domain(cls, value: object) -> str | None:
        return normalise(value, DOMAIN_LABELS)


class EvaluationImport(BaseModel):
    evaluations: list[EvaluationInput] = Field(min_length=1)


class Evaluation(EvaluationInput):
    id: str
    status: Literal["completed", "failed"] = "completed"
    overall_score: float | None
    created_at: datetime


class Execution(BaseModel):
    id: str
    request_id: str
    model: str
    status: ExecutionStatus
    response: str | None = None
    # Reasoning text, stored apart from the answer and never sent to a judge.
    thinking: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    queue_wait_ms: float | None = None
    execution_ms: float | None = None
    # Provider timings and token counts (load_ms, tokens_per_second, ...).
    metrics: dict | None = None
    warnings: list[str] = []
    evaluations: list[Evaluation] = []
    # How many follow-up messages continue the conversation from this answer.
    follow_up_count: int = 0


class RequestRecord(BaseModel):
    id: str
    batch_id: str | None = None
    # Set on a follow-up: the answer this message continues from.
    parent_execution_id: str | None = None
    prompt: str
    system: str | None = None
    source_material: str | None = None
    routing_mode: RoutingMode
    think: bool | None = None
    max_tokens: int | None = None
    created_at: datetime
    executions: list[Execution] = []
    # The router's decision. Null for manual requests, and for automatic ones
    # until routing finishes. On batch prompts it is a "shadow" decision.
    routing: RoutingDecision | None = None


class PromptInput(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None
    source_material: str | None = None


class RequestCreate(PromptInput):
    # Omitted = automatic routing. One model = manual run; several = comparison.
    models: list[str] | None = Field(default=None, min_length=1)
    think: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)


class RoutePreview(PromptInput):
    # Also ask the classifier model if the quick rules are unsure (slower).
    use_llm: bool = False


class FollowUpCreate(BaseModel):
    prompt: str = Field(min_length=1)
    # None = keep the conversation's setting.
    think: bool | None = None


class ThreadTurn(BaseModel):
    request_id: str
    prompt: str
    source_material: str | None
    created_at: datetime
    execution: Execution


class ContextUsage(BaseModel):
    """Estimate of how full the model's context window is before the next message."""

    used_tokens: int
    limit_tokens: int | None
    fraction: float | None
    state: Literal["ok", "warning", "full"]


class Thread(BaseModel):
    model: str
    system: str | None
    turns: list[ThreadTurn]
    context: ContextUsage
    # The answer a new message would continue from; None while a turn is in
    # progress or when no turn has completed.
    reply_to: str | None


class BatchCreate(BaseModel):
    name: str | None = None
    prompts: list[PromptInput | str] = Field(min_length=1)
    # Omitted = every model installed at submission time.
    models: list[str] | None = Field(default=None, min_length=1)
    think: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)


class BatchSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    models: list[str]
    request_count: int
    # Number of executions in each status.
    counts: dict[ExecutionStatus, int]
    evaluated_count: int


class BatchDetail(BatchSummary):
    requests: list[RequestRecord]


class ExportItem(BaseModel):
    execution_id: str
    prompt: str
    system: str | None
    source_material: str | None
    # Earlier turns, when the response is a follow-up in a conversation.
    history: list[ChatMessage] = []
    response: str
    # Present only when the export is not blind.
    model: str | None = None


class BatchExport(BaseModel):
    batch_id: str
    batch_name: str
    blind: bool
    rubric: Rubric
    # How to judge and what to return (config/judge_prompt.md).
    instructions: str
    items: list[ExportItem]


class JudgePackageRequest(BaseModel):
    """What to hand to a judge: any mix of single requests and whole batches."""

    request_ids: list[str] = []
    batch_ids: list[str] = []
    # Hide model names from the judge.
    blind: bool = True
    # Leave out outputs this judge has already scored.
    exclude_judge: str | None = None


class JudgePackage(BaseModel):
    """Everything a judge needs, as one text to copy."""

    text: str
    item_count: int
    rubric_version: str
    blind: bool


class ModelScores(BaseModel):
    """Mean scores one judge gave one model across a batch."""

    model: str
    judge: str
    evaluated: int
    overall: float | None
    dimensions: dict[str, float | None]
    critical_issues: int
