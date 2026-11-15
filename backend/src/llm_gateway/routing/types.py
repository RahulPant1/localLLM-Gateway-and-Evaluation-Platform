"""Shared routing types."""

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

Category = Literal[
    "general_qa",
    "classification",
    "extraction",
    "structured_output",
    "reasoning",
    "coding",
    "medical",
]
Complexity = Literal["low", "medium", "high"]
CATEGORIES: tuple[str, ...] = get_args(Category)
COMPLEXITIES: tuple[str, ...] = get_args(Complexity)

# How the category was decided.
#   signals           the quick rules were clear enough
#   llm               the rules were unsure and the classifier model decided
#   signals_fallback  the classifier was needed but unavailable or gave an invalid answer
#   none              routing did not complete
DecidedBy = Literal["signals", "llm", "signals_fallback", "none"]


class SignalReport(BaseModel):
    """What the quick rules found."""

    scores: dict[str, float]
    matched: dict[str, list[str]]
    category: Category
    runner_up: Category | None
    # 0-1: how clearly the rules point at one category.
    clarity: float
    complexity: Complexity
    complexity_reasons: list[str]
    estimated_tokens: int


class RoutingDraft(BaseModel):
    """A routing decision before it is stored."""

    category: Category
    complexity: Complexity
    selected_model: str | None
    fallback_model: str | None
    # An estimate, not a guarantee. Its reliability is measured later against
    # evaluation scores.
    confidence: float = Field(ge=0, le=1)
    explanation: str
    escalate: bool
    escalation_reason: str | None
    decided_by: DecidedBy
    candidates: list[str]
    signals: SignalReport | None
    # The classifier model's answer or the reason it could not be used.
    llm: dict | None = None


class RoutingDecision(RoutingDraft):
    request_id: str
    # True for batch prompts: recorded for later analysis, not used to pick the model.
    shadow: bool
    # True when the selected model failed and the fallback was run instead.
    fallback_used: bool = False
    created_at: datetime
