"""Turns a category and complexity into a model, a fallback and escalation advice.

Deterministic: the same inputs always give the same choice, so the policy can
be read in config/models.yaml and tested without a model.
"""

from pydantic import BaseModel

from llm_gateway.providers.base import ModelInfo
from llm_gateway.routing.registry import Registry

# Leave room in the context window for the answer.
INPUT_BUDGET_FRACTION = 0.75


class Selection(BaseModel):
    model: str | None
    fallback: str | None
    candidates: list[str]
    notes: list[str]
    escalate: bool
    escalation_reason: str | None


def select(
    registry: Registry,
    category: str,
    complexity: str,
    *,
    installed: dict[str, ModelInfo] | None,
    loaded: set[str],
    estimated_tokens: int,
) -> Selection:
    """Pick the model for a task.

    `installed` is None when the provider could not be asked; the policy
    order is then used as it stands and availability is unknown.
    """
    notes: list[str] = []
    reasons: list[str] = []

    def available(name: str) -> bool:
        return installed is None or name in installed

    def fits(name: str) -> bool:
        info = installed.get(name) if installed else None
        return info is None or estimated_tokens <= info.num_ctx * INPUT_BUDGET_FRACTION

    # Flatten the policy groups, putting an already-loaded model first within its group.
    ordered: list[str] = []
    for group in registry.groups_for(category, complexity):
        present = [name for name in group if available(name)]
        missing = [name for name in group if not available(name)]
        if missing:
            notes.append(f"not installed: {', '.join(missing)}")
        preferred = sorted(present, key=lambda name: name not in loaded)
        if len(present) > 1 and preferred[0] in loaded:
            notes.append(f"{preferred[0]} preferred among equals because it is already loaded")
        ordered.extend(name for name in preferred if name not in ordered)

    if not ordered:
        ordered = [name for name in registry.last_resort if available(name)]
        if not ordered and installed:
            ordered = sorted(installed)
        if ordered:
            notes.append(f"no model profiled for {category} is installed; using a general model")
            reasons.append(f"No model suited to {category} tasks is installed")

    if installed is None:
        notes.append("model availability unknown (Ollama could not be reached)")

    # Models whose context window holds the input come first.
    fitting = [name for name in ordered if fits(name)]
    if ordered and not fitting:
        reasons.append(
            f"The input (about {estimated_tokens} tokens) is longer than any candidate model's "
            "context window, so part of it will be cut off"
        )
    elif fitting and fitting[0] != ordered[0]:
        notes.append(
            f"{ordered[0]} skipped: the input (about {estimated_tokens} tokens) "
            "does not fit its context window"
        )
    ranked = fitting + [name for name in ordered if name not in fitting]

    if complexity == "high":
        reasons.append(
            "High-complexity task: local 4-8B models may fall short, so check the answer "
            "carefully or use a stronger model"
        )
    if not ranked:
        reasons.append("No local model is available")

    return Selection(
        model=ranked[0] if ranked else None,
        fallback=ranked[1] if len(ranked) > 1 else None,
        candidates=ranked,
        notes=notes,
        escalate=bool(reasons),
        escalation_reason="; ".join(reasons) or None,
    )
