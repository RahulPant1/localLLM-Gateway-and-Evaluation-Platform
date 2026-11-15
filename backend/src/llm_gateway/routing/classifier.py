"""Asks the small classifier model for a category when the quick rules are unsure."""

import json

from pydantic import BaseModel, ValidationError

from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import GenerationRequest, ModelInfo, ProviderError
from llm_gateway.routing.types import CATEGORIES, COMPLEXITIES, Category, Complexity

# Only the start of a long request is needed to tell what kind of task it is.
MAX_PROMPT_CHARS = 2000
MAX_SOURCE_CHARS = 600

SYSTEM = """You label requests for a model router. Do not answer the request. \
Reply with JSON only.

Categories:
- general_qa: factual questions, explanations, summaries, translation, everyday writing
- classification: assign a label, category or sentiment to given input
- extraction: pull specific facts, names, dates or values out of given text
- structured_output: the main goal is data in a format such as JSON, CSV, YAML or a table
- reasoning: multi-step analysis, comparisons, trade-offs, logic or maths problems
- coding: write, explain, review or debug code
- medical: clinical content such as patient notes, lab results or medical reports

Complexity:
- low: one short, direct answer
- medium: several steps or a moderately long answer
- high: deep, multi-part or expert-level work"""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "complexity": {"type": "string", "enum": list(COMPLEXITIES)},
    },
    "required": ["category", "complexity"],
}


class Classification(BaseModel):
    category: Category
    complexity: Complexity


class ClassifierUnavailable(Exception):
    """The classifier could not be used; routing continues on the quick rules alone."""


def build_prompt(prompt: str, source_material: str | None) -> str:
    text = prompt[:MAX_PROMPT_CHARS]
    if len(prompt) > MAX_PROMPT_CHARS:
        text += "\n[request truncated]"
    if source_material:
        start = source_material[:MAX_SOURCE_CHARS]
        text += f"\n\n[The request comes with source material that begins:]\n{start}"
    return f"Label this request.\n\n<request>\n{text}\n</request>"


def parse(raw: str) -> Classification:
    """Validate the model's answer. Anything unexpected is rejected, never guessed at."""
    try:
        return Classification.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ClassifierUnavailable(f"invalid classifier answer: {raw[:120]!r}") from exc


async def classify(
    queue: ExecutionQueue, model: ModelInfo, prompt: str, source_material: str | None
) -> tuple[Classification, float]:
    """Run the classifier through the GPU queue. Returns the result and time taken in ms."""
    request = GenerationRequest(
        model=model.name,
        prompt=build_prompt(prompt, source_material),
        system=SYSTEM,
        think=False,  # thinking makes a trivial call take 20+ seconds
        max_tokens=60,
        response_schema=SCHEMA,
    )
    try:
        record = await queue.run(request, model)
    except ProviderError as exc:
        raise ClassifierUnavailable(f"{exc.code}: {exc}") from exc
    return parse(record.result.response), record.execution_ms
