"""Loads the evaluation rubric (config/rubric.yaml) and the judge instructions."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from llm_gateway.taxonomy import taxonomy_text

logger = logging.getLogger(__name__)

# Fixed: the database and the score schema use exactly these names.
DIMENSIONS = ("correctness", "relevance", "completeness", "instruction_adherence", "groundedness")
FALLBACK_VERSION = "v1"
FALLBACK_INSTRUCTIONS = (
    "Score each item on its own against its prompt. Give each rubric dimension a score from 0 "
    "to 10. Set groundedness to null when there is no source material. Do not give an overall "
    "score; the gateway computes it."
)


class Dimension(BaseModel):
    name: str
    weight: float = 1.0
    description: str = ""
    # Score range -> what a score in that range means.
    anchors: dict[str, str] = {}


class Rubric(BaseModel):
    version: str = FALLBACK_VERSION
    scale: str = "0-10"
    dimensions: list[Dimension] = [Dimension(name=name) for name in DIMENSIONS]
    # Set when the file could not be used; equal weights then apply.
    error: str | None = None

    @model_validator(mode="after")
    def _check_dimensions(self) -> "Rubric":
        names = [d.name for d in self.dimensions]
        if sorted(names) != sorted(DIMENSIONS):
            raise ValueError(f"dimensions must be exactly: {', '.join(DIMENSIONS)}")
        if any(d.weight < 0 for d in self.dimensions) or not any(
            d.weight > 0 for d in self.dimensions if d.name != "groundedness"
        ):
            raise ValueError("weights must not be negative, and at least one must be positive")
        return self

    @property
    def weights(self) -> dict[str, float]:
        return {d.name: d.weight for d in self.dimensions}


def load_rubric(path: Path) -> Rubric:
    """Read the rubric. A missing or invalid file never stops the app."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("the file must contain a mapping")
        return Rubric(**data)
    except (OSError, yaml.YAMLError, ValidationError, ValueError, TypeError) as exc:
        message = f"Rubric {path} could not be loaded: {str(exc).splitlines()[-1].strip()}"
        logger.error(message)
        return Rubric(error=message)


def load_judge_instructions(path: Path) -> str:
    """The judge instructions, followed by the fixed task type and domain lists."""
    try:
        text = path.read_text(encoding="utf-8").strip() or FALLBACK_INSTRUCTIONS
    except OSError:
        logger.error("Judge instructions %s could not be read", path)
        text = FALLBACK_INSTRUCTIONS
    return f"{text}\n\n{taxonomy_text()}"
