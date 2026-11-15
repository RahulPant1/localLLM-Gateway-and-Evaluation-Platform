"""Loads config/models.yaml: what each model is good at, and the routing policy."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from llm_gateway.routing.types import CATEGORIES, COMPLEXITIES

logger = logging.getLogger(__name__)

# A policy entry is a list of groups; models in one group are equally suitable.
Groups = list[list[str]]


class ModelProfile(BaseModel):
    strengths: str = ""


class Registry(BaseModel):
    classifier_model: str | None = None
    models: dict[str, ModelProfile] = {}
    policy: dict[str, dict[str, Groups]] = {}
    last_resort: list[str] = []
    # Set when the file could not be read; routing then works without profiles.
    error: str | None = None

    @field_validator("policy", mode="before")
    @classmethod
    def _normalise_policy(cls, value: object) -> object:
        """Accept `[a, [b, c], d]` and turn bare names into groups of one."""
        if not isinstance(value, dict):
            raise ValueError("policy must be a mapping of category to model lists")
        result: dict[str, dict[str, Groups]] = {}
        for category, levels in value.items():
            if category not in CATEGORIES:
                raise ValueError(f"unknown category '{category}'")
            if not isinstance(levels, dict):
                raise ValueError(f"policy for '{category}' must have a 'default' list")
            result[category] = {}
            for level, entries in levels.items():
                if level != "default" and level not in COMPLEXITIES:
                    raise ValueError(f"unknown complexity '{level}' under '{category}'")
                result[category][level] = [
                    [entry] if isinstance(entry, str) else list(entry) for entry in entries or []
                ]
        return result

    def groups_for(self, category: str, complexity: str) -> Groups:
        levels = self.policy.get(category, {})
        return levels.get(complexity) or levels.get("default") or []

    def categories_of(self, model: str) -> list[str]:
        """Categories for which this model is a listed candidate."""
        return [
            category
            for category, levels in self.policy.items()
            if any(model in group for groups in levels.values() for group in groups)
        ]

    def is_profiled(self, model: str) -> bool:
        return model in self.models or bool(self.categories_of(model))


def load_registry(path: Path) -> Registry:
    """Read the registry. A missing or invalid file never stops the app."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("the file must contain a mapping")
        return Registry(**data)
    except (OSError, yaml.YAMLError, ValidationError, ValueError, TypeError) as exc:
        message = f"Model registry {path} could not be loaded: {str(exc).splitlines()[0]}"
        logger.error(message)
        return Registry(error=message)
