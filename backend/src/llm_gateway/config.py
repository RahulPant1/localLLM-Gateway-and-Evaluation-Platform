"""Application configuration.

Settings come from, in order of precedence: real environment variables, the
`.env` file in the project root, then the defaults below. Every variable is
prefixed with `GATEWAY_` so it cannot collide with unrelated variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/llm_gateway/config.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_url: str = "http://127.0.0.1:11434"
    database_path: Path = Path("data/gateway.db")
    judge_model: str = "claude-opus-5-5"
    # Context window requested for models whose Modelfile does not set one.
    default_num_ctx: int = Field(default=8192, gt=0)
    # Model registry and routing policy.
    registry_path: Path = Path("config/models.yaml")
    # Evaluation rubric and the instructions given to every judge.
    rubric_path: Path = Path("config/rubric.yaml")
    judge_prompt_path: Path = Path("config/judge_prompt.md")
    # Folder of tool-calling test suites (JSON).
    tool_suites_dir: Path = Path("config/tool_suites")
    # "auto": ask the classifier model when the quick rules are unsure. "off": rules only.
    router_llm: Literal["auto", "off"] = "auto"
    # Below this clarity (0-1) the quick rules count as unsure.
    router_clarity_threshold: float = Field(default=0.4, ge=0, le=1)
    # Upper limit for one generation, including any model load.
    generation_timeout_seconds: float = Field(default=300.0, gt=0)
    # Memory guard: when free system memory falls below this many MB, loaded
    # models are unloaded before the next generation so Ollama's prompt cache
    # is released. 0 switches the guard off.
    min_free_memory_mb: int = Field(default=1500, ge=0)
    # Whether requests through the OpenAI-compatible endpoint think by default.
    # Off keeps other tools responsive; a request can ask for it with
    # reasoning_effort "medium" or "high".
    api_think: bool = False
    # Cloud model the Overview screen prices the local token counts against
    # (US dollars per million tokens). Set both prices to 0 to hide the estimate.
    cloud_compare_model: str = "claude-opus-5-5"
    cloud_input_per_mtok: float = Field(default=4.0, ge=0)
    cloud_output_per_mtok: float = Field(default=20.0, ge=0)
    # SecretStr keeps the key out of repr(), logs and JSON dumps by default.
    anthropic_api_key: SecretStr | None = None

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        # `GATEWAY_ANTHROPIC_API_KEY=` in .env means "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "database_path", "registry_path", "rubric_path", "judge_prompt_path", "tool_suites_dir"
    )
    @classmethod
    def _resolve_database_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("ollama_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def evaluator_configured(self) -> bool:
        return self.anthropic_api_key is not None


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process. Tests override this dependency."""
    return Settings()
