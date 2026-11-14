"""Application configuration.

Settings come from, in order of precedence: real environment variables, the
`.env` file in the project root, then the defaults below. Every variable is
prefixed with `GATEWAY_` so it cannot collide with unrelated variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
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
    # SecretStr keeps the key out of repr(), logs and JSON dumps by default.
    anthropic_api_key: SecretStr | None = None

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        # `GATEWAY_ANTHROPIC_API_KEY=` in .env means "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_path")
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
