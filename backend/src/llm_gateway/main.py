"""FastAPI application entry point."""

import platform
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from llm_gateway import __version__
from llm_gateway.config import Settings, get_settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: Literal["wsl", "linux", "other"]
    ollama_url: str
    judge_model: str
    # Whether a key is set; the key itself is never returned.
    evaluator_configured: bool


def detect_environment() -> Literal["wsl", "linux", "other"]:
    if platform.system() != "Linux":
        return "other"
    return "wsl" if "microsoft" in platform.release().lower() else "linux"


def create_app() -> FastAPI:
    app = FastAPI(title="Local LLM Gateway", version=__version__)

    @app.get("/api/health")
    def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        """Report that the backend process is up and how it is configured.

        This does not contact Ollama or Claude; live connectivity is part of
        the system status endpoint added in Phase 1.
        """
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=detect_environment(),
            ollama_url=settings.ollama_url,
            judge_model=settings.judge_model,
            evaluator_configured=settings.evaluator_configured,
        )

    return app


app = create_app()
