"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx2
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_gateway import __version__
from llm_gateway.api import router as runs_router
from llm_gateway.catalog import ModelCatalog
from llm_gateway.config import PROJECT_ROOT, Settings, get_settings
from llm_gateway.execution import ExecutionQueue, QueueState
from llm_gateway.providers.base import (
    GenerationFailed,
    GenerationTimeout,
    LoadedModel,
    ModelInfo,
    ModelNotFound,
    ModelProvider,
    ProviderError,
    ProviderUnavailable,
)
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.runner import Runner
from llm_gateway.store import Store
from llm_gateway.system import Environment, GpuStatus, detect_environment, gpu_status


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: Environment
    ollama_url: str
    judge_model: str
    # Whether a key is set; the key itself is never returned.
    evaluator_configured: bool


class ModelListEntry(ModelInfo):
    loaded: bool


class ModelListResponse(BaseModel):
    provider: str
    models: list[ModelListEntry]


class ProviderStatus(BaseModel):
    name: str
    url: str
    reachable: bool
    version: str | None = None
    error: str | None = None
    loaded_models: list[LoadedModel] = []


class EvaluatorStatus(BaseModel):
    configured: bool
    judge_model: str


class StatusResponse(BaseModel):
    # "degraded" = the backend is up but local models cannot be run.
    status: Literal["ok", "degraded"]
    version: str
    environment: Environment
    provider: ProviderStatus
    gpu: GpuStatus
    queue: QueueState
    evaluator: EvaluatorStatus


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

ERROR_STATUS: dict[type[ProviderError], int] = {
    ModelNotFound: 404,
    GenerationTimeout: 504,
    GenerationFailed: 502,
    ProviderUnavailable: 503,
}


def get_provider(request: Request) -> ModelProvider:
    return request.app.state.provider


def get_catalog(request: Request) -> ModelCatalog:
    return request.app.state.catalog


def get_queue(request: Request) -> ExecutionQueue:
    return request.app.state.queue


SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[ModelProvider, Depends(get_provider)]
CatalogDep = Annotated[ModelCatalog, Depends(get_catalog)]
QueueDep = Annotated[ExecutionQueue, Depends(get_queue)]


def create_app(provider: ModelProvider | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass a fake `provider` and explicit `settings`."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings or get_settings()
        client = None
        active = provider
        if active is None:
            client = httpx2.AsyncClient(base_url=config.ollama_url)
            active = OllamaProvider(
                client,
                default_num_ctx=config.default_num_ctx,
                generation_timeout_seconds=config.generation_timeout_seconds,
            )
        store = Store(config.database_path)
        # Anything still queued or running belongs to a process that has gone.
        store.fail_unfinished()
        app.state.provider = active
        app.state.catalog = ModelCatalog(active)
        app.state.queue = ExecutionQueue(active)
        app.state.store = store
        app.state.runner = Runner(store, active, app.state.catalog, app.state.queue)
        yield
        await app.state.runner.shutdown()
        store.close()
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Local LLM Gateway", version=__version__, lifespan=lifespan)
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(runs_router)

    @app.exception_handler(ProviderError)
    async def provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        status = next((s for cls, s in ERROR_STATUS.items() if isinstance(exc, cls)), 500)
        return JSONResponse(status_code=status, content={"error": exc.code, "detail": str(exc)})

    @app.get("/api/health")
    def health(settings: SettingsDep) -> HealthResponse:
        """Report that the backend process is up and how it is configured.

        Does not contact Ollama; see /api/status for live connectivity.
        """
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=detect_environment(),
            ollama_url=settings.ollama_url,
            judge_model=settings.judge_model,
            evaluator_configured=settings.evaluator_configured,
        )

    @app.get("/api/models")
    async def models(
        provider: ProviderDep, catalog: CatalogDep, refresh: bool = False
    ) -> ModelListResponse:
        """Installed models, read live from the provider (cached a few seconds)."""
        installed, loaded = await asyncio.gather(
            catalog.list(refresh=refresh), provider.list_loaded()
        )
        loaded_names = {m.name for m in loaded}
        return ModelListResponse(
            provider=provider.name,
            models=[
                ModelListEntry(**m.model_dump(), loaded=m.name in loaded_names) for m in installed
            ],
        )

    @app.get("/api/status")
    async def status(
        settings: SettingsDep, provider: ProviderDep, queue: QueueDep
    ) -> StatusResponse:
        """Live system status. Each section degrades on its own."""

        async def provider_status() -> ProviderStatus:
            base = {"name": provider.name, "url": settings.ollama_url}
            try:
                version, loaded = await asyncio.gather(provider.version(), provider.list_loaded())
            except ProviderError as exc:
                return ProviderStatus(**base, reachable=False, error=str(exc))
            return ProviderStatus(**base, reachable=True, version=version, loaded_models=loaded)

        provider_info, gpu = await asyncio.gather(provider_status(), gpu_status())
        return StatusResponse(
            status="ok" if provider_info.reachable else "degraded",
            version=__version__,
            environment=detect_environment(),
            provider=provider_info,
            gpu=gpu,
            queue=queue.state(),
            evaluator=EvaluatorStatus(
                configured=settings.evaluator_configured, judge_model=settings.judge_model
            ),
        )

    # The built frontend, if present. Registered last so /api routes win.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()
