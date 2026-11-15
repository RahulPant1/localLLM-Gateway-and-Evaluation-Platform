"""Short-lived cache of the provider's model list."""

import time

from llm_gateway.providers.base import ModelInfo, ModelNotFound, ModelProvider


class ModelCatalog:
    def __init__(self, provider: ModelProvider, ttl_seconds: float = 10.0) -> None:
        self._provider = provider
        self._ttl = ttl_seconds
        self._models: list[ModelInfo] | None = None
        self._fetched_at = 0.0

    async def list(self, *, refresh: bool = False) -> list[ModelInfo]:
        stale = time.monotonic() - self._fetched_at > self._ttl
        if refresh or stale or self._models is None:
            # On failure the exception propagates and the old list is dropped:
            # showing stale models while the provider is down would mislead.
            self._models = None
            self._models = await self._provider.list_models()
            self._fetched_at = time.monotonic()
        return self._models

    async def get(self, name: str) -> ModelInfo:
        """Find a model by exact name, re-reading the provider once if it is missing."""
        for refresh in (False, True):
            for model in await self.list(refresh=refresh):
                if model.name == name:
                    return model
        raise ModelNotFound(f"Model '{name}' is not installed")
