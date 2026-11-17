"""Short-lived cache of the provider's model list."""

import time
from collections.abc import Callable

from llm_gateway.providers.base import ModelInfo, ModelNotFound, ModelProvider


class ModelCatalog:
    def __init__(
        self,
        provider: ModelProvider,
        ttl_seconds: float = 10.0,
        hidden: Callable[[], set[str]] = set,
    ) -> None:
        self._provider = provider
        # Models taken out of the lineup: not listed, not routed to, not run.
        self._hidden = hidden
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
        hidden = self._hidden()
        return [m for m in self._models if m.name not in hidden]

    async def get(self, name: str) -> ModelInfo:
        """Find a model by exact name, re-reading the provider once if it is missing."""
        for refresh in (False, True):
            for model in await self.list(refresh=refresh):
                if model.name == name:
                    return model
        if name in self._hidden():
            raise ModelNotFound(f"Model '{name}' is hidden; restore it in the Models tab to use it")
        raise ModelNotFound(f"Model '{name}' is not installed")
