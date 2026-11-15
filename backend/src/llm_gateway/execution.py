"""Runs generations one at a time.

The GPU holds a single model, so two concurrent generations on different
models would make Ollama evict and reload them repeatedly. A lock serialises
all generations; asyncio.Lock wakes waiters in arrival order, so it is a FIFO
queue.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from llm_gateway.providers.base import (
    GenerationRequest,
    GenerationResult,
    LowMemory,
    ModelInfo,
    ModelProvider,
    OnDelta,
    ProviderError,
)
from llm_gateway.system import memory_available_mb

logger = logging.getLogger(__name__)


class ExecutionRecord(BaseModel):
    result: GenerationResult
    started_at: datetime
    # Time spent waiting for earlier generations, kept apart from the model's own time.
    queue_wait_ms: float
    # Wall-clock time of the provider call, as measured by the gateway.
    execution_ms: float
    # Free system memory just before the generation started.
    memory_available_mb: int | None = None


class QueueState(BaseModel):
    busy: bool
    running_model: str | None
    waiting: int
    # Times a model was unloaded to give memory back (see MemoryGuard).
    memory_unloads: int = 0


class MemoryGuard:
    """Stops the machine running out of memory during long runs.

    Ollama keeps a cache of every prompt a loaded model has seen, in system
    memory, and it only shrinks when the model is unloaded. Many prompts in a
    row on one model can fill the memory of a small machine (or of WSL),
    which then stalls completely. Before each generation the guard checks
    free memory; when it is below `min_free_mb` it unloads the loaded models
    so their caches are released, and waits for the memory to come back. The
    next generation reloads the model, which costs a few seconds.
    """

    def __init__(
        self,
        provider: ModelProvider,
        min_free_mb: int,
        *,
        read_available=None,
        settle_seconds: float = 15.0,
    ) -> None:
        self._provider = provider
        self.min_free_mb = min_free_mb
        # Below this even after unloading, a generation is refused.
        self.critical_mb = min_free_mb // 3
        # Looked up at call time so the reading can be replaced in tests.
        self._read = read_available or (lambda: memory_available_mb())
        self._settle_seconds = settle_seconds
        self.unloads = 0

    async def check(self) -> int | None:
        """Free memory in MB once it is safe to generate. Raises LowMemory if it is not."""
        available = self._read()
        if not self.min_free_mb or available is None or available >= self.min_free_mb:
            return available
        try:
            loaded = [m.name for m in await self._provider.list_loaded()]
            for name in loaded:
                await self._provider.unload(name)
        except ProviderError:
            loaded = []
        if loaded:
            self.unloads += 1
            logger.warning(
                "Free memory was %s MB (limit %s MB): unloaded %s to release it",
                available,
                self.min_free_mb,
                ", ".join(loaded),
            )
            deadline = time.monotonic() + self._settle_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                available = self._read()
                if available is None or available >= self.min_free_mb:
                    break
        if available is not None and available < self.critical_mb:
            raise LowMemory(
                f"Only {available} MB of memory is free, even with no model loaded "
                f"(at least {self.critical_mb} MB is needed). Close other programs and try again."
            )
        return available


class ExecutionQueue:
    def __init__(self, provider: ModelProvider, memory_guard: MemoryGuard | None = None) -> None:
        self._provider = provider
        self._guard = memory_guard
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._running_model: str | None = None

    def state(self) -> QueueState:
        return QueueState(
            busy=self._lock.locked(),
            running_model=self._running_model,
            waiting=self._waiting,
            memory_unloads=self._guard.unloads if self._guard else 0,
        )

    async def run(
        self,
        request: GenerationRequest,
        model: ModelInfo,
        on_start: Callable[[], None] | None = None,
        on_delta: OnDelta | None = None,
    ) -> ExecutionRecord:
        """Wait for a turn, then generate. `on_start` is called when the turn
        begins; `on_delta` receives the answer piece by piece."""
        enqueued = time.perf_counter()
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        try:
            self._running_model = model.name
            free_mb = await self._guard.check() if self._guard else None
            if on_start:
                on_start()
            started_at = datetime.now(UTC)
            start = time.perf_counter()
            if on_delta is None:
                result = await self._provider.generate(request, model)
            else:
                result = await self._provider.generate(request, model, on_delta=on_delta)
            end = time.perf_counter()
        finally:
            self._running_model = None
            self._lock.release()
        return ExecutionRecord(
            result=result,
            started_at=started_at,
            queue_wait_ms=round((start - enqueued) * 1000, 1),
            execution_ms=round((end - start) * 1000, 1),
            memory_available_mb=free_mb,
        )
