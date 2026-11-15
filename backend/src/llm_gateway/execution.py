"""Runs generations one at a time.

The GPU holds a single model, so two concurrent generations on different
models would make Ollama evict and reload them repeatedly. A lock serialises
all generations; asyncio.Lock wakes waiters in arrival order, so it is a FIFO
queue.
"""

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from llm_gateway.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ModelProvider,
    OnDelta,
)


class ExecutionRecord(BaseModel):
    result: GenerationResult
    started_at: datetime
    # Time spent waiting for earlier generations, kept apart from the model's own time.
    queue_wait_ms: float
    # Wall-clock time of the provider call, as measured by the gateway.
    execution_ms: float


class QueueState(BaseModel):
    busy: bool
    running_model: str | None
    waiting: int


class ExecutionQueue:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._running_model: str | None = None

    def state(self) -> QueueState:
        return QueueState(
            busy=self._lock.locked(), running_model=self._running_model, waiting=self._waiting
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
        )
