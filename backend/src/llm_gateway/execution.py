"""Runs generations one at a time.

The GPU holds a single model, so two concurrent generations on different
models would make Ollama evict and reload them repeatedly. A lock serialises
all generations; asyncio.Lock wakes waiters in arrival order, so it is a FIFO
queue.
"""

import asyncio
import time
from datetime import UTC, datetime

from pydantic import BaseModel

from llm_gateway.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ModelProvider,
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

    async def run(self, request: GenerationRequest, model: ModelInfo) -> ExecutionRecord:
        enqueued = time.perf_counter()
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        try:
            self._running_model = model.name
            started_at = datetime.now(UTC)
            start = time.perf_counter()
            result = await self._provider.generate(request, model)
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
