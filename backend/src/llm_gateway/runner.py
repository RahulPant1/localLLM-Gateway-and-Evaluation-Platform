"""Runs stored executions in the background and records their outcome.

Submitting a request only writes database rows and schedules work here, so
the HTTP call returns at once and the result survives a page refresh.
"""

import asyncio
import logging
from collections.abc import Iterable

from llm_gateway.catalog import ModelCatalog
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import GenerationRequest, ModelProvider, ProviderError
from llm_gateway.schemas import Execution, RequestRecord
from llm_gateway.store import Store

logger = logging.getLogger(__name__)


def compose_prompt(request: RequestRecord) -> str:
    """The text sent to the model: the prompt, followed by any source material."""
    if not request.source_material:
        return request.prompt
    return f"{request.prompt}\n\nSource material:\n{request.source_material}"


class Runner:
    def __init__(
        self, store: Store, provider: ModelProvider, catalog: ModelCatalog, queue: ExecutionQueue
    ) -> None:
        self._store = store
        self._provider = provider
        self._catalog = catalog
        self._queue = queue
        self._tasks: set[asyncio.Task] = set()

    async def _loaded_names(self) -> set[str]:
        try:
            return {m.name for m in await self._provider.list_loaded()}
        except ProviderError:
            return set()

    async def schedule(self, requests: Iterable[RequestRecord]) -> None:
        """Queue every execution of the given requests as one background job.

        Work is grouped by model, so a batch of 10 prompts on 4 models costs
        4 model loads instead of 40. The model already in memory goes first.
        """
        pairs = [(e, r) for r in requests for e in r.executions if e.status == "queued"]
        loaded = await self._loaded_names()
        order: dict[str, int] = {}
        for execution, _ in pairs:
            order.setdefault(execution.model, len(order))
        pairs.sort(key=lambda pair: (pair[0].model not in loaded, order[pair[0].model]))
        task = asyncio.create_task(self._run_all(pairs))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_all(self, pairs: list[tuple[Execution, RequestRecord]]) -> None:
        for index, (execution, request) in enumerate(pairs):
            try:
                await self._run_one(execution, request)
            except asyncio.CancelledError:
                for remaining, _ in pairs[index:]:
                    self._store.fail_execution(
                        remaining.id, "interrupted", "The backend stopped before this finished"
                    )
                raise

    async def _run_one(self, execution: Execution, request: RequestRecord) -> None:
        """Run one execution. Any failure is recorded on that execution only."""
        try:
            model = await self._catalog.get(execution.model)
            record = await self._queue.run(
                GenerationRequest(
                    model=execution.model,
                    prompt=compose_prompt(request),
                    system=request.system,
                    think=request.think,
                    max_tokens=request.max_tokens,
                ),
                model,
                on_start=lambda: self._store.mark_running(execution.id),
            )
        except ProviderError as exc:
            self._store.fail_execution(execution.id, exc.code, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - one bad execution must not stop a batch
            logger.exception("Execution %s failed unexpectedly", execution.id)
            self._store.fail_execution(execution.id, "internal_error", f"{type(exc).__name__}")
            return
        result = record.result
        self._store.complete_execution(
            execution.id,
            response=result.response,
            thinking=result.thinking,
            queue_wait_ms=record.queue_wait_ms,
            execution_ms=record.execution_ms,
            metrics=result.model_dump(exclude={"model", "response", "thinking", "warnings"}),
            warnings=result.warnings,
        )

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
