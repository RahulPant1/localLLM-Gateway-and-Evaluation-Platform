"""Runs stored requests in the background and records their outcome.

Submitting a request only writes database rows and schedules work here, so
the HTTP call returns at once and the result survives a page refresh.
"""

import asyncio
import logging
from collections.abc import Iterable

from llm_gateway.catalog import ModelCatalog
from llm_gateway.conversation import compose_prompt, history_messages
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import GenerationRequest, ModelProvider, ProviderError
from llm_gateway.routing.router import Router
from llm_gateway.schemas import Execution, RequestRecord
from llm_gateway.store import Store

logger = logging.getLogger(__name__)

# Failures where trying the fallback model makes sense. An unreachable
# provider is not one: the fallback would fail the same way.
RETRY_ON_FALLBACK = {"model_not_found", "generation_failed", "generation_timeout"}


class Runner:
    def __init__(
        self,
        store: Store,
        provider: ModelProvider,
        catalog: ModelCatalog,
        queue: ExecutionQueue,
        router: Router,
    ) -> None:
        self._store = store
        self._provider = provider
        self._catalog = catalog
        self._queue = queue
        self._router = router
        self._tasks: set[asyncio.Task] = set()
        # Request id -> queue of live events, for callers that stream an answer.
        self._listeners: dict[str, asyncio.Queue] = {}

    def subscribe(self, request_id: str) -> asyncio.Queue:
        """Receive this request's live events: ("start", model),
        ("delta", kind, text) and finally ("end",). Subscribe before scheduling."""
        return self._listeners.setdefault(request_id, asyncio.Queue())

    def unsubscribe(self, request_id: str) -> None:
        self._listeners.pop(request_id, None)

    def _publish(self, request_id: str, *event: object) -> None:
        listener = self._listeners.get(request_id)
        if listener is not None:
            listener.put_nowait(event)

    async def _loaded_names(self) -> set[str]:
        try:
            return {m.name for m in await self._provider.list_loaded()}
        except ProviderError:
            return set()

    async def schedule(
        self, requests: Iterable[RequestRecord], *, shadow_route: bool = False
    ) -> asyncio.Task:
        """Run the given requests as one background job.

        `shadow_route` records what the router would have chosen for each
        request without letting it pick the model (used for batches).
        Cancelling the returned task stops the job and marks its unfinished
        work as failed.
        """
        task = asyncio.create_task(self._run_all(list(requests), shadow_route))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @staticmethod
    def _needs_routing(request: RequestRecord) -> bool:
        return (
            request.routing_mode == "automatic"
            and request.parent_execution_id is None
            and request.routing is None
        )

    async def _run_all(self, requests: list[RequestRecord], shadow_route: bool) -> None:
        try:
            pairs = await self._route(requests, shadow_route)
            # Group by model, so 10 prompts on 4 models cost 4 model loads, not 40.
            # The model already in memory goes first.
            loaded = await self._loaded_names()
            order: dict[str, int] = {}
            for execution, _ in pairs:
                order.setdefault(execution.model, len(order))
            pairs.sort(key=lambda pair: (pair[0].model not in loaded, order[pair[0].model]))
            pending = list(pairs)
            while pending:
                execution, request = pending[0]
                succeeded, code = await self._run_one(execution, request)
                pending.pop(0)
                # No await between a failure and queuing its fallback, so a client
                # never sees the request as finished in between.
                retry = self._fallback_for(request, execution, succeeded, code)
                if retry:
                    pending.insert(0, (retry, request))
        except asyncio.CancelledError:
            self._interrupt(requests)
            raise
        except Exception:
            logger.exception("Background run failed")
            self._interrupt(requests)
        finally:
            for request in requests:
                self._publish(request.id, "end")

    def _interrupt(self, requests: list[RequestRecord]) -> None:
        """Close whatever this job left unfinished."""
        self._store.fail_unrouted([r.id for r in requests])
        for request in requests:
            current = self._store.get_request(request.id)
            for execution in current.executions if current else []:
                if execution.status in ("queued", "running"):
                    self._store.fail_execution(
                        execution.id,
                        "interrupted",
                        "Stopped early: the backend shut down or the caller disconnected",
                    )

    async def _route(
        self, requests: list[RequestRecord], shadow_route: bool
    ) -> list[tuple[Execution, RequestRecord]]:
        """Decide models where needed and return every execution that is ready to run."""
        pairs: list[tuple[Execution, RequestRecord]] = []
        for request in requests:
            if self._needs_routing(request) or (shadow_route and request.routing is None):
                shadow = not self._needs_routing(request)
                # A conversation from an API client is routed by its opening
                # message, so it stays on one model instead of swapping per turn.
                opening = next((m.content for m in request.history if m.role == "user"), None)
                draft = await self._router.route(
                    opening or request.prompt, request.source_material, request.system
                )
                self._store.add_decision(request.id, draft, shadow=shadow)
                if not shadow and draft.selected_model:
                    self._store.add_execution(request.id, draft.selected_model)
                request = self._store.get_request(request.id)
            pairs.extend((e, request) for e in request.executions if e.status == "queued")
        return pairs

    def _fallback_for(
        self, request: RequestRecord, execution: Execution, succeeded: bool, code: str | None
    ) -> Execution | None:
        """After the routed model fails, queue the fallback model once."""
        routing = request.routing
        if (
            succeeded
            or routing is None
            or routing.shadow
            or routing.fallback_used
            or code not in RETRY_ON_FALLBACK
            or not routing.fallback_model
            or execution.model != routing.selected_model
        ):
            return None
        self._store.mark_fallback_used(request.id)
        routing.fallback_used = True
        return self._store.add_execution(request.id, routing.fallback_model)

    async def _run_one(
        self, execution: Execution, request: RequestRecord
    ) -> tuple[bool, str | None]:
        """Run one execution. Any failure is recorded on that execution only."""
        try:
            history = list(request.history)
            if request.parent_execution_id:
                history = history_messages(self._store.ancestors(request.parent_execution_id))
            model = await self._catalog.get(execution.model)
            on_delta = None
            if request.id in self._listeners:
                self._publish(request.id, "start", execution.model)

                def on_delta(kind: str, text: str) -> None:
                    self._publish(request.id, "delta", kind, text)

            record = await self._queue.run(
                GenerationRequest(
                    model=execution.model,
                    prompt=compose_prompt(request),
                    system=request.system,
                    history=history,
                    think=request.think,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ),
                model,
                on_start=lambda: self._store.mark_running(execution.id),
                on_delta=on_delta,
            )
        except ProviderError as exc:
            self._store.fail_execution(execution.id, exc.code, str(exc))
            return False, exc.code
        except Exception as exc:  # noqa: BLE001 - one bad execution must not stop a batch
            logger.exception("Execution %s failed unexpectedly", execution.id)
            self._store.fail_execution(execution.id, "internal_error", f"{type(exc).__name__}")
            return False, "internal_error"
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
        return True, None

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
