"""Runs a tool-calling test run in the background and scores each answer."""

import asyncio
import logging

from llm_gateway.catalog import ModelCatalog
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import ModelProvider, ProviderError
from llm_gateway.store import Store
from llm_gateway.tooluse.checker import extract_text_calls, score
from llm_gateway.tooluse.suites import ToolSuite

logger = logging.getLogger(__name__)

# Enough for a call or two. Models that can think get more room even when thinking
# is switched off, because some (deepseek-r1) cannot actually turn it off and would
# otherwise be cut off before answering.
MAX_TOKENS = 1024
MAX_TOKENS_THINKING = 4096


class ToolRunner:
    def __init__(
        self, store: Store, provider: ModelProvider, catalog: ModelCatalog, queue: ExecutionQueue
    ) -> None:
        self._store = store
        self._provider = provider
        self._catalog = catalog
        self._queue = queue
        self._tasks: dict[str, asyncio.Task] = {}

    def active(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def schedule(self, run_id: str, suite: ToolSuite) -> None:
        if self.active(run_id):
            return
        self._tasks[run_id] = asyncio.create_task(self._run(run_id, suite))

    async def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _run(self, run_id: str, suite: ToolSuite) -> None:
        run = self._store.tool_run(run_id)
        if run is None:
            return
        cases = {case.id: case for case in suite.cases}
        await self._queue.long_jobs.acquire()
        try:
            # Cases queued while the run is going (a model added to it) are
            # picked up by the next pass.
            while pending := self._store.tool_results(run_id, status="queued"):
                run = self._store.tool_run(run_id)
                # One model at a time, so each model is loaded once.
                try:
                    loaded = {m.name for m in await self._provider.list_loaded()}
                except ProviderError:
                    loaded = set()
                order = {m: i for i, m in enumerate(run["models"])}
                pending.sort(
                    key=lambda r: (
                        r["model"] not in loaded,
                        order.get(r["model"], 99),
                        r["position"],
                    )
                )
                for result in pending:
                    await self._run_one(run, result, cases[result["case_id"]])
        except asyncio.CancelledError:
            self._store.fail_unfinished()
            raise
        finally:
            self._queue.long_jobs.release()

    async def _run_one(self, run: dict, result: dict, case) -> None:
        self._store.mark_tool_running(result["id"])
        try:
            model = await self._catalog.get(result["model"])
            turn, queue_wait_ms, execution_ms = await self._queue.run_tool_turn(
                [{"role": "user", "content": case.prompt}],
                case.tools,
                model,
                think=run["think"],
                max_tokens=MAX_TOKENS_THINKING if model.supports_thinking else MAX_TOKENS,
                temperature=0,
            )
        except ProviderError as exc:
            self._store.fail_tool_result(result["id"], exc.code, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
            logger.exception("Tool case %s crashed", case.id)
            self._store.fail_tool_result(result["id"], "internal_error", str(exc))
            return
        calls, call_format = turn.tool_calls, "native" if turn.tool_calls else "none"
        if not calls:
            # Some models write the call as JSON in their answer. It is scored,
            # but recorded as "text" so the shortfall stays visible.
            calls = extract_text_calls(turn.content)
            call_format = "text" if calls else "none"
        verdict = score(case.expected, calls, case.tools)
        reason = verdict.reason
        if not verdict.passed and turn.done_reason == "length":
            reason += " (the answer hit the token limit before it finished)"
        self._store.finish_tool_result(
            result["id"],
            calls=[c.model_dump() for c in calls],
            call_format=call_format,
            content=turn.content,
            thinking=turn.thinking,
            passed=verdict.passed,
            reason=reason,
            prompt_tokens=turn.prompt_tokens,
            output_tokens=turn.output_tokens,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
        )
