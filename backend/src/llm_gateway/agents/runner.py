"""Runs agent scenarios: the model calls tools, the simulated apps answer,
until the model gives a final answer or a limit is reached. Then the checks
score the final state.

Each model turn goes through the shared queue on its own, and the whole run
holds the queue's long-job lock, so a batch or tool-use run does not swap
models in between.
"""

import asyncio
import logging

from llm_gateway.agents.checks import evaluate
from llm_gateway.agents.loop import run_loop
from llm_gateway.agents.suites import AgentSuite, Scenario
from llm_gateway.agents.world import World, tool_schemas
from llm_gateway.catalog import ModelCatalog
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import ModelProvider, ProviderError
from llm_gateway.store import Store

logger = logging.getLogger(__name__)


class AgentRunner:
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

    def schedule(self, run_id: str, suite: AgentSuite) -> None:
        if self.active(run_id):
            return
        self._tasks[run_id] = asyncio.create_task(self._run(run_id, suite))

    async def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _run(self, run_id: str, suite: AgentSuite) -> None:
        scenarios = {s.id: s for s in suite.scenarios}
        await self._queue.long_jobs.acquire()
        try:
            while pending := self._store.tool_results(run_id, status="queued", kind="agent"):
                run = self._store.tool_run(run_id, kind="agent")
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
                    scenario = scenarios.get(result["case_id"])
                    if scenario is None:
                        self._store.fail_tool_result(
                            result["id"], "unknown_scenario", "Not in the suite", kind="agent"
                        )
                        continue
                    await self._run_one(run, result, scenario, suite)
        except asyncio.CancelledError:
            self._store.fail_unfinished()
            raise
        finally:
            self._queue.long_jobs.release()

    async def _run_one(
        self, run: dict, result: dict, scenario: Scenario, suite: AgentSuite
    ) -> None:
        self._store.mark_tool_running(result["id"], kind="agent")
        world = World(suite.world, scenario.world, scenario.apps)
        tools = tool_schemas(scenario.apps)
        messages: list[dict] = [
            {"role": "system", "content": suite.system_prompt},
            {"role": "user", "content": scenario.prompt},
        ]
        try:
            model = await self._catalog.get(result["model"])
            loop = await run_loop(
                self._queue,
                model,
                messages,
                tools,
                world.call,
                think=run["think"],
                max_steps=scenario.max_steps,
            )
        except ProviderError as exc:
            self._store.fail_tool_result(result["id"], exc.code, str(exc), kind="agent")
            return
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the run
            logger.exception("Agent scenario %s crashed", scenario.id)
            self._store.fail_tool_result(result["id"], "internal_error", str(exc), kind="agent")
            return

        checks = [evaluate(check, world, loop.answer) for check in scenario.checks]
        passed = sum(c.passed for c in checks)
        effects = {k: v for k, v in world.effects.items() if k != "calls" and v}
        self._store.finish_agent_result(
            result["id"],
            passed=passed == len(checks),
            score=round(passed / len(checks), 3) if checks else 0.0,
            checks=[c.__dict__ for c in checks],
            trace=loop.trace,
            effects=effects,
            answer=loop.answer,
            stop_reason=loop.stop,
            steps=len(loop.trace),
            tool_calls=loop.tool_calls,
            invalid_calls=loop.invalid_calls,
            text_calls=loop.text_calls,
            prompt_tokens=loop.prompt_tokens,
            output_tokens=loop.output_tokens,
            queue_wait_ms=loop.queue_wait_ms,
            execution_ms=loop.execution_ms,
        )
