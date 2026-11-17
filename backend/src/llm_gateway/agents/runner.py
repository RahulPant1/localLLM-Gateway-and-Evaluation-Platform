"""Runs agent scenarios: the model calls tools, the simulated apps answer,
until the model gives a final answer or a limit is reached. Then the checks
score the final state.

Each model turn goes through the shared queue on its own, and the whole run
holds the queue's long-job lock, so a batch or tool-use run does not swap
models in between.
"""

import asyncio
import logging
import time

from llm_gateway.agents.checks import evaluate
from llm_gateway.agents.suites import AgentSuite, Scenario
from llm_gateway.agents.world import World, tool_schemas
from llm_gateway.catalog import ModelCatalog
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import ModelProvider, ProviderError
from llm_gateway.store import Store
from llm_gateway.tooluse.checker import extract_text_calls

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024
MAX_TOKENS_THINKING = 4096
# The same call with the same arguments this many times in a row ends the run.
REPEAT_LIMIT = 3


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
        offered = {t["function"]["name"] for t in tools}
        messages: list[dict] = [
            {"role": "system", "content": suite.system_prompt},
            {"role": "user", "content": scenario.prompt},
        ]
        trace: list[dict] = []
        totals = {"tool_calls": 0, "invalid_calls": 0, "text_calls": 0}
        tokens = {"prompt": 0, "output": 0}
        waited = 0.0
        answer, stop = "", "step_limit"
        last_call, repeats = None, 0
        started = time.perf_counter()
        try:
            model = await self._catalog.get(result["model"])
            for step in range(scenario.max_steps):
                turn, queue_wait_ms, execution_ms = await self._queue.run_tool_turn(
                    messages,
                    tools,
                    model,
                    think=run["think"],
                    max_tokens=MAX_TOKENS_THINKING if model.supports_thinking else MAX_TOKENS,
                    temperature=0,
                )
                waited += queue_wait_ms
                tokens["prompt"] += turn.prompt_tokens or 0
                tokens["output"] += turn.output_tokens or 0
                calls, as_text = turn.tool_calls, False
                if not calls:
                    found = extract_text_calls(turn.content)
                    if found and all(c.name in offered for c in found):
                        calls, as_text = found, True
                entry: dict = {
                    "step": step + 1,
                    "content": "" if as_text else turn.content,
                    "thought": bool(turn.thinking),
                    "calls": [],
                    "ms": execution_ms,
                    "calls_as_text": as_text,
                }
                trace.append(entry)
                if not calls:
                    answer = turn.content
                    stop = "cut_off" if turn.done_reason == "length" else "answered"
                    break
                if as_text:
                    totals["text_calls"] += len(calls)
                messages.append(
                    {
                        "role": "assistant",
                        "content": entry["content"],
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                        ],
                    }
                )
                looping = False
                for call in calls:
                    outcome = world.call(call.name, call.arguments)
                    totals["tool_calls"] += 1
                    totals["invalid_calls"] += outcome.invalid
                    entry["calls"].append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "result": outcome.text,
                            "invalid": outcome.invalid,
                            "error": outcome.error,
                        }
                    )
                    messages.append(
                        {"role": "tool", "content": outcome.text, "tool_name": call.name}
                    )
                    key = (call.name, repr(sorted(call.arguments.items())))
                    repeats = repeats + 1 if key == last_call else 1
                    last_call = key
                    looping = looping or repeats >= REPEAT_LIMIT
                if looping:
                    stop = "loop"
                    break
        except ProviderError as exc:
            self._store.fail_tool_result(result["id"], exc.code, str(exc), kind="agent")
            return
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the run
            logger.exception("Agent scenario %s crashed", scenario.id)
            self._store.fail_tool_result(result["id"], "internal_error", str(exc), kind="agent")
            return

        checks = [evaluate(check, world, answer) for check in scenario.checks]
        passed = sum(c.passed for c in checks)
        effects = {k: v for k, v in world.effects.items() if k != "calls" and v}
        self._store.finish_agent_result(
            result["id"],
            passed=passed == len(checks),
            score=round(passed / len(checks), 3) if checks else 0.0,
            checks=[c.__dict__ for c in checks],
            trace=trace,
            effects=effects,
            answer=answer,
            stop_reason=stop,
            steps=len(trace),
            tool_calls=totals["tool_calls"],
            invalid_calls=totals["invalid_calls"],
            text_calls=totals["text_calls"],
            prompt_tokens=tokens["prompt"],
            output_tokens=tokens["output"],
            queue_wait_ms=round(waited, 1),
            execution_ms=round((time.perf_counter() - started) * 1000 - waited, 1),
        )
