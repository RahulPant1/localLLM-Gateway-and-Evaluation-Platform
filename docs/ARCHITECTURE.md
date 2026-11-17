# Architecture and Agreed Decisions

Status: architecture approved on 2026-11-14. Phases 0 to 3 are complete (2026-11-15), plus judging by an external judge (Evaluate tab) and the Overview tab. The in-app Claude API judge (rest of Phase 4) is not built; judging is done by an external judge instead. How to run and use the app: [USAGE.md](USAGE.md).

This document records what was found in the environment, the architecture that was agreed, and the reasons. Requirements are in [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md).

## Working agreement

- Build iteratively, one phase at a time. Each phase ends with passing tests and a short review before the next begins.
- Record design decisions and the reasons for them, not only the code.
- Do not reinstall Ollama, download models, or change the Ollama service or `.wslconfig` without asking.

## Environment (inspected 2026-11-14)

| Area | Finding |
|---|---|
| Host | Windows 11, i7-13700HX (24 threads), 15.6 GB RAM |
| WSL | Ubuntu 22.04, WSL2, systemd, mirrored networking (Windows reaches WSL services on `localhost`) |
| WSL limits | `.wslconfig`: 6 GB RAM, 4 CPUs, no swap |
| GPU | RTX 4060 Laptop, 8188 MiB VRAM, about 7.5 GB free at idle |
| Ollama | v0.40.3, systemd service as user `ollama`, bound to `127.0.0.1:11434`, default settings |
| Tooling in WSL | Python 3.10.12, `uv` 0.8.3, git, Node via nvm (16, 18, 20, 23; only on PATH in interactive shells) |
| Claude API key | Not configured anywhere yet |

### Models visible to the Ollama service

| Model | Size | Quant | Capabilities | Intended role |
|---|---|---|---|---|
| `qwen3.5:4b` | 3.3 GB | Q4_K_M | tools, vision, thinking | General Q&A, classification, extraction, structured output; also the router's classifier |
| `qwen2.5-coder:7b` | 4.7 GB | Q4_K_M | tools, insert | Coding and code review |
| `deepseek-r1:8b` | 4.9 GB | Q4_K_M | tools, thinking | Reasoning and analysis |
| `MedAIBase/MedGemma1.5:4b-it-q8_0` | 4.1 GB | Q8_0 | completion only, `num_ctx` 4096 | Medical documents |

Notes:

- Llama 3.2 3B is listed in the requirements but is not installed. The app discovers models dynamically, so it will appear if it is installed later.
- A second model store exists at `~/.ollama/models` (12 GB: `mistral`, `biomistral-7b`, a duplicate MedGemma). The Ollama service does not read it. Leave it alone.
- No two models fit in VRAM together. Every change of model is a swap costing several seconds.

## Architecture

```
Browser (Windows) --> FastAPI backend (WSL) --> Ollama (WSL, 127.0.0.1:11434)
                          |   |
                          |   +--> Claude API (evaluation only, only if a key is set)
                          +--> SQLite (requests, routing decisions, executions, evaluations)
```

| Component | Choice | Reason |
|---|---|---|
| Backend | Python + FastAPI, run inside WSL, managed with `uv` | Sits beside Ollama; async suits slow model calls; Pydantic gives one schema for routing decisions, evaluations and API validation |
| Storage | SQLite | Single user, local, no extra service |
| Model access | A small provider interface with an Ollama implementation | Lets another provider be added later without redesign |
| GPU use | One generation at a time through an execution queue; comparisons run sequentially, ordered to reuse the loaded model | 8 GB VRAM holds one model |
| Model knowledge | A YAML registry mapping models to task categories, context size and strengths | New models appear automatically as "unprofiled" and can be used manually |
| Frontend | React + Vite + TypeScript single-page app over the JSON API | Side-by-side comparison and live status are awkward in server-rendered pages; Streamlit was rejected because it fights custom layouts and background evaluation |

### Data model

Separate tables so that a failure in one stage never alters another stage's record:

- `requests`: the prompt, optional source material, routing mode.
- `routing_decisions`: category, complexity, selected model, fallback, confidence, explanation, escalation flag.
- `executions`: model, response, thinking text (stored separately), timestamps, duration, status, error. One request can have many executions (comparison).
- `evaluations`: judge model, rubric version, dimension scores, overall score, summary, strengths, weaknesses, recommendations, critical issues, status.
- `human_reviews`: a person's annotation or score override; the original evaluation is kept.

### Routing

Two stages:

1. Classify. Deterministic signals first (code fences, medical vocabulary, extraction or classification phrasing, length). Only when these are ambiguous, ask `qwen3.5:4b` for a classification constrained to a JSON schema.
2. Decide. A deterministic policy table picks the model and fallback from category, complexity, availability and which model is already loaded.

Reasons: classifying every request with an LLM forces a model swap before the real work starts, and a policy table is easy to inspect and test.

Routing confidence is derived from agreement between signals, not from the model's self-report. It is stored and later compared against evaluation scores in a report. Routing policy is never changed automatically.

"Escalate" is a recommendation flag only ("a stronger local model is advised" or "likely beyond local models"). The app never sends a request to the cloud for answering.

### Claude evaluation

- Uses the Claude API with its own key and per-token billing. This is separate from the Claude Code subscription used for development.
- Key lives in a gitignored `.env` in WSL, read only by the backend, never sent to the browser, redacted from logs.
- Default judge model: `claude-opus-5-5`, set in configuration. Every evaluation records the judge model and rubric version.
- The API's JSON-schema structured output guarantees the evaluation shape. A refusal or error is recorded as "evaluation failed".
- The judge receives the request, the response, optional source material and the rubric. The responding model's name and thinking text are removed. Each response is scored on its own, not against others.
- Triggered on demand per response, with an optional auto-evaluate toggle (cost and privacy).
- Dimensions scored 0 to 10: correctness, relevance, completeness, instruction adherence, groundedness. Groundedness is "not applicable" when no source material is supplied.
- The overall score is computed by the app as a weighted mean of the dimension scores, with configurable weights, so it is reproducible.
- Evaluation status is tracked separately from execution status: not requested, pending, completed, failed, evaluator unavailable.
- Without a key, routing, execution, history and comparison all still work.

## Phases

| Phase | Delivers | Tested by |
|---|---|---|
| 0. Skeleton | git, `uv` project, configuration, test harness, health endpoint | Health endpoint test |
| 1. Ollama gateway | Model discovery, execution queue, latency capture, system status (Ollama, GPU, loaded model); measure model swap time | Mocked-Ollama unit tests and a live smoke test |
| 2. Manual execution and history | Persistence, manual model selection, history API, first UI screen | API tests; model-unavailable test |
| 3. Router | Registry, classifier, policy, structured decision, fallback | Labelled prompt set; invalid-decision tests |
| 4. Evaluator | Rubric, Claude integration, statuses, human review | Fake-judge tests; live test once a key exists |
| 5. Comparison and analysis | Same prompt across models, per-category aggregates | End-to-end test with fakes |
| 6. Hardening and docs | Router confidence report, install and run instructions | Acceptance criteria in section 11 of the requirements |

The UI grows with each phase from Phase 2. The Claude API key is not needed before Phase 4.

## Phase 1 findings (measured 2026-11-15)

Full table in [SWAP_TIMES.md](SWAP_TIMES.md).

- Swapping to a different model costs about 5 to 7 seconds once model files are in the operating system's file cache, regardless of which model.
- A first load from disk is slower: 19 seconds for `deepseek-r1:8b` from an empty GPU, 15 seconds for MedGemma.
- A request to the already-loaded model has no measurable load cost (under 0.2 seconds total for a tiny prompt).
- Thinking dominates latency on simple prompts: a warm `qwen3.5:4b` took 22.7 seconds on a one-line question because it produced 1,424 thinking tokens. With thinking off the same model answers a tiny prompt in under 0.1 seconds.

Consequences for the router (Phase 3):

- The two-stage design holds. An LLM classification costs a swap to `qwen3.5:4b` plus a swap to the chosen model, about 10 to 12 seconds of overhead, so deterministic signals should decide whenever they can.
- The classifier call must run with thinking off.
- Preferring the already-loaded model when it is adequate saves about 6 seconds per request.

## Phase 2 additions (2026-11-15)

Phase 2 went beyond the original table: batch evaluation (many prompts across all models, stored for later judging) is the main workflow.

- **Batches.** `POST /api/batches` runs many prompts on many models (all installed models by default) and stores every output. Executions are grouped by model, with the loaded model first.
- **Export for judges.** `GET /api/batches/{id}/export` returns completed outputs keyed by `execution_id`. Blind by default: no model names, no thinking text.
- **Evaluation import.** `POST /api/evaluations` stores results from any judge: a model, Claude Code, or a person. The `evaluations` table and the app-computed overall score therefore already exist; Phase 4 adds the in-app Claude API judge on the same table.
- **Background execution with polling.** A request returns at once; the UI polls for status. Streaming was deferred.
- **Storage.** Python's built-in `sqlite3`, all SQL in `store.py`, schema versions tracked with `PRAGMA user_version`.
- **Observed:** `deepseek-r1:8b` thinks even when asked not to, so a low output-token limit can leave its answer empty.
- **Memory:** the Vite dev server uses about 260 MB and the backend about 50 MB.

## Follow-up chat (added 2026-11-15, before Phase 3)

- **No conversation table.** A follow-up is an ordinary request whose `parent_execution_id` names the answer it continues from. Each turn keeps its own timings, status and evaluations. From a comparison, each model's answer can be continued separately.
- **History sent to the model.** Every earlier prompt and answer, plus the first turn's system prompt. Thinking text is never sent back.
- **Context limit: warn and stop.** Usage is estimated from the token counts Ollama reports. The UI warns at 70% of the model's context and the API refuses new messages at 90% (`context_full`), because Ollama silently drops old text once the window fills. Dropping or summarising old turns was deferred.
- **Same model throughout.** A follow-up is never re-routed.
- **After a failed turn**, the next message continues from the last completed answer.
- **Deferred:** streaming replies; including earlier turns in the judge export (needed when Phase 4 evaluates follow-up answers; batch exports are unaffected because batch prompts are first turns).
- Model output is rendered as Markdown in the UI; raw HTML in output is not rendered.

## Phase 3: router as built (2026-11-15)

- **Registry and policy** live in `config/models.yaml`: model strengths, the classifier model, and for each category an ordered list of models (optionally per complexity). Models in an inner list are equals; the loaded one is preferred. Unlisted models are "unprofiled": usable manually, never auto-selected except as a last resort.
- **Quick rules** (`routing/signals.py`): weighted regular-expression rules vote for a category; a separate set of cues estimates complexity. "Clarity" (0 to 1) measures how decisively one category wins.
- **Classifier model** is asked only when clarity is below 0.4 *and* the two leading categories would pick different models. It runs with thinking off and schema-constrained JSON, about 0.5 seconds when `qwen3.5:4b` is loaded, plus a model swap when it is not. An invalid answer or failure falls back to the rules with confidence capped at 0.5. `GATEWAY_ROUTER_LLM=off` disables it.
- **Confidence** comes from how the decision was reached, never from a model's self-report: 0.35 to 0.95 from rule clarity; 0.85 when the classifier agrees with the rules; 0.6 when the classifier decides alone. These are starting values to be checked against evaluation scores in Phase 6.
- **Escalation** is advice only. It is set for high complexity, when no suitable model is installed, when the input exceeds every candidate's context, or when nothing is installed.
- **Context fit:** a candidate whose context window cannot hold the input (75% budget) is passed over for one that can.
- **Fallback at run time:** if the selected model fails (not found, generation failed, timeout), the fallback model runs once; both attempts are kept. Not retried when Ollama itself is unreachable.
- **Automatic requests** are routed in the background: the record first has no executions and `routing: null`, then gains the decision and an execution. `POST /api/route` previews a decision without running anything.
- **Shadow routing:** every batch prompt gets a decision recorded with `shadow = true`. It does not choose the model; it lets Phase 6 compare the router's pick with the best-scoring model.
- **Follow-ups** are never re-routed.

Measured accuracy (2026-11-15):

| Set | What | Result |
|---|---|---|
| Tuning set, 66 prompts | Quick rules alone, right category | 65 of 66 |
| Held-out set, 25 prompts (rules never tuned on it) | Quick rules alone, right category | 18 of 25 |
| Held-out set | Full router with classifier, right category | 20 of 25 |
| Held-out set | Full router, right model | 23 of 25 |

On the held-out set the classifier was needed for 11 of 25 prompts. Both wrong-model cases were coding questions phrased as plain questions that went to the general model. The held-out file must stay untuned; add new prompts to `routing_prompts.yaml`.

## Evaluation by Claude Code (2026-11-15)

The first working judge is Claude Code itself, not the API, so no API key is needed.

- **Rubric** in `config/rubric.yaml`: version, the five fixed dimensions with descriptions and score anchors, and weights. The overall score is the weighted mean of the scored dimensions, computed by the gateway. Each stored evaluation records the rubric version.
- **Judge instructions** in `config/judge_prompt.md`: how to judge and the result format. Written to be reused unchanged as the system prompt of the API judge later.
- **Export** now carries the rubric and instructions with the items, and can leave out outputs a named judge has already scored.
- **`backend/scripts/evals.py`** (`list`, `export`, `import`, `summary`) works directly on the database, so the backend need not be running.
- **`/evaluate-batch`** is a project command for Claude Code (`.claude/commands/evaluate-batch.md`) that runs export, judging, import and summary. The export is blind, and the command tells the judge not to look up model names until scores are stored.
- **Per-model score summary**: `GET /api/batches/{id}/scores`, shown under the batch grid.

Still to do when a key exists (original Phase 4): the API judge call with structured output, evaluation statuses (pending, failed, unavailable), cost tracking, the Evaluate buttons, and human review.

Caveat: Claude Code judging is only as blind as the judge's discipline. The same session could read the database. For a stricter separation, run `/evaluate-batch` in a fresh Claude Code session that has not seen the batch results.

## Evaluate and Overview tabs (2026-11-15)

- **Evaluate tab** replaces the panel that sat inside a batch. `POST /api/judge-package` builds the judge text for any mix of history requests and batches; `GET /api/judge-prompt` returns the instructions. Pasted scores go through the existing `POST /api/evaluations`.
- **Overview tab** is read-only. `GET /api/overview?days=N` returns routing, performance, per-model and per-category rows, and evaluation totals in one response, computed in `overview.py` from `store.requests_since()`. No new tables.
- A "warm" answer is one whose model load took under 1 second (`WARM_LOAD_MS`).
- The **router check** compares the router's pick with judge scores on prompts where the pick and at least one other model were scored. It also reports the best single model on the same prompts. This is the first piece of the Phase 6 router confidence report.
- The cloud cost figure is an estimate from token counts and configurable prices; local output tokens include thinking tokens.
- Charts are plain CSS bars; no chart library was added.
- **Task type and domain** (migration 4: two nullable columns on `evaluations`). The judge classifies each prompt from fixed lists in `taxonomy.py`; the lists are appended to the judge instructions by `load_judge_instructions`. They are separate from the router's seven categories: the router guesses before the answer exists, the judge labels afterwards with a finer list. A prompt's label is the one most of its evaluations carry. Unknown labels are stored as `other`.
- The Evaluate tab lists only items with unscored outputs; fully scored ones return through "Evaluate again".

## OpenAI-compatible endpoint (2026-11-15)

- `openai_api.py`: `POST /v1/chat/completions` and `GET /v1/models`. A call becomes an ordinary stored request (`source = 'api'`), run by the same Runner and queue, so the Playground list, evaluation and the Overview cover it with no extra code.
- `openai_tools.py`: the function-calling path of `/v1`. A request with `tools` (or tool messages) is converted to Ollama's message format, sent to a tool-capable model through the queue's `run_tool_turn`, and stored as an ordinary request whose answer lists the calls. With `auto`, a routed model without tool support is swapped for the next one that has it. Calls a model wrote as JSON text are converted when they name an offered function.
- `tooluse/`: single-turn function-calling tests. `checker.py` scores calls against BFCL's accepted answers; `runner.py` runs each (case, model) pair through the queue's `run_tool_turn`.
- `agents/`: multi-step agent tests. `world.py` holds the simulated files, mailbox and API, and records every side effect; `checks.py` scores the final state; `runner.py` runs the loop (model turn, tool calls, results back) per (scenario, model). Tool-use and agent runs share one storage shape in `store.py` (`kind="tool"` or `"agent"`).
- `connectors/`: real apps for agents, all read-only. `files.py` (one folder, path checks against escaping it), `gmail.py` (Gmail REST API with the `gmail.readonly` scope; refreshes its token with the saved refresh token, no Google libraries), `web.py` (fixed tools over Open-Meteo, Frankfurter and Wikipedia). `live.py` runs a Playground agent with the shared loop in `agents/loop.py` and stores it as an ordinary request (`source = 'agent'`); the steps live in the execution's metrics and are rewritten after each step, so polling shows progress. `scripts/gmail_auth.py` does the one-time OAuth sign-in (loopback redirect with PKCE).
- `lineup.py`: the Models view: coverage per model, **Run all tests**, and hiding. Hidden models are filtered where rows are read in `store.py` and in `ModelCatalog`, so every screen, export and route leaves them out without deleting anything.
- Long jobs (a batch, a tool-use run, an agent run) hold `ExecutionQueue.long_jobs` for their whole length, so two of them never alternate models generation by generation; single prompts only take the per-generation lock and still get a turn.
- API clients send the whole conversation each time, so the earlier turns are stored on the request (`requests.history`, migration 5) instead of through `parent_execution_id`. Migration 5 also adds `source` and `temperature`.
- **Streaming:** `ModelProvider.generate` takes an optional `on_delta` callback; with it the Ollama provider reads `/api/chat` line by line. The Runner publishes `start`, `delta` and `end` events to a per-request queue that the HTTP handler turns into server-sent events. Requests made in the app poll for the growing text (next point).
- **Live text in the app:** the Runner always streams from the provider and keeps the text so far in memory (`Runner._live`). `GET /api/requests/{id}`, the thread and the batch detail put it on executions that are still running, and the screens poll every 0.5 s. Polling was kept instead of a browser event stream because it survives a page refresh and needed no new frontend plumbing; the cost is text arriving in half-second steps. Partial text is never written to the database.
- **`thinking_exhausted`:** an empty answer with reasoning present, where generation stopped at the token limit or the context filled, is stored as a failed execution (reasoning, timings and token counts kept). It does not trigger the routing fallback.
- **Second attempts:** `POST /api/executions/{id}/retry` adds another execution to the same request with `executions.think = 0` (migration 6), which overrides the request's thinking setting for that attempt. The request therefore keeps both attempts, stays in its batch, and exports (completed executions only) carry the answer that succeeded. The batch grid shows the latest attempt per model. A follow-up message is the exception: it is retried as a new request with the same parent, so it takes the failed message's place in the thread.
- A streaming client that disconnects cancels the job, which closes the Ollama connection and frees the GPU. A non-streaming call runs to completion.
- If the routed model fails after sending part of an answer and the fallback model runs, the stream contains both parts. Ollama failures normally happen before the first piece.
- No authentication: uvicorn binds to 127.0.0.1.

## Guardrails after the WSL stall (2026-11-15)

A 112-run batch stalled WSL for 21 minutes after 13 prompts on one model. Cause, from the Ollama and kernel logs: Ollama's per-model prompt cache grew by about 120 MB per prompt (1.8 GB after 13; its limit is 8 GB, WSL has 5.8 GB and no swap), the kernel reported page allocation failures, and nothing in WSL could make progress, including the gateway's own 300 s timeout.

- **`MemoryGuard`** (`execution.py`): checked inside the queue before every generation. Below `min_free_memory_mb` it unloads loaded models through the provider (`unload`, Ollama `keep_alive: 0`), waits for memory to return, and raises `LowMemory` (`low_memory`, 503) if memory is still under a third of the limit. The memory reading is stored in each execution's metrics.
- **Database lock** (`_SafeConnection` in `store.py`): the one SQLite connection is used by the event loop and by worker threads serving read requests. Concurrent use corrupted a cursor once ("tuple index out of range", a 500). Every statement now runs and is read fully under a lock, and a transaction holds the lock throughout.
- **Resume**: `POST /api/batches/{id}/resume` requeues executions that failed with `interrupted`, `generation_timeout`, `low_memory` or `provider_unavailable`.
- `nvidia-smi` timing out no longer makes `/api/status` fail.
- Not changed: Ollama's own cache limit and the WSL memory or swap settings. Both would address the cause more directly; they are changes outside the app.

## Known risks

- Model swap latency dominates comparisons and LLM-classified routes. Phase 1 measures it before the router design is final.
- WSL has 6 GB RAM and no swap. Ollama, the backend and a dev server together may run out of memory. Raising the cap is a change outside the app.
- Ollama truncates over-long prompts silently. Set explicit context sizes per model and warn when input exceeds them.
- `deepseek-r1` and `qwen3.5` emit thinking text. Store it separately from the answer and exclude it from evaluation.
- Evaluation sends the request and response to Anthropic. Do not use real patient data with MedGemma when evaluation is on.
- Judge scores are a measurement, not ground truth; hence rubric versioning and human override.

## Project location

The project lives in the WSL filesystem, not on the Windows drive. Reason: SQLite locking and Node file-watching are slow and unreliable across the Windows/WSL file boundary.

## Judge model options

| Model | Input / output per million tokens | Note |
|---|---|---|
| `claude-opus-5-5` | $4 / $20 | Agreed default; roughly a few cents per evaluation |
| `claude-sonnet-5-5` | $2 / $10 | Cheaper option |
| `claude-haiku-5-5` | $0.10 / $0.50 | Cheapest; suited to bulk benchmark runs |

Prices as of November 2026; check current pricing before Phase 4. An API key comes from the Claude Console and needs prepaid credits. It is separate from the Claude Pro subscription.
