# Architecture and Agreed Decisions

Status: architecture approved on 2026-11-14. Phases 0 and 1 are complete (2026-11-15). Next step is Phase 2.

This document records what was found in the environment, the architecture that was agreed, and the reasons. Requirements are in [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md).

## Working agreement

- Build iteratively, one phase at a time. Each phase ends with passing tests and a short review before the next begins.
- Explain design decisions as they come up; the owner wants to understand them, not only receive code.
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

- Llama 3.2 3B is listed in the requirements but is not installed. The app discovers models dynamically, so it will appear if the owner installs it later.
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
- `human_reviews`: the owner's annotation or score override; the original evaluation is kept.

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

## Known risks

- Model swap latency dominates comparisons and LLM-classified routes. Phase 1 measures it before the router design is final.
- WSL has 6 GB RAM and no swap. Ollama, the backend and a dev server together may run out of memory. Raising the cap is the owner's decision.
- Ollama truncates over-long prompts silently. Set explicit context sizes per model and warn when input exceeds them.
- `deepseek-r1` and `qwen3.5` emit thinking text. Store it separately from the answer and exclude it from evaluation.
- Evaluation sends the request and response to Anthropic. Do not use real patient data with MedGemma when evaluation is on.
- Judge scores are a measurement, not ground truth; hence rubric versioning and human override.

## Resuming in a new session

The architecture was planned in a Claude Code session on the Windows side, which is no longer attached to this project. Everything agreed there is in this file. To continue:

1. Open this folder in VS Code through the WSL extension (`WSL: Connect to WSL`, then open `/home/rahul/localLLM-Gateway-and-Evaluation-Platform`).
2. Start Claude Code and say: "Read DEVELOPMENT_PLAN.md and docs/ARCHITECTURE.md, then begin Phase 0."

What Phase 0 should do, in order:

1. `git init` and add a `.gitignore` (must cover `.env`, the virtualenv, the SQLite file, `node_modules`).
2. Create the `uv` project for the backend. Python 3.10.12 is the system version; `uv` can install a newer one for the project without touching the system.
3. Add configuration loading (Ollama URL, database path, judge model, optional API key) with a `.env.example`.
4. Add the test harness and a health endpoint with one passing test.
5. Stop and review with the owner before Phase 1.

Each later phase starts the same way: propose the detailed steps for that phase, get the owner's agreement, then build and test.

## Project location

The project lives in the WSL filesystem, not on the Windows drive. Reason: SQLite locking and Node file-watching are slow and unreliable across the Windows/WSL file boundary. The original Windows folder was removed after the move.

## Judge model options

| Model | Input / output per million tokens | Note |
|---|---|---|
| `claude-opus-5-5` | $4 / $20 | Agreed default; roughly a few cents per evaluation |
| `claude-sonnet-5-5` | $2 / $10 | Cheaper option |
| `claude-haiku-5-5` | $0.10 / $0.50 | Cheapest; suited to bulk benchmark runs |

Prices as of November 2026; check current pricing before Phase 4. An API key comes from the Claude Console and needs prepaid credits. It is separate from the Claude Pro subscription.
