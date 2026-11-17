# Results: five local models on 28 prompts, and function calling

What was measured, how, and what it does and does not show. Numbers come from one batch run on one machine; read the limits at the end before quoting any of them.

## Setup

- **Machine:** laptop with an NVIDIA RTX 4060 (8 GB VRAM), WSL2 Ubuntu with 6 GB RAM and no swap. Ollama 0.40.3. One model fits in VRAM at a time.
- **Models:** `qwen3.5:4b`, `qwen2.5-coder:7b`, `deepseek-r1:8b`, `MedAIBase/MedGemma1.5:4b-it-q8_0`, and later `gemma4:e4b` (Q4_K_M, 6.6 GB), added with **Run all tests** in Tests → Models. `qwen3.5:4b`, `deepseek-r1:8b` and `gemma4:e4b` can think before answering, and thinking was on. Context window 8192 tokens (4096 for MedGemma).
- **Prompts:** [sample_batch_prompts.txt](sample_batch_prompts.txt), 28 short prompts written by the author: general knowledge, classification, extraction, structured output (JSON, YAML, table), maths, reasoning, coding, and a few medical-text tasks (all fictional). Each prompt ran once on each model: 112 answers from the first four models, then 28 from `gemma4:e4b`.
- **Judge:** Claude, through Claude Code (`/evaluate-batch`), model Claude Fable 5.2. The judge saw the answers without model names, in random order, against the fixed rubric in [`config/rubric.yaml`](../config/rubric.yaml): correctness, relevance, completeness and instruction adherence, each 0 to 10 (groundedness was not applicable). The overall score is the plain mean of those four, computed by the app, not by the judge. The judge also labelled each prompt with a task type and a domain from the fixed lists in `backend/src/llm_gateway/taxonomy.py`, which follow the task families used by public benchmarks.
- **The fifth model was judged later, by a different Claude model.** The first 112 answers were judged by Claude Fable 5.2; the 28 `gemma4:e4b` answers by Claude Opus 5.5, with the same rubric, instructions and blind export, and the task labels already given to each prompt. Compare `gemma4:e4b` with the others with that in mind.
- **Two prompts were re-run.** `qwen3.5:4b` ran out of room while thinking on two prompts (see finding 3). Those two were run again with thinking off, and the scores for them are for the second attempt.

## Scores by model

| Model | Overall | Correct | Relevant | Complete | Follows instructions | Median time | 90th percentile | Tokens/s | Critical issues |
|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5:4b` | **9.55** | 9.43 | 9.59 | 9.64 | 9.52 | 16.7 s | 65.9 s | 62.8 | 0 |
| `gemma4:e4b` | 9.35 | 9.32 | 9.50 | 9.52 | 9.05 | 5.7 s | 14.1 s | 102.0 | 2 |
| `deepseek-r1:8b` | 8.93 | 9.11 | 9.21 | 8.82 | 8.57 | 13.1 s | 20.4 s | 47.2 | 0 |
| `MedGemma1.5:4b` | 8.81 | 8.70 | 8.95 | 9.11 | 8.48 | 2.6 s | 11.6 s | 49.0 | 2 |
| `qwen2.5-coder:7b` | 8.77 | 8.11 | 9.27 | 8.82 | 8.88 | 2.0 s | 8.7 s | 49.8 | 4 |

Times are per answer and include a model swap when one was needed (5 to 19 s, see [SWAP_TIMES.md](SWAP_TIMES.md)). Critical issues are answers the judge flagged as potentially harmful if trusted, for example a wrong dosing schedule, a normal lab value reported as abnormal, or a follow-up interval read as one year instead of one week.

## Scores by task type

Mean overall score; the best model in a row is in bold when it leads by a clear margin. With 1 to 5 prompts per row these are indications, not rankings.

| Task type (prompts) | `qwen3.5:4b` | `gemma4:e4b` | `deepseek-r1:8b` | `qwen2.5-coder:7b` | `MedGemma1.5:4b` |
|---|---|---|---|---|---|
| Information extraction (5) | 9.93 | 9.95 | 9.75 | 8.65 | 9.57 |
| Knowledge Q&A (5) | **9.23** | 8.62 | 7.98 | 8.32 | 8.42 |
| Structured output (5) | **9.60** | 8.83 | 8.38 | 8.55 | 9.00 |
| Classification (4) | 9.97 | 9.97 | 8.88 | 9.97 | 9.75 |
| Coding (4) | 9.54 | 9.38 | 9.44 | **9.69** | 8.53 |
| Maths (2) | 9.82 | 9.50 | 9.75 | 9.82 | 9.12 |
| Reasoning (2) | 8.00 | **9.88** | 8.50 | 5.62 | 5.56 |
| Summarisation (1) | 9.88 | 8.62 | 9.75 | 8.38 | 9.12 |

| Domain (prompts) | `qwen3.5:4b` | `gemma4:e4b` | `deepseek-r1:8b` | `qwen2.5-coder:7b` | `MedGemma1.5:4b` |
|---|---|---|---|---|---|
| General (15) | **9.93** | 9.74 | 9.38 | 9.10 | 9.42 |
| Science and technology (9) | **9.24** | 9.11 | 8.49 | 8.64 | 7.97 |
| Medical (4) | **8.81** | 8.40 | 8.25 | 7.81 | 8.37 |

## Findings

1. **The small general model led.** `qwen3.5:4b` scored highest overall and in most task types. Of the specialists, the coder model led on coding (9.69 against 9.54, a small margin) and `deepseek-r1:8b` on reasoning (8.50 against 8.00, on two prompts).
2. **The medical model did not win on medical text.** On the four medical prompts `MedGemma1.5:4b` scored 8.37 against 8.81 for `qwen3.5:4b`. The judge is not a clinician, and four prompts is very few.
3. **Thinking mode can cost an entire answer.** On two of the 28 prompts `qwen3.5:4b` used its whole 8,192-token context window on reasoning (about 8,100 tokens, 132 s and 136 s) and returned an empty answer. Both prompts were answered normally with thinking off. On the 26 prompts where thinking finished, `qwen3.5:4b` averaged 9.74. The app now reports this case as a failure and offers "Run again without thinking".
4. **Speed and quality pull apart.** The two models that think were the slowest (median 13 to 17 s). The coder and medical models answered in about 2 to 3 s at a lower score.
5. **The newer model came close to the leader at a third of the time.** `gemma4:e4b` scored 9.35, second to `qwen3.5:4b` (9.55), with a median of 5.7 s and about 100 tokens per second, the fastest generation of the five. It scored best on the two reasoning prompts (9.88) and tied on extraction, and was weaker on knowledge questions, summarising and exact formats (it wraps JSON and YAML in code fences). Both of its critical issues were on medical prompts: it read "review in 1/52" as one month instead of one week, and scheduled metformin three times a day when the letter said twice.

## The router on the same prompts

This section covers the first four models and the routing table they were measured with. `gemma4:e4b` was added to the table afterwards, using these results: it is now first for reasoning, an equal choice with `qwen3.5:4b` for extraction and classification, and a backup for general questions and structured output. The router has not been re-measured with it.

The router's pick for each prompt was recorded but did not choose the model; all four models ran every prompt. This allows comparing its pick with the judged scores. How the router works is in the [README](../README.md#the-router).

| Measure | Result |
|---|---|
| Router's pick was among the top-scored models (ties count) | 18 of 28 |
| Router's pick was the single best model | 8 of 28 |
| Pick scored more than 0.5 below the best | 4 of 28 (three medical, one coding) |
| Mean score of the router's picks | 9.54 |
| Mean score if the best model were chosen for every prompt | 9.72 |
| Mean score always using `qwen3.5:4b` | 9.55 |
| Prompts where the classifier model was needed | 4 of 28 |

On this set the router **matched the best single model but did not beat it**. It loses about 0.19 points to a perfect per-prompt choice. Most of the loss comes from the hand-written routing table sending medical prompts to the medical model, which did not score best on them. Ten of the 18 "top" picks are ties on easy prompts where several models scored 10.

The router's confidence value is **not calibrated**: picks with confidence above 0.9 were among the top in 4 of 6 cases, and picks below 0.7 in 6 of 7.

Separately, on a set of 25 labelled prompts that the rules were never adjusted against, the quick rules alone found the right category for 18, and the full router (rules plus classifier) for 20; it chose the intended model for 23. The 66-prompt set the rules were tuned on is not a fair accuracy measure and is not reported.

## Function calling (tool use)

Can each model call functions correctly? 40 cases sampled from the Berkeley Function Calling Leaderboard (BFCL v3, Apache-2.0; seed 2026), 10 in each category. Each case gives the model a request and function definitions through Ollama's `tools` field, with no system prompt, thinking off and temperature 0. Scoring is automatic (no judge) and follows BFCL's checks; details are in [USAGE.md](USAGE.md#tool-use-function-calling). `MedGemma1.5:4b` has no tool support in Ollama and was not tested.

| Model | Correct (of 40) | Sent as a real tool call | One function (10) | Pick one of several (10) | Several calls at once (10) | No function fits (10) | Mean time |
|---|---|---|---|---|---|---|---|
| `gemma4:e4b` | **37** | 37 of 37 | 9 | 10 | 10 | 8 | 1.1 s |
| `qwen3.5:4b` | 36 | 36 of 36 | 9 | 8 | 10 | 9 | 3.1 s |
| `qwen2.5-coder:7b` | 30 | **1 of 30** | 10 | 10 | 9 | 1 | 1.0 s |
| `deepseek-r1:8b` | 10 | 10 of 10 | 0 | 0 | 0 | 10 | 29.3 s |

6. **Two models are usable for tools, two are not.** `gemma4:e4b` and `qwen3.5:4b` called the right function with the right arguments in 36 to 37 of 40 cases, always through the proper tool-call field.
7. **The coder model knows what to call but does not call it.** `qwen2.5-coder:7b` picked the right function and arguments in nearly every case, but wrote the call as JSON text in its answer in 29 of 30, so an application expecting real tool calls would see none. It also "called" a function in 9 of the 10 cases where none fitted.
8. **deepseek-r1 ignores the tools.** `deepseek-r1:8b` accepts function definitions but never called one: it reasoned at length (about 30 s per case, even with thinking requested off, which this model cannot do) and answered in prose. Its 10 points in the last column come only from never calling anything. It was given 4,096 output tokens so that a cut-off could not explain this.

## What this does not show

- **Small sample.** 28 prompts, one to five per task type. One run per model per prompt, so run-to-run variation is unknown.
- **Tool use is single calls only.** These cases test one turn: choosing and filling in calls. Multi-step tasks, where the model reads a tool's result and decides what to do next, are not tested yet.
- **One judge, not independent of the author.** All 112 first-round scores come from one Claude session, the one used to build the tool. The judging was blind to model names, but it has not been checked against a second judge or against human ratings. Judge agreement is the obvious next test.
- **Author-written prompts.** The prompts, the routing rules and the routing table were all written by the same person, so the router has not been tested on someone else's prompts.
- **Short, easy prompts.** Scores near 10 for several models mean the set cannot separate them. It says little about long documents or hard problems.
- **Settings matter.** Results are for the default sampling settings, context size 8192 and thinking on; other settings would change them, thinking especially.

## Reproduce

1. Start the app ([USAGE.md](USAGE.md)) with the models installed. A model installed later can be added to every existing test with **Run all tests** in **Tests → Models**.
2. Paste `docs/sample_batch_prompts.txt` into **Tests → Batches** (New batch) and run it on all models.
3. Judge it in the **Evaluate** tab (copy the package into any chat model and paste the JSON reply back), or run `/evaluate-batch` in Claude Code. See the rubric in `config/rubric.yaml`.
4. Read the tables in the **Overview** tab.
5. For function calling, open **Tests → Tool use**, open **New tool-use run**, tick the models and run the 40 cases. Scoring is immediate.
