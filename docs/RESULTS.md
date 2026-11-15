# Results: four local models on 28 prompts

What was measured, how, and what it does and does not show. Numbers come from one batch run on one machine; read the limits at the end before quoting any of them.

## Setup

- **Machine:** laptop with an NVIDIA RTX 4060 (8 GB VRAM), WSL2 Ubuntu with 6 GB RAM and no swap. Ollama 0.40.3. One model fits in VRAM at a time.
- **Models:** `qwen3.5:4b`, `qwen2.5-coder:7b`, `deepseek-r1:8b`, `MedAIBase/MedGemma1.5:4b-it-q8_0`. `qwen3.5:4b` and `deepseek-r1:8b` think before answering, and thinking was on. Context window 8192 tokens (4096 for MedGemma).
- **Prompts:** [sample_batch_prompts.txt](sample_batch_prompts.txt), 28 short prompts written by the author: general knowledge, classification, extraction, structured output (JSON, YAML, table), maths, reasoning, coding, and a few medical-text tasks (all fictional). Each prompt ran once on each model, 112 answers in total.
- **Judge:** Claude, through Claude Code (`/evaluate-batch`), model Claude Fable 5.2. The judge saw the answers without model names, in random order, against the fixed rubric in [`config/rubric.yaml`](../config/rubric.yaml): correctness, relevance, completeness and instruction adherence, each 0 to 10 (groundedness was not applicable). The overall score is the plain mean of those four, computed by the app, not by the judge. The judge also labelled each prompt with a task type and a domain from the fixed lists in `backend/src/llm_gateway/taxonomy.py`, which follow the task families used by public benchmarks.
- **Two prompts were re-run.** `qwen3.5:4b` ran out of room while thinking on two prompts (see finding 3). Those two were run again with thinking off, and the scores for them are for the second attempt.

## Scores by model

| Model | Overall | Correct | Relevant | Complete | Follows instructions | Median time | 90th percentile | Tokens/s | Critical issues |
|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5:4b` | **9.55** | 9.43 | 9.59 | 9.64 | 9.52 | 16.7 s | 65.9 s | 62.8 | 0 |
| `deepseek-r1:8b` | 8.93 | 9.11 | 9.21 | 8.82 | 8.57 | 13.1 s | 20.4 s | 47.2 | 0 |
| `MedGemma1.5:4b` | 8.81 | 8.70 | 8.95 | 9.11 | 8.48 | 2.6 s | 11.6 s | 49.0 | 2 |
| `qwen2.5-coder:7b` | 8.77 | 8.11 | 9.27 | 8.82 | 8.88 | 2.0 s | 8.7 s | 49.8 | 4 |

Times are per answer and include a model swap when one was needed (5 to 19 s, see [SWAP_TIMES.md](SWAP_TIMES.md)). Critical issues are answers the judge flagged as potentially harmful if trusted, for example a wrong dosing schedule, a normal lab value reported as abnormal, or a follow-up interval read as one year instead of one week.

## Scores by task type

Mean overall score; the best model in a row is in bold when it leads by a clear margin. With 1 to 5 prompts per row these are indications, not rankings.

| Task type (prompts) | `qwen3.5:4b` | `deepseek-r1:8b` | `qwen2.5-coder:7b` | `MedGemma1.5:4b` |
|---|---|---|---|---|
| Information extraction (5) | **9.93** | 9.75 | 8.65 | 9.57 |
| Knowledge Q&A (5) | **9.23** | 7.98 | 8.32 | 8.42 |
| Structured output (5) | **9.60** | 8.38 | 8.55 | 9.00 |
| Classification (4) | 9.97 | 8.88 | 9.97 | 9.75 |
| Coding (4) | 9.54 | 9.44 | **9.69** | 8.53 |
| Maths (2) | 9.82 | 9.75 | 9.82 | 9.12 |
| Reasoning (2) | 8.00 | **8.50** | 5.62 | 5.56 |
| Summarisation (1) | 9.88 | 9.75 | 8.38 | 9.12 |

| Domain (prompts) | `qwen3.5:4b` | `deepseek-r1:8b` | `qwen2.5-coder:7b` | `MedGemma1.5:4b` |
|---|---|---|---|---|
| General (15) | **9.93** | 9.38 | 9.10 | 9.42 |
| Science and technology (9) | **9.24** | 8.49 | 8.64 | 7.97 |
| Medical (4) | **8.81** | 8.25 | 7.81 | 8.37 |

## Findings

1. **The small general model led.** `qwen3.5:4b` scored highest overall and in most task types. Of the specialists, the coder model led on coding (9.69 against 9.54, a small margin) and `deepseek-r1:8b` on reasoning (8.50 against 8.00, on two prompts).
2. **The medical model did not win on medical text.** On the four medical prompts `MedGemma1.5:4b` scored 8.37 against 8.81 for `qwen3.5:4b`. The judge is not a clinician, and four prompts is very few.
3. **Thinking mode can cost an entire answer.** On two of the 28 prompts `qwen3.5:4b` used its whole 8,192-token context window on reasoning (about 8,100 tokens, 132 s and 136 s) and returned an empty answer. Both prompts were answered normally with thinking off. On the 26 prompts where thinking finished, `qwen3.5:4b` averaged 9.74. The app now reports this case as a failure and offers "Run again without thinking".
4. **Speed and quality pull apart.** The two models that think were the slowest (median 13 to 17 s). The coder and medical models answered in about 2 to 3 s at a lower score.

## The router on the same prompts

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

## What this does not show

- **Small sample.** 28 prompts, one to five per task type. One run per model per prompt, so run-to-run variation is unknown.
- **One judge, not independent of the author.** All 110 scores come from one Claude session, the one used to build the tool. The judging was blind to model names, but it has not been checked against a second judge or against human ratings. Judge agreement is the obvious next test.
- **Author-written prompts.** The prompts, the routing rules and the routing table were all written by the same person, so the router has not been tested on someone else's prompts.
- **Short, easy prompts.** Scores near 10 for several models mean the set cannot separate them. It says little about long documents or hard problems.
- **Settings matter.** Results are for the default sampling settings, context size 8192 and thinking on; other settings would change them, thinking especially.

## Reproduce

1. Start the app ([USAGE.md](USAGE.md)) with the four models installed.
2. Paste `docs/sample_batch_prompts.txt` into **Batch runs** and run it on all models.
3. Judge it in the **Evaluate** tab (copy the package into any chat model and paste the JSON reply back), or run `/evaluate-batch` in Claude Code. See the rubric in `config/rubric.yaml`.
4. Read the tables in the **Overview** tab.
