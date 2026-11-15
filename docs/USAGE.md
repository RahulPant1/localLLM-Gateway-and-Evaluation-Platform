# Running and using the gateway

All commands run inside WSL. Ollama must be running (it starts with WSL).

## Start

Everyday use, one process:

```bash
cd frontend && npm run build        # only after frontend code changes
cd ../backend && uv run uvicorn llm_gateway.main:app
```

Open `http://localhost:8000` in a Windows browser.

While changing the frontend, run two processes so the browser reloads on save:

```bash
cd backend && uv run uvicorn llm_gateway.main:app --reload
cd frontend && npm run dev          # then open http://localhost:5173
```

The frontend needs Node 20.19 or newer; `nvm use` in `frontend/` selects Node 23.

## Tests

```bash
cd backend
uv run pytest             # fast; no Ollama or GPU needed
uv run pytest -m live     # real Ollama; loads a model
```

## Screens

- **Overview**: totals for routing, local performance, requests and evaluation scores (see below).
- **Workspace**: one prompt on one model, or on several to compare them.
- **Batch runs**: many prompts on many models, with a results grid and export.
- **History**: past single and comparison requests.
- **Evaluate**: hand stored outputs to a judge and store the scores.

## Overview

The first tab summarises everything stored, for the last 7 days, 30 days or all time. It refreshes every 10 seconds and changes nothing.

- **Routing**: how many prompts the router decided, which model it picked how often, whether the quick rules or the classifier model decided, and how often the fallback was used. Batch prompts count too: the router's pick is recorded for each, although every model runs.
- **Local performance**: tokens generated, speed, GPU time, and the share of answers that needed no model swap (a load under 1 second). The cloud cost line is an estimate: the same token counts at the list price of the model named in `.env` (`GATEWAY_CLOUD_COMPARE_MODEL` and the two `GATEWAY_CLOUD_*_PER_MTOK` prices; set both prices to 0 to hide it).
- **Requests**: answers, typical (median) and slowest time, failures and tokens, per model or per task type. Manual requests have no task type and appear as "Not routed".
- **Evaluation**: average judge scores per model and per rubric dimension, and a table of average score per model for each task type or domain the judge assigned, with the best model marked. The **router check** counts the prompts where the router's pick was also the top-scored model. It needs prompts where the pick and at least one other model were both scored, which in practice means a judged batch.

To fill it with a meaningful sample, paste [sample_batch_prompts.txt](sample_batch_prompts.txt) (28 prompts, four per task type) into **Batch runs**, run it on all models, then judge it in the **Evaluate** tab.

## Automatic routing

In the Workspace, **Automatic** lets the gateway choose the model; **Manual** lets you pick one or several. With Automatic, the response shows a routing panel: the chosen model, task type, complexity, confidence, fallback and the reasons.

- What each model is used for is set in `config/models.yaml`. Edit it and restart the backend.
- `GATEWAY_ROUTER_LLM=off` in `.env` makes routing use only the quick rules, never the classifier model.
- To see how the router treats a set of prompts without running them:

  ```bash
  cd backend
  uv run python scripts/router_report.py --misses     # quick rules only, instant
  uv run python scripts/router_live.py                # full router, uses Ollama
  ```

- Add your own typical prompts, with the category you expect, to `backend/tests/data/routing_prompts.yaml`.

In batch results, a dot (●) marks the model the router would have picked for each prompt.

## Follow-up chat

Every completed answer has a **Continue chat with this model** button, in the Workspace, History and batch results. It opens a conversation in which the model sees all earlier messages. The meter under the conversation shows how much of the model's context window is used; at 90% the conversation cannot be continued and a new request is needed. Local models hold about 8,000 tokens (MedGemma about 4,000), far less than cloud assistants.

## Using the gateway from other tools

The gateway speaks the OpenAI chat format at `http://localhost:8000/v1`, so editors, chat front ends and scripts can use your local models through it. Use the model name `auto` to let the router choose, or an installed model's name. Any API key is accepted; none is checked.

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "auto",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}]
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
answer = client.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "What is a mutex?"}]
)
print(answer.choices[0].message.content, "- answered by", answer.model)
```

In a tool's settings, choose the "OpenAI compatible" provider type, base URL `http://localhost:8000/v1`, model `auto`, and any text as the key.

- **Stored:** every call appears in History marked "API", with the conversation the tool sent, and can be evaluated like any other run.
- **Streaming:** with `"stream": true` the answer arrives piece by piece. If the tool disconnects, generation stops.
- **Thinking is off by default** for these calls, so tools stay responsive. A request turns it on with `"reasoning_effort": "medium"` or `"high"`; `GATEWAY_API_THINK=true` in `.env` changes the default. Thinking text is streamed in a separate `reasoning_content` field, never in the answer.
- **Conversations with `auto`** are routed by their first user message, so a conversation stays on one model.
- **Supported:** text messages, `stream`, `max_tokens`, `temperature`, `reasoning_effort`. **Ignored:** tools (function calling), `stop`, `top_p`, `response_format`. Images are rejected.
- **One at a time:** a call waits its turn behind a running batch or another call. The gateway listens on this machine only.

## Batch runs and later evaluation

1. In **Batch runs**, type prompts separated by a line containing only `---`, pick models (all by default) and run. Work is grouped by model, so 10 prompts on 4 models cost 4 model loads.
2. Every output is stored in `data/gateway.db`, whether or not it is ever evaluated.
3. Have a judge score the outputs (next section). Scores then appear in the batch grid, in a per-model table under it, and on each output.

## Evaluating from the browser

Open the **Evaluate** tab. "Evaluate this run" in History or the Workspace, and "Evaluate this batch" in a batch, open it with that item ticked.

1. **Choose.** The tab lists runs and batches that still have unscored outputs. Tick the ones to judge, or use "Tick everything not fully scored". Fully scored items are not listed: they stay in History and Batch runs, where **Evaluate again** puts one back in this tab, for a second judge or a re-score.
2. **Copy.** The box holds one text with the judge instructions, the rubric and every ticked output. Click **Copy to clipboard**, paste it into Claude Code or any chat model, and send it. Model names are hidden unless you untick the box. A tick box leaves out outputs a named judge has already scored.
3. **Paste back.** Paste the JSON reply into the second box and click **Store scores**. Surrounding text or a code fence is fine.

The scores then appear on each answer, in the batch grid and in the Overview. Each new evaluation is added next to earlier ones; an answer's score is the average across them.

The judge also labels every prompt with a **task type** (knowledge Q&A, reasoning, maths, coding, summarisation, information extraction, classification, structured output, writing, translation, other) and a **domain** (general, medical, legal, finance, science and technology, other). The lists are fixed in `backend/src/llm_gateway/taxonomy.py`, follow the task families used by public LLM benchmarks, and are sent to the judge with the instructions. The Overview uses them for the "Which model suits which task" table. A label the judge invents is stored as "other". Scores stored before this existed show as "Not categorised" until the run is evaluated again. Only the first answer of each run is sent, not follow-up chat. For a large selection the text may be too long for one message; the token estimate under the box shows its size. In that case use `/evaluate-batch` below, which works through a batch in chunks.

## Evaluating a batch with Claude Code

No API key is needed: Claude Code itself acts as the judge.

1. Open Claude Code in the project folder.
2. Type `/evaluate-batch <batch name>` (for example `/evaluate-batch Batch 2026-11-15 10:30`).

Claude Code then exports the batch without model names, scores every output against the rubric, stores the scores, and reports the mean scores per model. An interrupted run can be repeated: outputs it already scored are skipped.

The pieces it uses, which you can edit:

| File | What it is |
|---|---|
| `config/rubric.yaml` | The five dimensions, what low and high scores mean, and the weights. Change `version` when you change it. |
| `config/judge_prompt.md` | The instructions every judge follows, and the result format. |
| `.claude/commands/evaluate-batch.md` | The steps Claude Code runs for `/evaluate-batch`. |

The same steps by hand, from `backend/` (the backend does not need to be running):

```bash
uv run python scripts/evals.py list
uv run python scripts/evals.py export "<batch>" --judge claude-code --out batch.json
uv run python scripts/evals.py import results.json
uv run python scripts/evals.py summary "<batch>"
```

`batch.json` contains the rubric, the judge instructions and the outputs, so it can be handed to any judge: another model, a person, or a chat window. Results go back in the format shown at the end of `config/judge_prompt.md`. The gateway computes the overall score itself from the rubric weights. Importing never overwrites: several judges can score the same output, and each record is kept. `export --named` includes model names, for your own reading; do not give that version to a judge.

Batches can also be submitted without the UI:

```bash
curl -s localhost:8000/api/batches -H 'content-type: application/json' \
  -d '{"name": "my set", "prompts": ["first prompt", "second prompt"]}'
```

A prompt may be an object with `prompt`, `system` and `source_material`. Add `"models": [...]` to limit the models, and `"think": false` to switch thinking off where the model allows it.
