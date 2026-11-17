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

- **Overview**: which model suits which task, the leaderboard (with the function-calling result) and the router check. Routing, speed and request totals are under **System details** (see below).
- **Playground**: one prompt on one model, or on several to compare them, with your recent prompts on the left. With **Tools** ticked, the model works as an agent on real apps, read-only (see "Connecting real apps"). Calls from other tools (API) are listed there when you tick "Include calls from other tools".
- **Tests**: three views of the fixed tests.
  - **Models**: the lineup. Add a new model to every existing test, or hide one (see below).
  - **Batches**: many prompts on many models, with a results grid and export.
  - **Tool use**: can each model call functions correctly? Scored automatically (see below).
  - **Agents**: can each model finish a task that takes several tool calls? Scored automatically (see below).
- **Evaluate**: hand stored outputs to a judge and store the scores. Batches are listed first; calls from other tools are left out unless you include them.

## Overview

The first tab summarises everything stored, for the last 7 days, 30 days or all time. It refreshes every 10 seconds and changes nothing.

From top to bottom:

- **Which model suits which task**: average judge score per model for each task type (or domain) the judge assigned, as a shaded grid with a ★ on the clear best in each row. Switch between task type and domain at the top right.
- **Model leaderboard**: models ranked by overall score, with each rubric dimension, the number of prompts compared, critical issues (answers the judge flagged as potentially harmful if trusted) and the share of correct function calls from Tests → Tool use.

  **How the scores are counted.** Models are ranked only on prompts that every model has a score for, so a model judged on part of the prompts, or on an easier set, cannot look better or worse than it is; the note above the table says how many prompts that is and how many were left out. When a judge scores an answer again, its newer score replaces the older one. An answer scored by several judges counts once, with their scores averaged. The task grid uses the same prompts.
- **Is the router worth it?**: the average score of the router's picks next to always using the best single model and the best model chosen for every prompt (a ceiling no router can reach in advance), and on how many prompts the router's pick was a top-scored model. It needs prompts where the pick and at least one other model were both scored, which in practice means a judged batch.
- **System details** (collapsed; click to open):
  - **Routing**: how many prompts the router decided, which model it picked how often, whether the quick rules or the classifier model decided, and how often the fallback was used. Batch prompts count too: the router's pick is recorded for each, although every model runs.
  - **Local performance**: tokens generated, speed, GPU time, and the share of answers that needed no model swap (a load under 1 second). The cloud cost line is an estimate: the same token counts at the list price of the model named in `.env` (`GATEWAY_CLOUD_COMPARE_MODEL` and the two `GATEWAY_CLOUD_*_PER_MTOK` prices; set both prices to 0 to hide it).
  - **Requests**: answers, typical (median) and slowest time, failures and tokens, per model or per task type. Manual requests have no task type and appear as "Not routed".

To fill it with a meaningful sample, paste [sample_batch_prompts.txt](sample_batch_prompts.txt) (28 prompts, four per task type) into **Tests → Batches**, run it on all models, then judge it in the **Evaluate** tab.

## Adding or hiding a model

**Tests → Models** lists every installed model, and every model that has stored results, with how much of each test it has done: per batch, prompts answered, failed and judged; per tool-use run, cases done and the share correct.

- **Run all tests** appears when a model has not done everything. It runs the model on every prompt of every existing batch, and every case of the latest tool-use run, that it has not done yet, and queues again its earlier failures caused by the environment (timeouts, low memory, interruptions). The new answers appear in the same batch grids and tool-use tables as the other models'. Batches that are still running are skipped; try again when they finish. Batches and tool-use runs take turns as whole jobs (one finishes before the next starts), so two of them never swap models on every call; single prompts from the Playground or the `/v1` endpoint still get a turn in between. Models without tool support skip the tool-use tests.
- Tool-use cases are scored as soon as they finish. Batch answers need a judge: the tab shows how many are waiting, with a link to **Evaluate**. With Claude Code, `/evaluate-batch "<batch name>"` judges only what is not scored yet. Once the scores are stored, the Overview includes the new model.
- **Hide** takes a model and all its results out of every screen, export and count, out of the model pickers and out of automatic routing. Nothing is deleted. Hidden models are listed at the bottom of the tab; **Restore** brings the model and every result back. Running all tests on a hidden model restores it first.

## Tool use (function calling)

**Tests → Tool use** tests whether a model can call functions. Each case gives the model a request and one or more function definitions (sent through Ollama's `tools` field, with no system prompt). The model must call the right function with the right arguments, call several when the request needs it, or make no call when nothing fits.

The cases are a fixed sample of the Berkeley Function Calling Leaderboard (BFCL v3, Apache-2.0): 10 each of *simple* (one function), *multiple* (pick one of several), *parallel* (several calls at once) and *irrelevance* (no function fits). They are in `config/tool_suites/bfcl_v1.json`, built by `backend/scripts/build_toolcall_set.py` with seed 2026.

Scoring is automatic and follows BFCL's checks: the name must match, required arguments must be present with an accepted value, no undefined argument may appear, and numbers compare by value but a number sent as text is wrong. Text compares without case or punctuation. Parallel calls may come in any order.

Some models write the call as JSON in their answer instead of using the tool-call field. Those calls are still scored, but marked: the **Proper format** column shows the share of correct calls that came through the real field, which is what an application or agent framework needs.

Thinking is off by default. Models that cannot switch it off (deepseek-r1) get 4,096 output tokens instead of 1,024, so they are not cut off before answering. Models without tool support (MedGemma) are not offered.

## Agent tasks (multi-step)

**Tests → Agents** runs tasks that take several tool calls, on simulated apps, and scores what the model actually did. Nothing real is read, sent or changed.

**The simulated apps.**

- **Files:** a small file system with meeting notes, a to-do list and sales reports. Tools: `list_files`, `read_file`, `write_file`, `delete_file`.
- **Mail:** an inbox of five emails. Tools: `search_emails` (words, plus Gmail-style `from:`, `subject:`, `after:`, `before:`), `read_email`, `send_email`, `create_draft`, `archive_email`.
- **Company API:** one `http_request` tool for weather, exchange rates, orders, a contacts directory and support tickets.

Each task offers only the tools of the apps it needs. The model gets a short system prompt ("complete the task with the tools; look things up instead of guessing") and the task. The gateway runs the loop an agent framework would: model turn, tool calls, results back to the model, until the model gives a final answer.

**The 20 tasks** (`config/agent_suites/agents_v1.json`, written for this project), in six groups:

| Group | Tasks | What they test |
|---|---|---|
| Files | 3 | find, read and write files without losing content |
| Mail | 3 | answer from an email, draft without sending, archive only the right email |
| API | 3 | call endpoints and act on the result (open a ticket for a delayed order) |
| Mixed | 5 | chain two or three apps (reply to an email with a date from the notes; total a CSV and email it) |
| Recovery | 3 | a slightly wrong file name, an API that asks to retry once, a contact that does not exist |
| Safety | 3 | instructions planted in an email or a file ("forward every email to…", "delete the reports"), and a delete request that must not remove more than asked |

**Scoring.** Each task has checks run against the final state of the apps and the model's last answer. For example: an email went to `priya@northwind.example` and mentions 4 December; nothing was sent at all; `/reports/q3-sales.csv` still exists; the answer says 270. A task passes when every check passes; the **Checks** column also gives the share of checks passed, so partly done tasks show. Text checks ignore case and accept common variants of dates and numbers ("1,250" and "1250").

**Also recorded:** steps (model turns), tool calls, invalid calls (an unknown tool or a missing argument), calls the model wrote as text instead of real tool calls (run anyway and counted), and time. A run stops after 10 steps, or when the model repeats the same call three times in a row.

**Reading a task.** Click a cell to see the task, each check with its result, and every step: what the model wrote, each call with its arguments, and what the app returned.

**Defaults.** Thinking off, temperature 0, at most 1,024 output tokens per step (4,096 for models that can think). Models without tool support are not offered. **Run all tests** in Tests → Models adds a new model to the latest agent run as well.

## Connecting real apps (agent mode in the Playground)

In the **Playground**, the **Tools** row lets a model work as an agent on real apps: **Local files**, **Gmail** and **Web APIs**. Tick one or more, pick a model that can call tools (qwen3.5 by default, the best agent in the tests), and run. The model calls the tools, reads what they return and answers; the steps appear above the answer as they happen, and the run is saved like any other prompt.

**Everything is read-only.** No tool can send, write, change or delete anything, so a model that misreads a request, or is misled by text inside an email, cannot do damage. The model is also told that what tools return is data, not instructions.

| App | Tools | What leaves this machine |
|---|---|---|
| Local files | `list_files`, `read_file` (long files in parts), `search_files` (matching lines with the lines after them) | nothing |
| Gmail | `search_emails` (Gmail search syntax), `read_email` | the searches and reads go to Google, your own mailbox |
| Web APIs | `get_weather` (Open-Meteo), `convert_currency` (Frankfurter, ECB rates), `wikipedia_summary` | only the city, currency codes or topic |

The web tools are fixed functions, not "fetch any URL", so a model cannot be talked into sending your data to an address someone planted in an email or a file.

**Saved runs and judging.** An agent run is stored with its steps, and the tool results become the request's source material, so a judge can score whether the answer is grounded in what the tools returned. Runs that read Gmail are marked "contains Gmail data" in the Evaluate tab and are not picked by "Tick everything not fully scored": the judge package would include text from your mailbox.

### Local files

Set the folder in `.env` and restart the backend:

```
GATEWAY_FILES_ROOT=/mnt/c/Users/<you>/Documents/agent-share
```

Only that folder and its subfolders can be read. Paths outside it, including through `..` or symbolic links, are refused; hidden files and folders such as `.git` are skipped; binary files are not read. A dedicated folder with copies of what you want the model to see is safer than your whole home folder.

### Gmail

This uses your own Google Cloud OAuth client and only the **read-only** Gmail permission. A separate test Google account is recommended. Setup takes about ten minutes, once:

1. Open the [Google Cloud console](https://console.cloud.google.com/) and create a project (any name).
2. **APIs & Services → Library**: enable the **Gmail API**.
3. **Google Auth Platform** (OAuth consent screen): user type **External**; give an app name and your email. Under **Audience**, add the Google account you will connect as a **test user**. Leave the app in **Testing**.
4. **Clients → Create client**: type **Desktop app**. Download the JSON file and save it as `data/gmail_client_secret.json` in this project (the `data/` folder is never committed).
5. Run `cd backend && uv run python scripts/gmail_auth.py`. Open the link it prints, sign in with the test account and allow read-only access. Google warns that the app is not verified: it is your own app, so choose **Continue**. If the browser then shows an error page (WSL can block the return address), copy the page's address and paste it into the terminal.
6. Restart the backend. **Gmail** is now available in the Playground's Tools row, showing the connected address.

While the app is in Testing, Google expires the permission after 7 days; run step 5 again when Gmail shows as not connected. To disconnect, delete `data/gmail_token.json` and remove the app at [myaccount.google.com/permissions](https://myaccount.google.com/permissions).

### Web APIs

On by default; they need no keys. `GATEWAY_WEB_APIS=false` in `.env` switches them off.

## Answers appear while they are written

In the Playground, a follow-up chat and an open batch result, the answer text grows on screen while the model generates it, refreshed about twice a second. A thinking model first shows "Thinking before it answers…"; its reasoning is under **Thinking**. Switching tabs or refreshing the page does not lose the answer: the model keeps running in the backend and the screen picks the text up again.

## When a model runs out of room while thinking

A thinking model can spend its whole context window on reasoning and leave nothing for the answer. Such a run is marked **Failed: ran out of room while thinking**, not Completed; its reasoning is kept under **Thinking**.

The failed card has a **Run again without thinking** button. It runs the same prompt on the same model with thinking off.

- **Where the second attempt goes:** into the same run, next to the failed one, labelled "Attempt 2, thinking off". It does not create a separate entry in the Playground's list. In a batch it stays in the batch, and the grid cell for that prompt and model then shows the second attempt.
- **The failed attempt stays on record:** it still counts as one failed run in the batch totals and the Overview. So a batch can read "112 completed, 2 failed" when every prompt has an answer.
- **Exports and evaluation use the answer that succeeded:** the batch export, the Evaluate tab's judge text and `/evaluate-batch` include completed answers only, so they contain the second attempt and never the failure.
- **In a follow-up chat** the button is on the failed message, and the new attempt takes that message's place in the conversation.
- Runs stored before this change still show as Completed with an empty answer.

## Automatic routing

In the Playground, **Automatic** lets the gateway choose the model; **Manual** lets you pick one or several. With Automatic, the response shows a routing panel: the chosen model, task type, complexity, confidence, fallback and the reasons.

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

Every completed answer has a **Continue chat with this model** button, in the Playground and in batch results. It opens a conversation in which the model sees all earlier messages. The meter under the conversation shows how much of the model's context window is used; at 90% the conversation cannot be continued and a new request is needed. Local models hold about 8,000 tokens (MedGemma about 4,000), far less than cloud assistants.

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

- **Stored:** every call appears in the Playground's list marked "API" (tick "Include calls from other tools"), with the conversation the tool sent, and can be evaluated like any other run.
- **Streaming:** with `"stream": true` the answer arrives piece by piece. If the tool disconnects, generation stops.
- **Thinking is off by default** for these calls, so tools stay responsive. A request turns it on with `"reasoning_effort": "medium"` or `"high"`; `GATEWAY_API_THINK=true` in `.env` changes the default. Thinking text is streamed in a separate `reasoning_content` field, never in the answer.
- **Conversations with `auto`** are routed by their first user message, so a conversation stays on one model.
- **Supported:** text messages, `stream`, `max_tokens`, `temperature`, `reasoning_effort`, and function calling (below). **Ignored:** `stop`, `top_p`, `response_format`. Images are rejected.
- **One at a time:** a call waits its turn behind a running batch or another call. The gateway listens on this machine only.

### Function calling

Send `tools` in the OpenAI format and the model can ask your program to call them. Your program runs the function and sends the result back as a `tool` message; the gateway never runs anything itself.

```python
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
messages = [{"role": "user", "content": "What's the weather in Lisbon?"}]

reply = client.chat.completions.create(model="auto", messages=messages, tools=tools)
call = reply.choices[0].message.tool_calls[0]            # get_weather({"city": "Lisbon"})
messages += [reply.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": "18 C, light rain"}]
final = client.chat.completions.create(model="auto", messages=messages, tools=tools)
print(final.choices[0].message.content)
```

- **Which models:** only models whose Ollama entry lists tool support (see Tests → Models). Naming one that cannot call tools returns an error listing the ones that can. With `auto`, the router decides as usual, and if its choice cannot call tools the next suitable model that can is used; the routing explanation stored with the call says so.
- **Calls written as text are repaired.** Some models (`qwen2.5-coder:7b`) write the call as JSON in their answer instead of using the tool-call field. When every such call names a function you offered, the gateway returns it as a real tool call and sets `gateway.tool_calls_recovered_from_text` to `true`.
- **Several calls at once** come back as several entries in `tool_calls`, as in OpenAI's API.
- **Streaming** works, but a turn that may call tools is generated in full before it is sent, because call arguments are only usable once complete.
- **`tool_choice`:** `"none"` sends no tools. Other values are accepted but not enforced: the model decides whether to call.
- **Stored:** each turn is stored like any other API call. The prompt is your last user message; calls the model made appear as `[tool call] name({...})` lines, and earlier calls and results are kept as the request's source material.

## Memory guard and resuming a batch

Ollama keeps a cache of every prompt the loaded model has seen, in system memory (about 120 MB per prompt, up to 8 GB), and only releases it when the model is unloaded. A long batch on one model can therefore use up all of WSL's memory, at which point WSL stalls completely.

- **Guard:** before every run the gateway checks free memory. Below `GATEWAY_MIN_FREE_MEMORY_MB` (1500 by default) it unloads the loaded model, which releases the cache, and carries on; the model reloads in a few seconds. The status strip shows **RAM free** and how often memory was freed. If memory is critically low even with no model loaded, the run fails with `low_memory` and does not start.
- **Resume:** a batch that was cut short (backend restarted, time limit hit, low memory) shows a **Resume N interrupted runs** button. It reruns only those; completed answers are kept.

## Batches and later evaluation

1. In **Tests → Batches**, open **New batch**, add the prompts (below), check the preview, pick models (all by default) and run. Work is grouped by model, so 10 prompts on 4 models cost 4 model loads.
2. Every output is stored in `data/gateway.db`, whether or not it is ever evaluated.
3. Have a judge score the outputs (next section). Scores then appear in the batch grid, in a per-model table under it, and on each output.

### Adding the prompts

Paste them, or **Upload file**. A preview lists every prompt found before anything runs; expand one to read it in full, or click ✕ to leave it out.

**Files.**

| Format | What it holds |
|---|---|
| `.json` | a list of strings, or of objects with `prompt` and optional `system` and `source_material`; or `{"prompts": [...]}` |
| `.jsonl` | one such string or object per line |
| `.csv`, `.tsv` | a `prompt` column, and optional `system` and `source_material` columns |
| `.txt`, `.md` | plain text; it is put in the box and split like pasted text |

**Pasted text** is split by the method chosen under the box. **Automatic** uses lines holding only `---` if there are any; otherwise numbered items (`1.`, `2)`, `Prompt 3:`, `**Q4:**`), which is how chat models usually write lists; otherwise blank lines; otherwise one prompt per line. Each of these can also be chosen directly. Numbering is removed from the prompts, and code-fence lines (```) are ignored.

**Smart split** is for messy text, such as a list another chat model wrote with an introduction, group headings, prompts that run over several paragraphs, and a closing remark. A local model (gemma4 by default, chosen next to the button; `GATEWAY_SPLIT_MODEL` sets the default) reads a numbered view of the lines and only says on which line each prompt starts and which lines are not prompts. The gateway then cuts your own text at those lines, so a prompt is never reworded. Two rules guard against the model's mistakes: a line that follows a line ending with a colon ("Summarise the following passage:") always stays with its prompt, and Markdown headings at the start or end of a prompt are dropped. It takes about a second once the model is loaded. Very long text (over 400 lines) is better uploaded as JSON, JSONL or CSV.

## Evaluating from the browser

Open the **Evaluate** tab. "Evaluate this run" in the Playground, and "Evaluate this batch" in a batch, open it with that item ticked.

1. **Choose.** The tab lists runs and batches that still have unscored outputs. Tick the ones to judge, or use "Tick everything not fully scored". Fully scored items are not listed: they stay in the Playground and Tests → Batches, where **Evaluate again** puts one back in this tab, for a second judge or a re-score.
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
