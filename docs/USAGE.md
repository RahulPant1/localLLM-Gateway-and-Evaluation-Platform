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

- **Workspace**: one prompt on one model, or on several to compare them.
- **Batch runs**: many prompts on many models, with a results grid and export.
- **History**: past single and comparison requests.

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

## Batch runs and later evaluation

1. In **Batch runs**, type prompts separated by a line containing only `---`, pick models (all by default) and run. Work is grouped by model, so 10 prompts on 4 models cost 4 model loads.
2. Every output is stored in `data/gateway.db`, whether or not it is ever evaluated.
3. Export the batch for a judge. The blind export leaves out model names and thinking text:

   ```bash
   curl -s "localhost:8000/api/batches/<batch id>/export" > batch.json
   ```

4. The judge (another model, Claude Code, or a person) scores each item 0 to 10 on correctness, relevance, completeness, instruction adherence and groundedness (null when there is no source material), keyed by `execution_id`.
5. Store the results:

   ```bash
   curl -s localhost:8000/api/evaluations -H 'content-type: application/json' -d @results.json
   ```

   ```json
   {
     "evaluations": [
       {
         "execution_id": "…",
         "judge": "claude-code",
         "scores": {"correctness": 8, "relevance": 9, "completeness": 7,
                    "instruction_adherence": 9, "groundedness": null},
         "summary": "…",
         "strengths": ["…"],
         "weaknesses": ["…"],
         "recommendations": ["…"],
         "critical_issues": []
       }
     ]
   }
   ```

The gateway computes the overall score itself as the mean of the scored dimensions. Importing never overwrites: several judges can score the same output, and each record is kept. Scores then appear in the batch grid and on each output.

To evaluate inside Claude Code, ask it to "export batch <name>, judge every item against the rubric, and import the results"; it can use the two calls above.

Batches can also be submitted without the UI:

```bash
curl -s localhost:8000/api/batches -H 'content-type: application/json' \
  -d '{"name": "my set", "prompts": ["first prompt", "second prompt"]}'
```

A prompt may be an object with `prompt`, `system` and `source_material`. Add `"models": [...]` to limit the models, and `"think": false` to switch thinking off where the model allows it.
