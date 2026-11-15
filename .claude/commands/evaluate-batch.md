---
description: Judge a gateway batch against the rubric and store the scores
argument-hint: <batch name or id> [judge name]
---

Evaluate the outputs of a Local LLM Gateway batch, acting as an independent judge.

Arguments: `$ARGUMENTS` (the batch name or id, optionally followed by a judge name; the judge name defaults to `claude-code`).

Steps:

1. If no batch was given, run `cd backend && uv run python scripts/evals.py list` and ask which batch to evaluate.
2. Export the batch, blind, to a file in your scratchpad directory:
   `cd backend && uv run python scripts/evals.py export "<batch>" --judge <judge> --out <scratchpad>/batch.json`
   The export contains the rubric, the judge instructions and the items. It leaves out items this judge has already scored, so an interrupted run can be resumed. If it reports no items, say so and stop.
3. Read the exported file. Follow its `instructions` and `rubric` exactly.
4. Judge every item independently. The export is blind on purpose: do not look up which model wrote an answer, do not run a named export, and do not query the database or the API for model names before all scores are stored.
5. Write the results to `<scratchpad>/results.json` in the format given in the instructions, with `judge` set to the judge name. For a large batch, work through the items in chunks of about 20 and import each chunk before continuing.
6. Store them: `cd backend && uv run python scripts/evals.py import <scratchpad>/results.json`
7. Only after everything is stored, run `cd backend && uv run python scripts/evals.py summary "<batch>"` and report the mean scores per model, plus any critical issues you recorded.
