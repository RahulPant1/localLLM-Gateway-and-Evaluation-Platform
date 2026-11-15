"""Export batches for a judge and store the judge's scores.

Works directly on the database, so the backend does not need to be running.
Run from backend/:

    uv run python scripts/evals.py list
    uv run python scripts/evals.py export "<batch name or id>" --out batch.json
    uv run python scripts/evals.py import results.json
    uv run python scripts/evals.py summary "<batch name or id>"

`export` is blind by default (no model names). `--judge NAME` leaves out
outputs that judge has already scored, so an interrupted run can be resumed.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_gateway.config import Settings, get_settings
from llm_gateway.evaluation import build_export, summarise_batch
from llm_gateway.rubric import load_judge_instructions, load_rubric
from llm_gateway.schemas import BatchSummary, EvaluationImport
from llm_gateway.store import Store


class CommandError(Exception):
    """A problem to report to the user without a traceback."""


def find_batch(store: Store, wanted: str) -> BatchSummary:
    """Find a batch by exact id, id prefix, or name (case-insensitive)."""
    batches = store.list_batches()
    key = wanted.strip().lower()
    matches = [b for b in batches if b.id == key] or [
        b for b in batches if b.name.lower() == key or (len(key) >= 6 and b.id.startswith(key))
    ]
    if not matches:
        matches = [b for b in batches if key in b.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise CommandError(f"No batch matches '{wanted}'. Run 'list' to see the batches.")
    names = ", ".join(f"{b.name} ({b.id[:8]})" for b in matches)
    raise CommandError(f"'{wanted}' matches several batches: {names}. Use the id.")


def cmd_list(store: Store, settings: Settings, args: argparse.Namespace) -> str:
    lines = []
    for batch in store.list_batches():
        done = batch.counts["completed"]
        lines.append(
            f"{batch.id[:8]}  {batch.created_at:%Y-%m-%d %H:%M}  {batch.name!r}: "
            f"{batch.request_count} prompts x {len(batch.models)} models, "
            f"{done} completed, {batch.counts['failed']} failed, "
            f"{batch.evaluated_count} evaluated"
        )
    return "\n".join(lines) or "No batches yet."


def cmd_export(store: Store, settings: Settings, args: argparse.Namespace) -> str:
    batch = find_batch(store, args.batch)
    export = build_export(
        store,
        batch.id,
        rubric=load_rubric(settings.rubric_path),
        instructions=load_judge_instructions(settings.judge_prompt_path),
        blind=not args.named,
        exclude_judge=args.judge,
    )
    text = export.model_dump_json(indent=2, exclude_none=True)
    if not args.out:
        return text
    Path(args.out).write_text(text, encoding="utf-8")
    skipped = f" (outputs already scored by {args.judge} left out)" if args.judge else ""
    return f"Wrote {len(export.items)} items from batch {batch.name!r} to {args.out}{skipped}"


def cmd_import(store: Store, settings: Settings, args: argparse.Namespace) -> str:
    try:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        body = EvaluationImport.model_validate(data)
    except OSError as exc:
        raise CommandError(f"Cannot read {args.file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CommandError(f"{args.file} is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        problems = "\n".join(
            f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:20]
        )
        raise CommandError(f"{args.file} does not match the expected format:\n{problems}") from exc
    missing = store.missing_executions([e.execution_id for e in body.evaluations])
    if missing:
        raise CommandError("Nothing was stored. Unknown execution ids: " + ", ".join(missing[:10]))
    rubric = load_rubric(settings.rubric_path)
    stored = store.add_evaluations(
        body.evaluations, weights=rubric.weights, rubric_version=rubric.version
    )
    return f"Stored {len(stored)} evaluations (rubric {rubric.version})."


def cmd_summary(store: Store, settings: Settings, args: argparse.Namespace) -> str:
    batch = find_batch(store, args.batch)
    rows = summarise_batch(store, batch.id)
    if not rows:
        return f"Batch {batch.name!r} has no evaluations yet."
    short = {
        "correctness": "correct",
        "relevance": "relevant",
        "completeness": "complete",
        "instruction_adherence": "instruct",
        "groundedness": "grounded",
    }
    header = f"{'model':<34} {'judge':<14} {'n':>3} {'overall':>8} " + " ".join(
        f"{label:>9}" for label in short.values()
    )
    lines = [f"Batch {batch.name!r}: mean scores (0-10)", header]

    def show(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    for row in sorted(rows, key=lambda r: -(r.overall or 0)):
        lines.append(
            f"{row.model[:34]:<34} {row.judge[:14]:<14} {row.evaluated:>3} {show(row.overall):>8} "
            + " ".join(f"{show(row.dimensions.get(name)):>9}" for name in short)
            + (f"   {row.critical_issues} critical" if row.critical_issues else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list batches").set_defaults(run=cmd_list)
    export = commands.add_parser("export", help="write a batch's outputs for a judge")
    export.add_argument("batch")
    export.add_argument("--out", help="file to write (default: print)")
    export.add_argument("--judge", help="leave out outputs this judge already scored")
    export.add_argument("--named", action="store_true", help="include model names (not blind)")
    export.set_defaults(run=cmd_export)
    importer = commands.add_parser("import", help="store a judge's results")
    importer.add_argument("file")
    importer.set_defaults(run=cmd_import)
    summary = commands.add_parser("summary", help="mean scores per model for a batch")
    summary.add_argument("batch")
    summary.set_defaults(run=cmd_summary)
    args = parser.parse_args(argv)

    settings = settings or get_settings()
    store = Store(settings.database_path)
    try:
        print(args.run(store, settings, args))
    except CommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
