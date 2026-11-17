"""Turns pasted text or an uploaded file into a list of batch prompts.

Prompt sets usually come from somewhere else - often another chat model - so
the format varies. Supported:

- Files: JSON (a list of strings, or objects with `prompt` and optional
  `system` / `source_material`, or {"prompts": [...]}), JSON Lines, CSV with a
  `prompt` column, and plain text.
- Plain text, split by a chosen method or detected automatically: lines
  holding only `---`, numbered items ("1.", "2)", "Prompt 3:"), blank lines,
  or one prompt per line.
- A "smart" split by a local model, for messy text. The model only says on
  which line each prompt starts and which lines are not prompts at all (an
  introduction, a closing remark); the text itself is cut here, so prompts
  are never reworded.
"""

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from llm_gateway.schemas import PromptInput

Method = Literal["auto", "separator", "numbered", "blank_lines", "lines", "smart"]

_SEPARATOR = re.compile(r"^\s*-{3,}\s*$")
# "1. ", "12) ", "Prompt 3:", "**Prompt 3:**", "Q4:", "#5 " at the start of a line.
_NUMBERED = re.compile(
    r"^\s*(?:[*_#>]+\s*)?(?:(?:prompt|question|q|task|item)\s*#?\s*\d+\s*[:.)\-–]|#?\d+\s*[.):\-–])\s*(?:[*_]+\s*)?",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^\s*```")
# The numbered view a model sees shows only the start of each line.
VIEW_CHARS = 140
MAX_SMART_LINES = 400


@dataclass
class Parsed:
    prompts: list[PromptInput]
    method: str
    notes: list[str] = field(default_factory=list)


def _clean(block: str, strip_number: bool) -> str:
    text = block.strip("\n").strip()
    if strip_number:
        text = _NUMBERED.sub("", text, count=1).strip()
    return text


def _blocks(lines: list[str], starts: list[int]) -> list[str]:
    bounds = [*starts, len(lines)]
    return ["\n".join(lines[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]


def _without_fences(text: str) -> str:
    """Drop ``` lines: chat models often wrap a whole list in a code block."""
    return "\n".join(line for line in text.splitlines() if not _FENCE.match(line))


def split_text(text: str, method: Method = "auto") -> Parsed:
    lines = _without_fences(text).splitlines()
    notes: list[str] = []
    if method == "auto":
        if any(_SEPARATOR.match(line) for line in lines):
            method = "separator"
        elif sum(bool(_NUMBERED.match(line)) for line in lines) >= 2:
            method = "numbered"
        elif (
            any(not line.strip() for line in lines)
            and len([b for b in re.split(r"\n\s*\n", "\n".join(lines)) if b.strip()]) > 1
        ):
            method = "blank_lines"
        else:
            method = "lines"

    if method == "separator":
        chunks = re.split(r"(?m)^\s*-{3,}\s*$", "\n".join(lines))
        prompts = [_clean(c, False) for c in chunks]
    elif method == "numbered":
        starts = [i for i, line in enumerate(lines) if _NUMBERED.match(line)]
        if not starts:
            return Parsed([], method, ["No numbered items found."])
        if any(line.strip() for line in lines[: starts[0]]):
            notes.append("Text before the first numbered item was left out.")
        prompts = [_clean(b, True) for b in _blocks(lines, starts)]
    elif method == "blank_lines":
        prompts = [_clean(b, False) for b in re.split(r"\n\s*\n", "\n".join(lines))]
    elif method == "lines":
        prompts = [_clean(line, True) for line in lines]
    else:
        raise ValueError(f"Unknown split method: {method}")
    return Parsed([PromptInput(prompt=p) for p in prompts if p], method, notes)


def _from_records(records: list) -> list[PromptInput]:
    prompts = []
    for record in records:
        if isinstance(record, str):
            text = record.strip()
            if text:
                prompts.append(PromptInput(prompt=text))
        elif isinstance(record, dict):
            text = str(
                record.get("prompt") or record.get("question") or record.get("input") or ""
            ).strip()
            if text:
                prompts.append(
                    PromptInput(
                        prompt=text,
                        system=(str(record["system"]).strip() or None)
                        if record.get("system")
                        else None,
                        source_material=(str(record["source_material"]).strip() or None)
                        if record.get("source_material")
                        else None,
                    )
                )
    return prompts


def parse_file(text: str, filename: str) -> Parsed:
    """A file's prompts, by its extension (or content, for JSON)."""
    name = filename.lower()
    stripped = text.strip()
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        records = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        return Parsed(_from_records(records), "jsonl")
    if name.endswith(".json") or (stripped[:1] in "[{" and not name.endswith((".txt", ".md"))):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("prompts") or data.get("items") or data.get("cases") or []
        if not isinstance(data, list):
            raise ValueError(
                "The JSON must be a list of prompts, or an object with a 'prompts' list."
            )
        return Parsed(_from_records(data), "json")
    if name.endswith(".csv") or name.endswith(".tsv"):
        dialect = "excel-tab" if name.endswith(".tsv") else "excel"
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
        if rows and not any(
            k and k.strip().lower() in ("prompt", "question", "input") for k in rows[0]
        ):
            raise ValueError("The CSV needs a column named 'prompt'.")
        lowered = [{(k or "").strip().lower(): v for k, v in row.items()} for row in rows]
        return Parsed(_from_records(lowered), "csv")
    return split_text(text)


# ---- smart split by a local model ----

SMART_SCHEMA = {
    "type": "object",
    "properties": {
        "starts": {"type": "array", "items": {"type": "integer"}},
        "skip": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["starts", "skip"],
}

SMART_INSTRUCTIONS = (
    "Below is a text containing several prompts (tasks or questions) meant to be sent "
    "one by one to an AI model. Each line is numbered as [n]; long lines are shortened.\n"
    "Decide where each prompt starts. A prompt may span several lines, including any "
    "passage, data or code it refers to.\n"
    "Return JSON with:\n"
    '- "starts": the line number where each prompt begins, in order;\n'
    '- "skip": line numbers that belong to no prompt: an introduction '
    '("Here are 20 prompts:"), a heading for a group of prompts, or a closing remark.\n'
    "Never skip a passage, data, example or code that a prompt refers to: it belongs "
    "to that prompt, usually right after a line ending with a colon.\n"
    "Do not rewrite anything; only give line numbers."
)


def numbered_view(text: str) -> tuple[list[str], str]:
    """The text's lines, and a compact numbered view of them for the model."""
    lines = _without_fences(text).splitlines()
    view = []
    for number, line in enumerate(lines, start=1):
        if line.strip():
            shown = line.strip()
            if len(shown) > VIEW_CHARS:
                shown = shown[:VIEW_CHARS] + " …"
            view.append(f"[{number}] {shown}")
    return lines, "\n".join(view)


_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S.*|\*\*[^*]+\*\*:?|__[^_]+__:?)\s*$")


def _material_after_colon(lines: list[str], index: int, skip: set[int]) -> bool:
    """True when the nearest earlier text line ends with a colon: the line is
    material that prompt refers to ("Summarise the following passage:")."""
    for j in range(index - 1, -1, -1):
        if lines[j].strip() and j not in skip:
            return lines[j].rstrip().endswith(":") and not _HEADING.match(lines[j])
    return False


def _trim_headings(block: str) -> str:
    """Drop group headings ("### Coding") at the start or end of a prompt."""
    rows = block.strip("\n").splitlines()
    while rows and (not rows[0].strip() or _HEADING.match(rows[0])):
        rows.pop(0)
    while rows and (not rows[-1].strip() or _HEADING.match(rows[-1])):
        rows.pop()
    return "\n".join(rows)


def apply_smart_split(lines: list[str], answer: dict) -> Parsed:
    """Cut the original lines where the model said prompts start.

    The model's "skip" list is checked: a line right after a line ending with
    a colon is the material a prompt refers to and is always kept.
    """
    total = len(lines)
    starts = sorted({int(n) - 1 for n in answer.get("starts", []) if 1 <= int(n) <= total})
    skip = {int(n) - 1 for n in answer.get("skip", []) if 1 <= int(n) <= total}
    starts = [s for s in starts if s not in skip and lines[s].strip()]
    if not starts:
        raise ValueError("The model did not mark any prompt starts.")
    skip = {i for i in skip if i < starts[0] or not _material_after_colon(lines, i, skip)}
    kept = [("" if i in skip else line) for i, line in enumerate(lines)]
    notes = []
    if skip:
        notes.append(f"{len(skip)} line(s) the model judged not to be prompts were left out.")
    if any(line.strip() for line in kept[: starts[0]]):
        notes.append("Text before the first prompt was left out.")
    prompts = [_clean(_trim_headings(b), True) for b in _blocks(kept, starts)]
    return Parsed([PromptInput(prompt=p) for p in prompts if p], "smart", notes)
