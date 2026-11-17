"""Build the tool-calling test set from the Berkeley Function Calling Leaderboard (BFCL).

BFCL (Apache-2.0, https://gorilla.cs.berkeley.edu) gives each question, the
functions the model may call, and the accepted answers. This script samples a
fixed number of items per category with a fixed seed, converts the function
definitions to the JSON-schema form models expect, and writes one file the
gateway loads. Run it from the backend folder:

    uv run python scripts/build_toolcall_set.py --out ../config/tool_suites/bfcl_v1.json
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

BASE = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)
SEED = 2026

# category -> (BFCL file, has answer file, how many items, what it tests)
CATEGORIES = {
    "simple": ("BFCL_v3_simple", True, 10, "one function available, one call needed"),
    "multiple": ("BFCL_v3_multiple", True, 10, "several functions available, pick the right one"),
    "parallel": ("BFCL_v3_parallel", True, 10, "the request needs several calls at once"),
    "irrelevance": ("BFCL_v3_irrelevance", False, 10, "no function fits: the model must not call"),
}

JSON_TYPES = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "number": "number",
}


def fetch_lines(path: str) -> list[dict]:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=60) as response:  # noqa: S310
        return [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]


def convert_schema(node):
    """BFCL uses Python-style type names; models expect JSON Schema."""
    if isinstance(node, list):
        return [convert_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            if value == "any":
                continue
            out[key] = JSON_TYPES.get(value, value)
        elif key in ("properties",) and isinstance(value, dict):
            out[key] = {name: convert_schema(spec) for name, spec in value.items()}
        else:
            out[key] = convert_schema(value)
    return out


def to_tool(function: dict) -> dict:
    # Dots are not valid in function names for several model templates.
    name = function["name"].replace(".", "_")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": function.get("description", ""),
            "parameters": convert_schema(function.get("parameters", {})),
        },
    }


def normalise_truth(truth: list[dict]) -> list[dict]:
    return [
        {
            "name": next(iter(call)).replace(".", "_"),
            "arguments": call[next(iter(call))],
        }
        for call in truth
    ]


def build() -> dict:
    rng = random.Random(SEED)
    cases = []
    for category, (file, has_answers, count, _tests) in CATEGORIES.items():
        questions = fetch_lines(f"{file}.json")
        answers = (
            {a["id"]: a["ground_truth"] for a in fetch_lines(f"possible_answer/{file}.json")}
            if has_answers
            else {}
        )
        picked = rng.sample(questions, count)
        for item in picked:
            turns = item["question"][0]
            if len(item["question"]) != 1 or len(turns) != 1:
                continue
            cases.append(
                {
                    "id": item["id"],
                    "category": category,
                    "prompt": turns[0]["content"],
                    "tools": [to_tool(f) for f in item["function"]],
                    # Empty list = the model should not call anything.
                    "expected": normalise_truth(answers[item["id"]]) if has_answers else [],
                }
            )
    return {
        "id": "bfcl_v1",
        "title": "Function calling (BFCL subset)",
        "source": "Berkeley Function Calling Leaderboard v3, Apache-2.0",
        "seed": SEED,
        "categories": {name: spec[3] for name, spec in CATEGORIES.items()},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    suite = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(suite, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(suite['cases'])} cases to {args.out}")


if __name__ == "__main__":
    main()
