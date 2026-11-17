"""Build a prompt set from items of public benchmarks, sampled with a fixed seed.

Each benchmark contributes a few randomly chosen items, so the set is not hand
picked. Choices made before sampling, not after looking at the items:
  - the seed and the number of items per benchmark (below);
  - a length cap on some benchmarks, so small context windows are not the test.
Items are read from the Hugging Face dataset server; nothing is stored except
the sampled items. The output keeps the gold answers apart from the prompts.

    uv run python scripts/build_benchmark_set.py --out ../docs/benchmark_set_v1.json
"""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from datetime import date

SEED = 2026
WINDOW = 100  # rows fetched per request (the server's maximum)
API = "https://datasets-server.huggingface.co/rows"
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SST_LABELS = ["negative", "positive"]


def fetch(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=30) as response:
                return json.load(response)
        except Exception:  # noqa: BLE001 - retry any network or server hiccup
            time.sleep(2**attempt)
    raise SystemExit(f"could not read {dataset} ({config}, {split})")


def options(labels: list[str], texts: list[str]) -> str:
    return "\n".join(f"{label}) {text}" for label, text in zip(labels, texts, strict=True))


def abstract(context: dict) -> str:
    return "\n\n".join(
        f"{label}: {text}"
        for label, text in zip(context["labels"], context["contexts"], strict=True)
    )


def long_enough(row: dict, low: int, high: int) -> bool:
    return low <= len(row["context"]) <= high


# Each spec: benchmark, where to read it, how many items, an optional filter, and
# how a row becomes a prompt. `task` is only a note; the judge assigns its own labels.
SPECS = [
    {
        "key": "mmlu",
        "benchmark": "MMLU",
        "task": "knowledge_qa",
        "domain": "general",
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "count": 4,
        "license": "MIT",
        "build": lambda r: {
            "prompt": f"{r['question']}\n\n{options(list('ABCD'), r['choices'])}\n\n"
            "Answer with the letter of the correct option, then one sentence explaining why.",
            "gold": {"answer": "ABCD"[r["answer"]], "subject": r["subject"]},
        },
    },
    {
        "key": "arc",
        "benchmark": "ARC-Challenge",
        "task": "reasoning",
        "domain": "science_technology",
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "count": 3,
        "license": "CC BY-SA 4.0",
        "build": lambda r: {
            "prompt": f"{r['question']}\n\n{options(r['choices']['label'], r['choices']['text'])}\n\n"
            "Answer with the letter of the correct option, then one sentence explaining why.",
            "gold": {"answer": r["answerKey"]},
        },
    },
    {
        "key": "gsm8k",
        "benchmark": "GSM8K",
        "task": "math",
        "domain": "general",
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "count": 4,
        "license": "MIT",
        "build": lambda r: {
            "prompt": f"{r['question']}\n\nShow your reasoning briefly, then give the final "
            "answer as a number on the last line.",
            "gold": {"answer": r["answer"].split("####")[-1].strip()},
        },
    },
    {
        "key": "humaneval",
        "benchmark": "HumanEval",
        "task": "coding",
        "domain": "science_technology",
        "dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "split": "test",
        "count": 4,
        "license": "MIT",
        "build": lambda r: {
            "prompt": "Complete this Python function. Reply with the complete function in one "
            f"code block.\n\n```python\n{r['prompt']}```",
            "gold": {"task_id": r["task_id"], "entry_point": r["entry_point"], "test": r["test"]},
        },
    },
    {
        "key": "squad_extract",
        "benchmark": "SQuAD v2",
        "task": "extraction",
        "domain": "general",
        "dataset": "rajpurkar/squad_v2",
        "config": "squad_v2",
        "split": "validation",
        "count": 3,
        "license": "CC BY-SA 4.0",
        "accept": lambda r: long_enough(r, 0, 1500),
        "build": lambda r: {
            "prompt": "Answer the question using only the passage. Give the shortest answer. If the "
            f'passage does not say, reply "Not stated in the passage."\n\nQuestion: {r["question"]}',
            "source_material": r["context"],
            "gold": {"answers": r["answers"]["text"] or "unanswerable (not stated in the passage)"},
        },
    },
    {
        "key": "squad_summary",
        "benchmark": "SQuAD v2 (passages)",
        "task": "summarization",
        "domain": "general",
        "dataset": "rajpurkar/squad_v2",
        "config": "squad_v2",
        "split": "validation",
        "count": 2,
        "license": "CC BY-SA 4.0",
        "accept": lambda r: long_enough(r, 700, 1500),
        "build": lambda r: {
            "prompt": "Summarise this passage in two sentences.",
            "source_material": r["context"],
            "gold": {"note": "open-ended; judged against the passage"},
        },
    },
    {
        "key": "ag_news",
        "benchmark": "AG News",
        "task": "classification",
        "domain": "general",
        "dataset": "fancyzhx/ag_news",
        "config": "default",
        "split": "test",
        "count": 2,
        "license": "research use (see dataset card)",
        "build": lambda r: {
            "prompt": "Classify the topic of this news snippet as World, Sports, Business or "
            f'Sci/Tech. Answer with the label only.\n\n"{r["text"]}"',
            "gold": {"answer": AG_LABELS[r["label"]]},
        },
    },
    {
        "key": "sst2",
        "benchmark": "SST-2 (GLUE)",
        "task": "classification",
        "domain": "general",
        "dataset": "stanfordnlp/sst2",
        "config": "default",
        "split": "validation",
        "count": 2,
        "license": "research use (see dataset card)",
        "build": lambda r: {
            "prompt": "Is the sentiment of this movie review sentence positive or negative? "
            f'Answer with one word.\n\n"{r["sentence"].strip()}"',
            "gold": {"answer": SST_LABELS[r["label"]]},
        },
    },
    {
        "key": "pubmedqa",
        "benchmark": "PubMedQA",
        "task": "knowledge_qa",
        "domain": "medical",
        "dataset": "qiaojin/PubMedQA",
        "config": "pqa_labeled",
        "split": "train",
        "count": 3,
        "license": "MIT",
        "accept": lambda r: len(abstract(r["context"])) <= 2500,
        "build": lambda r: {
            "prompt": "Based only on the abstract, answer the question with yes, no or maybe, "
            f"then give one sentence of justification.\n\nQuestion: {r['question']}",
            "source_material": abstract(r["context"]),
            "gold": {"answer": r["final_decision"]},
        },
    },
    {
        "key": "ifeval",
        "benchmark": "IFEval",
        "task": "structured_output",
        "domain": "general",
        "dataset": "google/IFEval",
        "config": "default",
        "split": "train",
        "count": 3,
        "license": "Apache 2.0",
        "accept": lambda r: len(r["prompt"]) <= 600,
        "build": lambda r: {
            "prompt": r["prompt"],
            "gold": {"instructions": r["instruction_id_list"], "kwargs": r["kwargs"]},
        },
    },
]


def sample(spec: dict) -> list[tuple[int, dict]]:
    """`count` distinct rows chosen at random (seeded) from rows that pass the filter."""
    rng = random.Random(f"{SEED}:{spec['key']}")
    source = (spec["dataset"], spec["config"], spec["split"])
    total = fetch(*source, 0, 1)["num_rows_total"]
    accept = spec.get("accept", lambda row: True)
    chosen: dict[int, dict] = {}
    while len(chosen) < spec["count"]:
        start = rng.randrange(0, max(1, total - WINDOW + 1))
        page = fetch(*source, start, min(WINDOW, total))["rows"]
        fresh = [r for r in page if accept(r["row"]) and r["row_idx"] not in chosen]
        if fresh:
            pick = rng.choice(fresh)
            chosen[pick["row_idx"]] = pick["row"]
    return sorted(chosen.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    items = []
    for spec in SPECS:
        for row_index, row in sample(spec):
            built = spec["build"](row)
            items.append(
                {
                    "id": f"{spec['key']}-{row_index}",
                    "benchmark": spec["benchmark"],
                    "license": spec["license"],
                    "source": f"https://huggingface.co/datasets/{spec['dataset']}",
                    "row": row_index,
                    "task_hint": spec["task"],
                    "domain_hint": spec["domain"],
                    "prompt": built["prompt"],
                    "source_material": built.get("source_material"),
                    "gold": built["gold"],
                }
            )
    keys = [(i["prompt"], i["source_material"]) for i in items]
    assert len(set(keys)) == len(keys), "duplicate prompts"
    document = {
        "name": "Benchmark set v1",
        "seed": SEED,
        "built": date.today().isoformat(),
        "items": items,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
    print(f"Wrote {len(items)} items from {len(SPECS)} benchmarks to {args.out}")


if __name__ == "__main__":
    main()
