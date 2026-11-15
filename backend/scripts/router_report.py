"""Show how the router's quick rules handle the labelled prompts.

uv run python scripts/router_report.py            # rules only, no model needed
uv run python scripts/router_report.py --misses   # only the prompts it got wrong
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

from llm_gateway.config import get_settings
from llm_gateway.routing import policy, signals
from llm_gateway.routing.registry import load_registry

PROMPTS = Path(__file__).resolve().parents[1] / "tests" / "data" / "routing_prompts.yaml"


def main() -> None:
    only_misses = "--misses" in sys.argv
    registry = load_registry(get_settings().registry_path)
    items = yaml.safe_load(PROMPTS.read_text())
    right = Counter()
    total = Counter()
    unsure = 0
    for item in items:
        report = signals.analyze(item["prompt"], item.get("source_material"))
        correct = report.category == item["category"]
        total[item["category"]] += 1
        right[item["category"]] += correct
        unsure += report.clarity < get_settings().router_clarity_threshold
        chosen = policy.select(
            registry,
            report.category,
            report.complexity,
            installed=None,
            loaded=set(),
            estimated_tokens=report.estimated_tokens,
        ).model
        if only_misses and correct:
            continue
        mark = "ok  " if correct else "MISS"
        print(
            f"{mark} want={item['category']:<17} got={report.category:<17} clarity={report.clarity:<5} "
            f"{report.complexity:<6} -> {chosen}\n     {item['prompt'][:100]!r}\n     {report.scores}"
        )
    print("\nAccuracy of the quick rules by category:")
    for category in total:
        print(f"  {category:<18} {right[category]}/{total[category]}")
    print(f"  {'all':<18} {sum(right.values())}/{sum(total.values())}")
    print(
        f"Unsure (would ask the classifier model, unless both guesses pick the same model): {unsure}"
    )


if __name__ == "__main__":
    main()
