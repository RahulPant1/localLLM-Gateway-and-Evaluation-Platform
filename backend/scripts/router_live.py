"""Measure the whole router (quick rules plus the classifier model) on the labelled prompts.

Uses the real Ollama service and loads the classifier model. Run from backend/:

    uv run python scripts/router_live.py                 # held-out prompts
    uv run python scripts/router_live.py routing_prompts.yaml
"""

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import httpx2
import yaml

from llm_gateway.catalog import ModelCatalog
from llm_gateway.config import get_settings
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.routing.registry import load_registry
from llm_gateway.routing.router import Router

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "routing_prompts_heldout.yaml"
    items = yaml.safe_load((DATA / name).read_text())
    settings = get_settings()
    registry = load_registry(settings.registry_path)
    async with httpx2.AsyncClient(base_url=settings.ollama_url) as client:
        provider = OllamaProvider(client, default_num_ctx=settings.default_num_ctx)
        router = Router(
            registry,
            provider,
            ModelCatalog(provider),
            ExecutionQueue(provider),
            clarity_threshold=settings.router_clarity_threshold,
        )
        category_ok = model_ok = 0
        decided = Counter()
        llm_ms = []
        for item in items:
            start = time.perf_counter()
            draft = await router.route(item["prompt"], item.get("source_material"))
            elapsed = (time.perf_counter() - start) * 1000
            decided[draft.decided_by] += 1
            if draft.decided_by == "llm":
                llm_ms.append(elapsed)
            # The model a correct classification would have led to.
            want = registry.groups_for(item["category"], draft.complexity)
            want_models = want[0] if want else []
            right_category = draft.category == item["category"]
            right_model = draft.selected_model in want_models
            category_ok += right_category
            model_ok += right_model
            if not right_category:
                mark = "same model" if right_model else "WRONG MODEL"
                print(
                    f"{mark:<11} want={item['category']:<17} got={draft.category:<17} "
                    f"by={draft.decided_by:<16} conf={draft.confidence} -> {draft.selected_model}\n"
                    f"            {item['prompt'][:90]!r}"
                )
    total = len(items)
    print(f"\n{name}: {total} prompts")
    print(f"  right category: {category_ok}/{total}")
    print(f"  right model:    {model_ok}/{total}")
    print(f"  decided by:     {dict(decided)}")
    if llm_ms:
        print(
            f"  classifier calls: {len(llm_ms)}, median {sorted(llm_ms)[len(llm_ms) // 2]:.0f} ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
