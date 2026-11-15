"""Measure how long each model takes to load, to inform the router design.

Runs two rounds over every installed model. Because only one model fits in
VRAM, each first call in a round is a swap (evict the previous model, load
this one); the second call is warm. Round 1 reads model files from disk;
round 2 shows how much the operating system's file cache helps.

Downloads nothing and changes no Ollama settings. Uses the GPU fully for a
few minutes. Run from backend/:

    uv run python scripts/measure_swap.py
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path

import httpx2

from llm_gateway.config import PROJECT_ROOT, get_settings
from llm_gateway.providers.base import GenerationRequest, ProviderError
from llm_gateway.providers.ollama import OllamaProvider

OUTPUT = PROJECT_ROOT / "docs" / "SWAP_TIMES.md"
PROMPT = "Reply with the single word: OK"
ROUNDS = 2


async def timed(provider, model):
    """One tiny generation. Returns (wall_ms, load_ms, tokens_per_second) or an error string."""
    request = GenerationRequest(model=model.name, prompt=PROMPT, think=False, max_tokens=8)
    start = time.perf_counter()
    try:
        result = await provider.generate(request, model)
    except ProviderError as exc:
        return f"{exc.code}: {exc}"
    wall_ms = (time.perf_counter() - start) * 1000
    return wall_ms, result.load_ms, result.tokens_per_second


async def main() -> None:
    settings = get_settings()
    async with httpx2.AsyncClient(base_url=settings.ollama_url) as client:
        provider = OllamaProvider(client, default_num_ctx=settings.default_num_ctx)
        version = await provider.version()
        models = await provider.list_models()
        loaded_before = [m.name for m in await provider.list_loaded()]
        print(f"Ollama {version}; loaded before start: {loaded_before or 'nothing'}")

        rows = []
        previous = loaded_before[0] if loaded_before else "(nothing)"
        for round_number in range(1, ROUNDS + 1):
            for model in models:
                swap = await timed(provider, model)
                warm = await timed(provider, model)
                vram = next(
                    (m.vram_bytes for m in await provider.list_loaded() if m.name == model.name), 0
                )
                rows.append((round_number, previous, model, swap, warm, vram))
                print(f"round {round_number}: {previous} -> {model.name}: {swap} | warm {warm}")
                previous = model.name

    def cell(measurement, index):
        if isinstance(measurement, str):
            return "failed" if index else measurement[:60]
        value = measurement[index]
        return "-" if value is None else f"{value / 1000:.1f} s"

    lines = [
        "# Model swap times",
        "",
        f"Measured {datetime.now():%Y-%m-%d %H:%M} with `backend/scripts/measure_swap.py` "
        f"on Ollama {version}, context size {settings.default_num_ctx} unless the model sets "
        "its own.",
        "",
        "- **Swap, total**: wall-clock time of a tiny request when a different model was loaded.",
        "- **Swap, load**: the part of that which Ollama reports as model loading.",
        "- **Warm, total**: the same request repeated with the model already loaded.",
        "- Round 1 mostly reads model files from disk; round 2 may benefit from the file cache.",
        "",
        "| Round | Previous model | Model | File size | VRAM | Swap, total | Swap, load "
        "| Warm, total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for round_number, previous, model, swap, warm, vram in rows:
        lines.append(
            f"| {round_number} | `{previous}` | `{model.name}` | {model.size_bytes / 1e9:.1f} GB "
            f"| {vram / 1e9:.1f} GB | {cell(swap, 0)} | {cell(swap, 1)} | {cell(warm, 0)} |"
        )
    Path(OUTPUT).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
