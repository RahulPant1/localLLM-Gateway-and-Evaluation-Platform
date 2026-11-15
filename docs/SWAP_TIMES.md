# Model swap times

Measured 2026-11-15 07:51 with `backend/scripts/measure_swap.py` on Ollama 0.40.3, context size 8192 unless the model sets its own.

- **Swap, total**: wall-clock time of a tiny request when a different model was loaded.
- **Swap, load**: the part of that which Ollama reports as model loading.
- **Warm, total**: the same request repeated with the model already loaded.
- Round 1 mostly reads model files from disk; round 2 may benefit from the file cache.

| Round | Previous model | Model | File size | VRAM | Swap, total | Swap, load | Warm, total |
|---|---|---|---|---|---|---|---|
| 1 | `(nothing)` | `deepseek-r1:8b` | 4.9 GB | 5.8 GB | 19.2 s | 18.9 s | 0.2 s |
| 1 | `deepseek-r1:8b` | `MedAIBase/MedGemma1.5:4b-it-q8_0` | 4.1 GB | 4.4 GB | 15.2 s | 9.4 s | 0.1 s |
| 1 | `MedAIBase/MedGemma1.5:4b-it-q8_0` | `qwen2.5-coder:7b` | 4.7 GB | 5.0 GB | 5.8 s | 5.7 s | 0.0 s |
| 1 | `qwen2.5-coder:7b` | `qwen3.5:4b` | 3.3 GB | 3.3 GB | 6.2 s | 5.9 s | 0.1 s |
| 2 | `qwen3.5:4b` | `deepseek-r1:8b` | 4.9 GB | 5.8 GB | 7.1 s | 6.7 s | 0.2 s |
| 2 | `deepseek-r1:8b` | `MedAIBase/MedGemma1.5:4b-it-q8_0` | 4.1 GB | 4.4 GB | 5.8 s | 5.6 s | 0.1 s |
| 2 | `MedAIBase/MedGemma1.5:4b-it-q8_0` | `qwen2.5-coder:7b` | 4.7 GB | 5.0 GB | 5.0 s | 4.8 s | 0.1 s |
| 2 | `qwen2.5-coder:7b` | `qwen3.5:4b` | 3.3 GB | 3.3 GB | 5.9 s | 5.7 s | 0.1 s |
