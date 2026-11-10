# QLoRA Fine-Tuning & Quantization Pipeline

An end-to-end pipeline that takes an open-source LLM or vision-language model from a
pretrained checkpoint to a quantized, served API: **QLoRA fine-tuning → LoRA merge →
multi-backend quantization → measurement → FastAPI + Gradio serving → Docker**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-orange)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why I built this

Most tutorials stop at "run `Trainer.train()`". In practice the hard parts are the ones
after it: merging adapters correctly, deciding between 4-bit / GGUF / AWQ for *this*
deployment target, and knowing what serving actually costs once you ship it.

So this repo is config-driven end to end — every experiment is a YAML diff — and every
claim about speed, memory or size is produced by a script in `benchmarks/` or
`src/evaluation/`, written to a JSON artifact, and reproducible with one command.

## What it does

```
   dataset (HF Hub)                     QLoRA fine-tuning                  artifact
┌────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
│ Dolly-15k (text)   │──▶│ 4-bit nf4 base model         │──▶│ LoRA adapter         │
│ The Cauldron (VLM) │   │ + LoRA on attn/MLP proj      │   │ (final_model/)       │
└────────────────────┘   │ + cosine LR, paged AdamW 8bit│   └──────────┬───────────┘
                         └──────────────────────────────┘              │
                                                                       ▼
                                                        ┌──────────────────────────┐
                                                        │ Merge adapter into base  │
                                                        │ → outputs/merged_model   │
                                                        └────────────┬─────────────┘
                                                                     │
              ┌──────────────────────────────┬───────────────────────┴──────────────┐
              ▼                              ▼                                      ▼
    ┌──────────────────┐          ┌──────────────────┐                  ┌──────────────────┐
    │ bitsandbytes 4bit│          │ GGUF (llama.cpp) │                  │ AWQ 4-bit        │
    │ nf4, GPU dev     │          │ Q4_K_M / Q5_K_M  │                  │ GPU throughput   │
    │                  │          │ Q8_0, CPU/edge   │                  │ (see limitations)│
    └────────┬─────────┘          └────────┬─────────┘                  └────────┬─────────┘
             └─────────────────────────────┼─────────────────────────────────────┘
                                           ▼
                             ┌───────────────────────────┐
                             │ Evaluation report         │
                             │ perplexity · memory · t/s │
                             └─────────────┬─────────────┘
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │ FastAPI /generate/*  +  Gradio demo UI    │
                     │ Docker image, one-command deployment      │
                     └───────────────────────────────────────────┘
```

| Capability | Implementation |
|---|---|
| Fine-tuning | QLoRA (4-bit nf4) on LLMs **and** VLMs; trains ~0.2–1.2% of weights (measured below) |
| Adapter merge | `merge_and_unload()` → standalone checkpoint |
| Quantization | bitsandbytes 4-bit, GGUF (Q4_K_M / Q5_K_M / Q8_0), AWQ |
| Measurement | Perplexity, LoRA budget, load time, peak RSS, p50/p95 latency, decode tok/s, artifact size |
| Serving | FastAPI with Pydantic schemas, `/docs`, `/health`, text + vision routes |
| Demo | Gradio UI with separate LLM and VLM tabs |
| Packaging | Dockerfile (CUDA base) + docker-compose for API and demo |
| Ops | Config + metric unit tests, GitHub Actions CI, structured logging, W&B tracking |

---

## Measurements

All values below are produced by scripts in this repo and saved under
`benchmarks/results/*.json` (committed, so the numbers are checkable rather than claimed).

### LoRA parameter budget

`benchmarks/lora_budget.py` instantiates the architecture on PyTorch's meta device — no
weights downloaded — and counts the `A`/`B` matrices LoRA adds per wrapped layer.

| Model (config) | Total params | Wrapped layers | Trainable | Share | Adapter size (bf16) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (`llm_finetune.yaml`) | 1,543,714,304 | 196 | 18,464,768 | **1.196%** | 35.2 MB |
| Qwen2-VL-2B-Instruct (`vlm_finetune.yaml`) | 2,208,985,600 | 112 | 4,358,144 | **0.197%** | 8.3 MB |

This is the number that makes QLoRA fit a single consumer GPU: 98.8% of the 1.5B model
stays frozen, and the thing you upload, version and hot-swap is a ~35 MB adapter.

### Serving baseline on CPU (measured on this machine)

Qwen2.5-0.5B-Instruct, greedy decoding, 12 logical cores, torch 2.15.0+cpu, Windows.
Three prompts, timings after one warmup pass.

| dtype | Load time | Peak RSS | Decode throughput | p50 latency | Perplexity (4 held-out texts) |
|---|---|---|---|---|---|
| float32 | 37.9 s | 2362.6 MB | **11.07 tok/s** | 5733.8 ms (64 new tokens) | 27.24 |
| bfloat16 | 4.5 s | **1375.6 MB** | 7.20 tok/s | 2231.0 ms (16 new tokens) | — |

Two things worth reading off this table:

- **Memory: −42%** for half precision (2362.6 → 1375.6 MB). That is the whole argument for
  quantization on constrained hardware, and quantization goes further still.
- **Speed: CPU bf16 is ~1.5× slower per token** (139 ms/token vs 90 ms/token). Half
  precision is a *memory* optimization on CPU, not a speed one — the speed win needs
  GPU tensor cores or GGUF/llama.cpp's CPU kernels. Measuring instead of assuming is what
  caught that.

### Quantization artifacts (GPU reference run — not reproduced here)

`Qwen2.5-1.5B-Instruct`, Dolly-15k (500 samples, 1 epoch), LoRA r=16 / alpha=32,
effective batch size 8, LR 2e-4 cosine.

**Perplexity, base vs. fine-tuned (lower is better)**

| Model | Perplexity |
|---|---|
| Base | 15.14 |
| QLoRA fine-tuned | 7.49 |
| Change | **−50.6%** |

**Artifact size**

| Artifact | Size | vs FP16 | When to use |
|---|---|---|---|
| Merged FP16 | 2.96 GB | — | Reference / re-quantization source |
| GGUF Q8_0 | 1.57 GB | −47% | CPU serving, near-lossless |
| GGUF Q5_K_M | 1.07 GB | −64% | Balanced CPU/edge |
| GGUF Q4_K_M | 0.94 GB | −68% | Maximum compression, CPU/edge |
| bitsandbytes 4-bit | on-the-fly | — | Fast GPU dev loop (no artifact on disk) |

> This table needs a CUDA GPU (bitsandbytes + llama.cpp), so it was not reproduced on the
> CPU-only machine used for the sections above. `evaluate.py` now writes
> `reduction_vs_fp16_percent` for every artifact, plus a `serving` block with load time,
> peak RSS, p50/p95 and tok/s — run it on your hardware to regenerate these rows.

### What each metric means

| Metric | How it's measured | Why it's in the report |
|---|---|---|
| Trainable share | LoRA params ÷ total params from the architecture | Predicts VRAM headroom and adapter size |
| Perplexity | `exp(mean token loss)` on held-out instruction samples | Detects a fine-tune that made the model worse |
| Load time | Wall clock around `from_pretrained` + `.to(device)` | Cold-start cost for autoscaled serving |
| Peak RSS | `psutil` resident memory of the process | Decides whether the model fits the box |
| p50 / p95 latency | Per-request timings, linear-interpolated percentile | Mean hides tail latency; p95 is what users feel |
| Decode throughput | Generated tokens ÷ generation seconds | Capacity planning, cost per 1k tokens |
| Artifact size | Recursive directory / file size on disk | What you ship, cache and download |
| Size reduction | `(reference − candidate) / reference` | Quantization decision, per deployment target |

### Reproduce any of it

```bash
pip install -r requirements.txt

# LoRA budget for either config (no weights downloaded, runs on CPU)
python benchmarks/lora_budget.py --config configs/llm_finetune.yaml

# Serving benchmark for any HF model or local checkpoint (CPU or GPU)
python benchmarks/benchmark_model.py --model Qwen/Qwen2.5-0.5B-Instruct --runs 3 --max-new-tokens 64
python benchmarks/benchmark_model.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 --runs 1 --max-new-tokens 16

# Full report once a fine-tune exists: perplexity + serving + artifact sizes
python src/evaluation/evaluate.py --config configs/quantization.yaml
```

Each run writes a timestamped JSON artifact with the exact environment (device, dtype,
torch/transformers versions, thread count, CPU/GPU model) next to the numbers.

---

## Quick start

### Serve the API (no GPU needed to boot)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]" pydantic pillow pyyaml

uvicorn api.main:app --port 8000
# → http://localhost:8000/docs
```

The inference stack is imported lazily, so the server always starts. Without a
checkpoint loaded, `/health` reports `model_not_loaded` and generation routes answer
`503` with a clear message instead of crashing on import.

### Full pipeline (CUDA GPU)

```bash
pip install -r requirements.txt        # torch + transformers + peft + bitsandbytes
cp .env.example .env                  # optional: WANDB key, HF token

python src/training/train_llm.py --config configs/llm_finetune.yaml
python src/quantization/quantize_bnb.py --config configs/quantization.yaml   # merges + 4-bit
python src/quantization/quantize_gguf.py --config configs/quantization.yaml
python src/evaluation/evaluate.py --config configs/quantization.yaml

# serve the merged checkpoint
MODEL_TYPE=llm MODEL_PATH=outputs/merged_model uvicorn api.main:app --port 8000
python demo/app.py                    # Gradio UI on :7860
```

### Docker

```bash
docker compose -f docker/docker-compose.yaml up --build   # API :8000, demo :7860
```

---

## Usage

**Fine-tune a VLM instead of an LLM**

```bash
python src/training/train_vlm.py --config configs/vlm_finetune.yaml
```

**Generate from the API**

```bash
curl -X POST http://localhost:8000/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain QLoRA in two sentences.", "max_new_tokens": 128, "temperature": 0.7}'
```

```json
{
  "generated_text": "QLoRA fine-tunes a frozen 4-bit model by training small low-rank adapters...",
  "prompt": "Explain QLoRA in two sentences.",
  "latency_ms": 1842.3
}
```

**Vision route** — `POST /generate/vision` with `image_base64` plus a question.

**Run the checks**

```bash
pip install pyyaml pytest
pytest tests -q
```

---

## Project structure

```
llm-qlora-finetuning-pipeline/
├── configs/                  # single source of truth for every setting
│   ├── llm_finetune.yaml
│   ├── vlm_finetune.yaml
│   └── quantization.yaml
├── data/dataset_loader.py    # LLM instruction + VLM image-text preprocessing
├── src/
│   ├── training/             # lora_config.py, train_llm.py, train_vlm.py
│   ├── quantization/         # quantize_bnb.py, quantize_gguf.py, quantize_awq.py
│   ├── evaluation/           # evaluate.py (report) + metrics.py (measurement primitives)
│   ├── inference/inference_engine.py
│   └── utils/                # config_loader.py, logger.py
├── benchmarks/
│   ├── benchmark_model.py    # load time, memory, latency, tok/s, perplexity
│   ├── lora_budget.py        # exact trainable-parameter accounting
│   └── results/              # committed JSON artifacts behind the numbers above
├── api/                      # FastAPI app + Pydantic schemas
├── demo/app.py               # Gradio UI
├── docker/                   # Dockerfile + docker-compose
├── tests/                    # config validation + metric unit tests
├── .github/workflows/ci.yml
└── requirements.txt
```

## Design decisions

- **Config over code.** Every model id, path and hyperparameter lives in YAML, validated
  by `tests/test_configs.py` before a run starts. A bad config fails in seconds, not after
  a 20-minute model download.
- **Measure, don't estimate.** Percentiles, memory and throughput live in one module
  (`src/evaluation/metrics.py`) used by both the benchmark script and the evaluation
  report, so the same definition can't drift between them — and the pure helpers are
  unit-tested in CI without torch installed.
- **Report the environment with the number.** A tok/s figure without the device, dtype and
  library versions isn't reproducible; every artifact carries a fingerprint.
- **One inference interface.** `UnifiedInferenceEngine` hides the LLM/VLM split, so the
  API and demo talk to one object regardless of model type.
- **Lazy heavy imports in the API.** `torch`/`transformers` load only when a model is
  actually configured, which keeps the container startable and the health check meaningful
  during deploys.
- **Logging, not prints.** `get_logger()` writes timestamped console output *and* a
  per-run file under `logs/`, so training failures are debuggable after the fact.

## Known limitations

- **CPU numbers are CPU numbers.** The measured table is single-machine, 12 logical cores,
  no GPU. GPU throughput, VRAM use and the quantization comparison need their own run.
- **AWQ is implemented but disabled.** `autoawq` breaks against recent `transformers`
  releases (`Qwen2Attention.forward` signature change). `configs/quantization.yaml` ships
  with `awq.enabled: false`; the script is complete and will work once the library catches up.
- **Small training subset by default.** `max_train_samples: 500` keeps a demo run short.
  Set it to `null` for a full Dolly-15k epoch.
- **bitsandbytes produces no on-disk artifact** — it quantizes at load time, so its size
  is not comparable to the GGUF files.
- **Base-model prompt format.** Training and inference both use the `### Instruction:` /
  `### Response:` template; switching base models means updating both sides.
- **Perplexity is a coarse signal.** It catches regressions, not instruction-following
  quality. Task benchmarks are still on the roadmap.

## Roadmap

- [ ] vLLM backend for high-throughput serving, benchmarked against the current engine
- [ ] Task benchmarks (MMLU-style accuracy, VQA accuracy) alongside perplexity
- [ ] Multi-GPU / larger-model benchmark matrix checked into `benchmarks/results/`
- [ ] Kubernetes manifests and image-push CI job
- [ ] Re-enable AWQ once `autoawq` ships a compatible release

## Tech stack

PyTorch · HuggingFace Transformers · PEFT · bitsandbytes · TRL · llama.cpp (GGUF) ·
AutoAWQ · FastAPI · Gradio · Docker · Weights & Biases · GitHub Actions

## License

MIT — see [LICENSE](LICENSE).

## Author

**Vineeth** — [github.com/Vineeth-4](https://github.com/Vineeth-4)

If this saved you time, a ⭐ helps.
