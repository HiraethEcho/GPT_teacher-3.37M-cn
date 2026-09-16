# GPT Teacher

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20MPS%20%7C%20CUDA-lightgrey)

[中文](README.md) | English

Train a 3.37M-param Chinese GPT from scratch on CPU — with inference, a live demo, and a Transformer inspector so the model you train is never a black box.

![Model structure](train/docs/model_structure.png)

## Results at a Glance

| Metric | Result |
|--------|--------|
| Parameters | 3.37M (weight tying included), 4-layer decoder-only Transformer |
| Training time | MPS 3-6 min / CUDA 5-10 min / CPU 30-60 min |
| Acceptance test | 6/6 passed (`uv run python -m core.evaluate`) |
| Behavioral regression | 27 golden samples (`uv run python train/scripts/regression.py`) |
| Unit tests | 56 (`uv run pytest core/tests/`) |

Sample generation (temperature=0):

```text
Q: 什么是注意力机制？ (What is the attention mechanism?)
A: 注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。
   (Attention computes relevance between queries and keys to weight features,
   letting the model dynamically focus on the most relevant parts.)
```

## Key Features

- **One-command pipeline**: tokenizer → training → evaluation → web demo
- **🔍 Transformer Inspector**: see inside your own model in the web demo — pipeline overview (token chips → per-layer attention → top-5 next-token prediction), intervention experiments (skip layer / first-N layers / attention-temperature triptych), and per-head heatmaps. All data comes from real forward passes of `best.pt`
- **Chinese ByteLevel BPE tokenizer**: trained from scratch, with token distribution and vocabulary usage visualization

  ![Tokenizer visualization](train/docs/tokenizer_visualization.png)

- **ONNX export**: open the full computation graph of your own model in [Netron](https://netron.app)
- **Multi-turn dialogue**: training set includes two-turn samples; the demo supports continuous questioning with history
- **Golden-sample regression**: every change is measured against 27 golden samples to catch silent behavioral drift

## Quick Start

```bash
# Install dependencies (Python 3.10-3.12 required)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv sync

# One command: tokenizer → train → evaluate → web demo
uv run python train/run.py
```

**Expected result**: `Result: 6/6 passed (100%)` — then open http://127.0.0.1:7860 in your browser.

See the module READMEs (in Chinese) for details:
- [train/README.md](train/README.md) — full tutorial: training, evaluation, inference, inspector, ONNX export, web demo
- [distill/README.md](distill/README.md) — experiment log: three distillation directions, all failed

## Project Structure

```
GPT_teacher-3.37M-cn/
├── core/       ← Shared library (GPT model, tokenizer, inference, visualization)
├── train/      ← Training module (data, config, checkpoints, web demo) → [README](train/README.md)
├── distill/    ← Knowledge distillation experiments → [README](distill/README.md)
├── pyproject.toml
└── uv.lock
```

| Module | Description | Quick Start |
|--------|-------------|-------------|
| **train** | Complete training pipeline for a 3.37M-param Chinese GPT | `uv run python train/run.py` |
| **distill** | Knowledge distillation experiments with a large-model teacher | `uv run python -m distill.train --kd --teacher Qwen/Qwen2.5-1.5B-Instruct` |
| **core** | Shared code: model, tokenizer, inference, evaluation, visualization | Used by the two modules above |

## Tech Stack

- Architecture: Decoder-only Transformer (GQA + SwiGLU + RMSNorm + RoPE + weight tying)
- Devices: CPU / MPS (Apple Silicon) / CUDA (NVIDIA GPU)
- Package management: uv

## License

[MIT](LICENSE) © 2025 少年唐
