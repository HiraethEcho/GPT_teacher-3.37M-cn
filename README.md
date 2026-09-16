# GPT Teacher

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20MPS%20%7C%20CUDA-lightgrey)

中文 | [English](README_EN.md)

从 0 到 1 在 CPU 上训练可推理可演示的 3.37M 参数中文 GPT——自带 Transformer 透视镜，训练出来的不是黑盒。

![模型结构](train/docs/model_structure.png)

## 效果一览

| 指标 | 结果 |
|------|------|
| 参数量 | 3.37M（含权重共享），4 层 Decoder-only Transformer |
| 训练耗时 | MPS 3-6 分钟 / CUDA 5-10 分钟 / CPU 30-60 分钟 |
| 验收测试 | 6/6 通过（`uv run python -m core.evaluate`） |
| 行为回归 | 27 条金样本（`uv run python train/scripts/regression.py`） |
| 单元测试 | 56 个（`uv run pytest core/tests/`） |

生成样例（temperature=0）：

```text
问: 什么是注意力机制？
答: 注意力机制通过计算查询和键的相关性分配权重，让模型动态关注最相关的部分。
```

## 核心特性

- **一键跑通**：分词器 → 训练 → 验收 → Web Demo，一条命令完成
- **🔍 Transformer 透视镜**：Web Demo 里看自己模型的真实内部——管道总览（token 色块 → 逐层注意力 → 下一个字的 top-5 预测）、干预实验（跳层 / 层数 / 注意力温度三联对照）、逐头热力图，数据全部来自 `best.pt` 的真实前向
- **中文 ByteLevel BPE 分词器**：从零训练，token 分布与词表使用可视化

  ![分词器可视化](train/docs/tokenizer_visualization.png)

- **ONNX 导出**：用 [Netron](https://netron.app) 打开自己模型的完整计算图
- **多轮对话**：训练集含两轮对话样本，Demo 支持带历史的连续提问
- **金样本回归**：每次改动量化模型行为变化，防止隐性退化

## 快速开始

```bash
# 安装依赖（需要 Python 3.10-3.12）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv sync

# 一键跑通：分词器 → 训练 → 验收 → Web Demo
uv run python train/run.py
```

详细使用说明见各模块 README：

- [train/README.md](train/README.md) — 训练、验收、推理、透视镜、ONNX 导出、Web Demo 的完整教程
- [distill/README.md](distill/README.md) — 知识蒸馏三种方向的实验记录

## 项目结构

```
GPT_teacher-3.37M-cn/
├── core/       ← 共享库（GPT 模型、分词器、推理引擎、可视化）
├── train/      ← 训练模块（数据、配置、checkpoint、Web Demo）→ [README](train/README.md)
├── distill/    ← 蒸馏模块（知识蒸馏训练与推理）→ [README](distill/README.md)
├── pyproject.toml
└── uv.lock
```

| 模块 | 说明 | 快速开始 |
|------|------|----------|
| **train** | 3.37M 参数中文 GPT 的完整训练流程 | `uv run python train/run.py` |
| **distill** | 用大模型做 teacher 的知识蒸馏实验 | `uv run python -m distill.train --kd --teacher Qwen/Qwen2.5-1.5B-Instruct` |
| **core** | 共享代码：模型定义、分词器、推理引擎、评估、可视化 | 被上面两个模块引用 |

## 技术栈

- 架构：Decoder-only Transformer（GQA + SwiGLU + RMSNorm + RoPE + 权重共享）
- 设备：CPU / MPS (Apple Silicon) / CUDA (NVIDIA GPU)
- 依赖管理：uv

## License

[MIT](LICENSE) © 2025 少年唐
