"""Web Demo：Gradio 交互演示（多轮对话 + 置信度 + 自洽性检测）。"""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

import random
import socket
import subprocess
import time
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
# gradio_client get_type() crashes when additionalProperties is bool instead of dict
import gradio as gr
import gradio_client.utils as _gcu
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure

from core.infer import generate, load_model_and_tokenizer
from core.model import GPT
from core.tokenizer import TokenizerLike
from core.visualize import capture_attention_weights
from train.inspector import (
    encode_question,
    find_focus_token,
    placeholder_figure,
    render_pipeline,
    render_temperature_compare,
    run_intervention,
    top_tokens,
)

_orig_get_type = _gcu.get_type


def _safe_get_type(schema: Any) -> str:
    """gradio_client 兼容补丁：schema 非字典时返回 "null"。"""
    if not isinstance(schema, dict):
        return "null"
    t: str = _orig_get_type(schema)
    return t


_gcu.get_type = _safe_get_type

FALLBACK_ANSWERS = [
    "这个问题我还不会，再学几年吧~",
    "emmm...我还没学到这个，换个问题试试？",
    "我只是一个 ~3.37M 的小模型，这个问题超纲了！",
    "呃...你问住我了，我回去好好学学再来回答。",
    "这个问题太难了，我选择卖萌 (◕ᴗ◕✿)",
]


DEFAULT_CKPT = "train/checkpoints/best.pt"


def _read_model_limits() -> tuple[int, int]:
    """读 config.yml 的层数/头数做滑条上限，避免拖到不存在的层靠 clamp 兜底。

    读不到（无 config.yml / 字段缺失）时退回宽松上限 16，clamp 逻辑仍兜底。

    Returns:
        (n_layer, n_head)。
    """
    try:
        with open("train/config.yml", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        return int(cfg["model"]["n_layer"]), int(cfg["model"]["n_head"])
    except (OSError, KeyError, ValueError, yaml.YAMLError):
        return 16, 16


N_LAYER, N_HEAD = _read_model_limits()

model: GPT | None = None
tokenizer: TokenizerLike | None = None
device: torch.device | None = None
model_info = ""


def ensure_model() -> bool:
    """懒加载模型：首次调用时从 DEFAULT_CKPT 恢复。

    Returns:
        模型是否可用（checkpoint 不存在时返回 False）。
    """
    global model, tokenizer, device, model_info
    if model is None:
        if not os.path.exists(DEFAULT_CKPT):
            return False
        model, tokenizer, _, dev = load_model_and_tokenizer(DEFAULT_CKPT)
        device = dev
        n_params = sum(p.numel() for p in model.parameters())
        model_info = f"模型: {n_params / 1e6:.2f}M 参数 | 设备: {device}"
    return True


def build_multi_turn_prompt(history: list[Any], new_message: str) -> str:
    """将对话历史拼成模型能理解的格式：用户:Q1\\n助手:A1\\n用户:Q2\\n助手:

    跳过助手消息为 None/空的占位轮次（快捷问题按钮会预填 (问题, None) 等待回答的轮次，
    若拼入会生成训练数据中不存在的 "助手:None" 模式，导致模型答非所问）。
    """
    parts = []
    for user_msg, assistant_msg in history:
        if assistant_msg is None or assistant_msg == "":
            continue
        parts.append(f"用户:{user_msg}\n助手:{assistant_msg}")
    parts.append(f"用户:{new_message}\n助手:")
    return "\n".join(parts)


def do_generate(
    prompt: str,
    temperature: float,
    top_k: float,
    top_p: float,
    max_tokens: float,
    repeat_penalty: float,
) -> tuple[str, float, float | None]:
    """执行推理。

    Args:
        prompt: 完整提示词（多轮时已拼接历史）。
        temperature: 采样温度。
        top_k: top-k 采样保留数。
        top_p: nucleus 采样阈值。
        max_tokens: 最大生成长度。
        repeat_penalty: 重复惩罚系数。

    Returns:
        (回答文本, 耗时秒, 平均置信度)；空回答时置信度为 None。
    """
    assert model is not None and tokenizer is not None, "模型未加载"
    start = time.time()
    result = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repeat_penalty,
        stop_strings=["用户:", "\n用户", "。", "；"],
        device=device,
        wrap_prompt=False,  # prompt 已由 build_multi_turn_prompt 拼好，避免双重 "用户:" 包装
        return_confidence=True,
    )
    elapsed = time.time() - start
    text = result["text"].strip()
    if not text or text in ("。", "，", ".", ","):
        text = random.choice(FALLBACK_ANSWERS)
        return text, elapsed, None
    return text, elapsed, result["avg_confidence"]


def chat(
    message: str,
    history: list[Any],
    temperature: float,
    top_k: float,
    top_p: float,
    max_tokens: float,
    repeat_penalty: float,
) -> list[Any]:
    """多轮对话回调：拼接历史后生成回答。"""
    if not ensure_model():
        history = history or []
        history.append(
            [
                message,
                "错误：未找到模型文件 train/checkpoints/best.pt，请先运行训练。\n\n运行命令: uv run python -m train.train",
            ]
        )
        return history

    prompt = build_multi_turn_prompt(history or [], message)
    text, elapsed, conf = do_generate(prompt, temperature, top_k, top_p, max_tokens, repeat_penalty)

    history = history or []
    # 快捷问题按钮预填了 (message, None) 占位轮：更新它而不是追加，避免问题显示两次
    if history and history[-1][1] is None:
        history[-1] = [message, text]
    else:
        history.append([message, text])
    return history


def chat_simple(
    prompt: str, temperature: float, top_k: float, top_p: float, max_tokens: float, repeat_penalty: float
) -> tuple[str, str]:
    """单轮问答模式，用于旧版按钮兼容。"""
    if not ensure_model():
        return "错误：未找到模型文件 train/checkpoints/best.pt，请先运行训练。", ""

    text, elapsed, conf = do_generate(prompt, temperature, top_k, top_p, max_tokens, repeat_penalty)

    if conf is None:
        conf_label = "低"
        response = f"{text}\n\n---\n推理耗时: {elapsed:.2f}s | 置信度: N/A ({conf_label})"
    else:
        conf_label = "高" if conf > 0.8 else ("中" if conf > 0.5 else "低")
        response = f"{text}\n\n---\n推理耗时: {elapsed:.2f}s | 置信度: {conf:.0%} ({conf_label})"
    return response, model_info


def check_consistency(prompt: str, max_tokens: float, repeat_penalty: float) -> tuple[str, str]:
    """自洽性检测：同一问题采样 5 次，统计最常见回答的占比。"""
    if not ensure_model():
        return "错误：模型未加载", ""

    assert model is not None and tokenizer is not None, "模型未加载"
    start = time.time()
    n_samples = 5
    results = []
    for _ in range(n_samples):
        r = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            repetition_penalty=repeat_penalty,
            stop_strings=["用户:", "\n用户", "。", "；"],
            device=device,
        )
        results.append(r)
    elapsed = time.time() - start

    counter = Counter(results)
    most_common_text, most_common_count = counter.most_common(1)[0]
    ratio = most_common_count / n_samples

    if not most_common_text.strip() or most_common_text.strip() in ("。", "，", ".", ","):
        most_common_text = random.choice(FALLBACK_ANSWERS)

    if ratio >= 0.8:
        verdict = f"自洽性: {most_common_count}/{n_samples} 次一致 (高可信度)"
    elif ratio >= 0.5:
        verdict = f"自洽性: {most_common_count}/{n_samples} 次一致 (中可信度)"
    else:
        verdict = f"自洽性: {most_common_count}/{n_samples} 次一致 (低可信度)"

    detail = "\n".join(f"  第{i + 1}次: {r}" for i, r in enumerate(results))
    response = (
        f"{most_common_text}\n\n---\n{verdict}\n检测耗时: {elapsed:.2f}s ({n_samples}次采样)\n\n所有回答:\n{detail}"
    )
    return response, model_info


def _prepare_attention_input(question: str) -> tuple[list[int], list[str]] | None:
    """把问题编码成模型输入：与推理同款包装 + 超长截断保护。

    Args:
        question: 用户输入的文本。

    Returns:
        (token ids, token 文本标签)；模型未加载或输入为空时返回 None。
    """
    if not ensure_model() or not question.strip():
        return None
    assert model is not None and tokenizer is not None, "模型未加载"
    return encode_question(model, tokenizer, question)


def show_attention(question: str, layer: float, head: float) -> Figure | None:
    """逐头注意力热力图（Head View）：选定层与注意力头，看它关注了哪些词。

    Args:
        question: 待可视化的文本。
        layer: 层编号（1 起，超过层数时取最后一层）。
        head: 头编号（1 起，0 表示该层各头平均）。

    Returns:
        matplotlib 热力图；输入无效时返回 None。
    """
    prepared = _prepare_attention_input(question)
    if prepared is None:
        return None
    prefix, tokens = prepared
    assert model is not None, "模型未加载"
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    weights_per_layer = capture_attention_weights(model, x)

    li = min(int(layer), len(weights_per_layer)) - 1
    hi = min(int(head), weights_per_layer[li].shape[0])
    matrix: np.ndarray[Any, Any] = weights_per_layer[li].mean(axis=0) if hi == 0 else weights_per_layer[li][hi - 1]
    head_label = "各头平均" if hi == 0 else f"Head {hi}"

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), max(6.5, len(tokens) * 0.5)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=7, rotation=90)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("被盯的字（Key）")
    ax.set_ylabel("看的字（Query）")

    # 自动结论：红框标出「最被盯的字」所在列
    focus, focus_val = find_focus_token(matrix)
    focus_tok = tokens[focus] if focus < len(tokens) else "?"
    ax.add_patch(
        mpatches.Rectangle(
            (focus - 0.5, -0.5), 1, len(tokens), fill=False, edgecolor="red", linewidth=2, linestyle="--"
        )
    )
    ax.set_title(
        f"第 {li + 1} 层 · {head_label}：每个字在盯着谁看\n"
        f"红框：「{focus_tok}」最被盯（总关注度 {focus_val:.1f}）｜输入：{question.strip()}"
    )
    fig.colorbar(im, ax=ax, label="颜色越深 = 越被盯着看", shrink=0.8)

    # 三步读图指南
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.5,
        0.015,
        "怎么看这张图：① 在左边（Y 轴）选一个字 ② 沿这一行往右看 ③ 颜色越深的格子 = 这个字越被盯着看；"
        "右上角空白 = 只许看过去，不许看未来",
        ha="center",
        fontsize=8.5,
        color="gray",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "whitesmoke"},
    )
    return fig


def show_attention_grid(question: str) -> Figure | None:
    """全模型注意力总览（Model View）：层 x 头网格，一眼看出各头分工。

    Args:
        question: 待可视化的文本。

    Returns:
        matplotlib 网格图；输入无效时返回 None。
    """
    prepared = _prepare_attention_input(question)
    if prepared is None:
        return None
    prefix, tokens = prepared
    assert model is not None, "模型未加载"
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    weights_per_layer = capture_attention_weights(model, x)

    n_layers = len(weights_per_layer)
    n_heads = weights_per_layer[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers), squeeze=False)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            ax.imshow(weights_per_layer[li][hi], cmap="Blues", vmin=0)
            ax.set_title(f"L{li + 1} · H{hi + 1}", fontsize=9)
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, fontsize=4, rotation=90)
            ax.set_yticklabels(tokens, fontsize=4)
    fig.suptitle(f"逐头注意力总览：{n_layers} 层 x {n_heads} 头\n输入：{question.strip()}", fontsize=12)
    # 模式判读图例：三种典型图案各是什么意思
    fig.text(
        0.5,
        0.005,
        "图案怎么读：对角线深带 = 只看自己和邻居（浅层管局部顺序）｜竖直深列 = 大家盯着同一个关键词（聚焦）"
        "｜又浅又散 = 全局混合（深层做整合）",
        ha="center",
        fontsize=9,
        color="gray",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "whitesmoke"},
    )
    fig.subplots_adjust(bottom=0.08)
    return fig


def show_pipeline(question: str, depth: float, skip: float, temp: float) -> tuple[Figure, Figure]:
    """管道总览 + 干预实验：调整层的使用方式，看模型「想法」怎么变。

    一次回调跑两遍前向：原版 logits（基线）与干预后 logits，
    注意力缩略图带当前温度重算（调温度时热力图本身变平/变尖）。

    Args:
        question: 待透视的文本。
        depth: 只用前 N 层（超过层数 = 全用）。
        skip: 跳过第 N 层（0 = 不跳）。
        temp: 注意力温度（1.0 = 原样）。

    Returns:
        (管道总览图, 温度对照图)；对照图仅在温度干预时生成，否则给
        引导提示图（gradio Plot 空态的裂图图标看着像故障）。
    """
    temp_hint = "💡 把「注意力温度」拖离 1.0，这里会出现并排对照：每个字在盯着谁（弧线） + 变化量（差值）"
    prepared = _prepare_attention_input(question)
    if prepared is None:
        return placeholder_figure("输入文本后点「运行透视镜」"), placeholder_figure(temp_hint)
    prefix, tokens = prepared
    assert model is not None and tokenizer is not None, "模型未加载"
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits_base, _ = model(x)

    n_layer = len(model.blocks)
    d, s, t = min(int(depth), n_layer), min(int(skip), n_layer), float(temp)
    logits_new = run_intervention(model, x, s, d, t)

    baseline = top_tokens(logits_base, tokenizer)
    intervened = top_tokens(logits_new, tokenizer)
    # 层内缩略图带温度重算：调温度时热力图本身变平/变尖（肉眼可见），
    # 而不是只靠右侧概率条的微小变化
    attn_per_layer = [w.mean(axis=0) for w in capture_attention_weights(model, x, attn_temp=t)]

    skipped: set[int] = set()
    if 1 <= s <= n_layer:
        skipped.add(s - 1)
    skipped |= set(range(d, n_layer))

    parts = []
    if 1 <= s <= n_layer:
        parts.append(f"跳过第 {s} 层")
    if d < n_layer:
        parts.append(f"只用前 {d} 层")
    temp_active = abs(t - 1.0) > 1e-9
    if temp_active:
        parts.append(f"注意力温度 {t:.1f}")
    interventions = "、".join(parts)

    # 「最被盯的字」取中层（聚焦模式最明显的位置）
    mid = n_layer // 2
    focus = find_focus_token(attn_per_layer[mid])
    pipeline = render_pipeline(tokens, attn_per_layer, baseline, intervened, skipped, interventions, focus)

    # 温度对照图：并排放 T=1 与当前温度，消灭"单图渐变靠对比记忆"的感知负担；
    # 无温度干预时给引导提示（空态裂图图标看着像故障）
    compare: Figure
    if temp_active:
        base_attn = [w.mean(axis=0) for w in capture_attention_weights(model, x, attn_temp=1.0)]
        compare = render_temperature_compare(base_attn[mid], attn_per_layer[mid], tokens, mid, t)
    else:
        compare = placeholder_figure(temp_hint)
    return pipeline, compare


# 快捷问题
EXAMPLE_QUESTIONS = [
    "什么是注意力机制？",
    "RoPE 是什么？",
    "什么是机器学习？",
    "计算 15 乘以 6 是多少？",
    "你是谁？",
    "太阳系有哪些行星？",
    "蒸馏水和纯水有什么区别？",
    "权重共享有什么好处？",
]

with gr.Blocks(
    title="GPT Teacher 教学演示",
    css="""
        .example-btn { min-width: 120px; margin: 4px; }
    """,
) as demo:
    with gr.Tab("💬 对话演示"):
        gr.Markdown(
            "# GPT Teacher 教学演示\n"
            "这是一个 ~3.37M 参数的微型 GPT 模型，展示了 Transformer Decoder-only 架构的推理过程。\n\n"
            "**快速开始**: 点击下方问题按钮，或在对话框中输入自己的问题，支持多轮连续对话。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=400)
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="输入你想问的问题...",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

                with gr.Accordion("单轮模式（含置信度和自洽性检测）", open=False):
                    single_input = gr.Textbox(label="输入问题", placeholder="单轮提问...", lines=2)
                    single_output = gr.Textbox(label="模型回答", lines=4)
                    with gr.Row():
                        single_btn = gr.Button("提问", variant="primary")
                        consistency_btn = gr.Button("自洽性检测 (5次采样)")

            with gr.Column(scale=1):
                temperature = gr.Slider(
                    0.0,
                    1.5,
                    value=0.0,
                    step=0.1,
                    label="Temperature (温度)",
                    info="0=精确复制，1=有创造性",
                )
                top_k = gr.Slider(
                    1,
                    100,
                    value=50,
                    step=1,
                    label="Top-K",
                    info="只从概率最高的K个词中选",
                )
                top_p = gr.Slider(
                    0.0,
                    1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P (核采样)",
                    info="累积概率阈值",
                )
                with gr.Accordion("高级参数", open=False):
                    max_tokens = gr.Slider(
                        16,
                        256,
                        value=128,
                        step=16,
                        label="Max Tokens (最大生成长度)",
                        info="控制回答的最大长度",
                    )
                    repeat_penalty = gr.Slider(
                        1.0,
                        2.0,
                        value=1.5,
                        step=0.1,
                        label="Repetition Penalty (重复惩罚)",
                        info="防止模型重复说同样的话",
                    )
                model_info_box = gr.Markdown("模型尚未加载，点击「发送」自动加载")

        gr.Markdown("### 点击试试这些问题")
        with gr.Row():
            for q in EXAMPLE_QUESTIONS[:4]:
                gr.Button(q, elem_classes="example-btn").click(
                    lambda q=q, history=[]: (history + [(q, None)], q),
                    inputs=[],
                    outputs=[chatbot, msg_input],
                ).then(
                    chat,
                    [msg_input, chatbot, temperature, top_k, top_p, max_tokens, repeat_penalty],
                    [chatbot],
                ).then(
                    lambda: ensure_model() and model_info or "模型尚未加载",
                    inputs=[],
                    outputs=[model_info_box],
                ).then(lambda: "", inputs=[], outputs=[msg_input])
        with gr.Row():
            for q in EXAMPLE_QUESTIONS[4:]:
                gr.Button(q, elem_classes="example-btn").click(
                    lambda q=q, history=[]: (history + [(q, None)], q),
                    inputs=[],
                    outputs=[chatbot, msg_input],
                ).then(
                    chat,
                    [msg_input, chatbot, temperature, top_k, top_p, max_tokens, repeat_penalty],
                    [chatbot],
                ).then(
                    lambda: ensure_model() and model_info or "模型尚未加载",
                    inputs=[],
                    outputs=[model_info_box],
                ).then(lambda: "", inputs=[], outputs=[msg_input])

        # 多轮对话
        send_btn.click(
            chat,
            [msg_input, chatbot, temperature, top_k, top_p, max_tokens, repeat_penalty],
            [chatbot],
        ).then(
            lambda: ensure_model() and model_info or "模型尚未加载",
            inputs=[],
            outputs=[model_info_box],
        ).then(lambda: "", inputs=[], outputs=[msg_input])

        msg_input.submit(
            chat,
            [msg_input, chatbot, temperature, top_k, top_p, max_tokens, repeat_penalty],
            [chatbot],
        ).then(
            lambda: ensure_model() and model_info or "模型尚未加载",
            inputs=[],
            outputs=[model_info_box],
        ).then(lambda: "", inputs=[], outputs=[msg_input])

        clear_btn.click(
            lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, msg_input],
        )

        # 单轮模式
        single_btn.click(
            chat_simple,
            [single_input, temperature, top_k, top_p, max_tokens, repeat_penalty],
            [single_output, model_info_box],
        )
        single_input.submit(
            chat_simple,
            [single_input, temperature, top_k, top_p, max_tokens, repeat_penalty],
            [single_output, model_info_box],
        )
        consistency_btn.click(
            check_consistency,
            [single_input, max_tokens, repeat_penalty],
            [single_output, model_info_box],
        )

    with gr.Tab("🔍 Transformer 透视镜"):
        gr.Markdown(
            "看数据怎么流过模型的每一层，并且**动手干预**——跳过某层、只用到第 N 层、"
            "调注意力温度，看模型的「想法」（下一个字的预测概率）怎么变。\n"
            "数据全部来自你自己训练的 best.pt 的真实前向，不是示意图。"
        )
        # 默认问题选温度敏感度高的 OOD 生题（扫描实测 18.9 vs 熟题 ~4）：
        # 模型对生题的注意力更犹豫，温度干预的可见变化大，教学效果好
        ins_input = gr.Textbox(label="输入文本", value="如何学好英语？", lines=1)
        with gr.Row():
            ins_depth = gr.Slider(1, N_LAYER, value=N_LAYER, step=1, label="用到前 N 层", info="例如 2 = 只经过前 2 层")
            ins_skip = gr.Slider(
                0, N_LAYER, value=0, step=1, label="跳过某层", info="0 = 不跳过；试试跳过第 2 层（盯关键词的那层）"
            )
            ins_temp = gr.Slider(
                0.2, 3.0, value=1.0, step=0.1, label="注意力温度", info="1 = 原样；大于 1 更分散；小于 1 更尖锐"
            )
        ins_btn = gr.Button("运行透视镜", variant="primary")
        # 不设组件 label：gradio 会把它渲染在容器顶部，与占满画布的 figure 视觉重叠
        ins_plot = gr.Plot()
        # 温度对照图：仅在温度干预时出现（左=原样 右=当前，并排对比）
        ins_compare_plot = gr.Plot()
        # 平铺不折叠：gradio 4.25 的 Accordion 内 Plot 更新会触发整组重挂载、
        # open 状态丢失（点按钮后折叠区自己合上），教学页长一点无妨
        gr.Markdown(
            "### 逐层逐头细节（热力图）\n拖动滑条逐层逐头看注意力（即时刷新）；先点上面「运行透视镜」加载模型。"
        )
        with gr.Row():
            attn_layer = gr.Slider(1, N_LAYER, value=1, step=1, label="层", info="第 N 层")
            attn_head = gr.Slider(0, N_HEAD, value=0, step=1, label="注意力头", info="0 = 该层各头平均")
        with gr.Row():
            attn_btn = gr.Button("查看注意力", variant="primary")
            attn_grid_btn = gr.Button("总览：所有层 × 所有头")
        attn_plot = gr.Plot()

        # 透视镜：按钮 / 回车 / 干预滑条联动（拖动即时重跑）
        _ins_outputs = [ins_plot, ins_compare_plot]
        ins_btn.click(show_pipeline, [ins_input, ins_depth, ins_skip, ins_temp], _ins_outputs)
        ins_input.submit(show_pipeline, [ins_input, ins_depth, ins_skip, ins_temp], _ins_outputs)
        for _ctl in (ins_depth, ins_skip, ins_temp):
            _ctl.change(show_pipeline, [ins_input, ins_depth, ins_skip, ins_temp], _ins_outputs)

        # 逐头细节：滑条拖动即时重画，无需再点按钮
        attn_btn.click(show_attention, [ins_input, attn_layer, attn_head], [attn_plot])
        attn_layer.change(show_attention, [ins_input, attn_layer, attn_head], [attn_plot])
        attn_head.change(show_attention, [ins_input, attn_layer, attn_head], [attn_plot])
        attn_grid_btn.click(show_attention_grid, [ins_input], [attn_plot])

if __name__ == "__main__":
    port = 7860

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    if is_port_in_use(port):
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        print(f"\n端口 {port} 已被占用 (PID: {', '.join(pids)})")
        choice = input("是否终止占用进程？[y/N] ").strip().lower()
        if choice == "y":
            for pid in pids:
                os.kill(int(pid), 9)
            print(f"已终止 PID {', '.join(pids)}")
        else:
            print("退出。请手动释放端口后重试。")
            raise SystemExit(0)

    from datetime import datetime

    print("\n" + "=" * 50)
    print("  GPT Teacher Web Demo 启动成功！")
    print(f"  进程启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("  打开浏览器访问: http://127.0.0.1:7860")
    print("  按 Ctrl+C 停止服务")
    print('  （出现 "To create a public link..." 提示说明服务已在运行，')
    print("    那是 gradio 的例行提示，无需任何操作）")
    print("=" * 50 + "\n")
    try:
        demo.queue().launch(
            server_name=os.environ.get("SERVER_NAME", "127.0.0.1"),
            server_port=port,
            show_error=True,
            share=False,
        )
    except KeyboardInterrupt:
        # 第一次 Ctrl+C 由 gradio 接住并开始关闭（server 线程 join 最多阻塞数秒）；
        # 等待期间再按一次会从 launch() 冒出第二次 KeyboardInterrupt，这里接住
        # 避免把 traceback 甩给用户
        pass
    finally:
        print("\n  Web Demo 已停止。")
