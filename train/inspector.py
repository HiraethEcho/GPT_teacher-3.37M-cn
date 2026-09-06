"""Transformer 透视镜：管道总览渲染 + 干预实验（跳层 / 用到前 N 层 / 注意力温度）。

干预通过 forward hook 临时替换模块输出实现，不改动模型本体；跑完即摘除。
所有渲染基于 matplotlib，输出 Figure 供 gradio 展示。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.visualize import display_token

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

if TYPE_CHECKING:
    import torch

    from core.model import GPT
    from core.tokenizer import TokenizerLike


def encode_question(model: GPT, tok: TokenizerLike, question: str) -> tuple[list[int], list[str]]:
    """把问题编码成模型输入：与推理同款包装 + 超长截断保护。

    Args:
        model: GPT 模型（提供 seq_len 上限）。
        tok: 分词器。
        question: 用户输入的文本。

    Returns:
        (token ids, token 文本标签)。
    """
    assert tok.bos_id is not None, "分词器缺少 BOS 特殊 token"
    prefix = [tok.bos_id, *tok.encode("用户:" + question.strip() + "\n助手:", add_special_tokens=False)]
    prefix = prefix[-model.seq_len :]  # 与 generate 相同的截断策略：保留最近上下文
    tokens = [display_token(tok.decode([tid]), tid) for tid in prefix]
    return prefix, tokens


def run_intervention(
    model: GPT, token_ids: torch.Tensor, skip_layer: int, depth: int, attn_temp: float
) -> torch.Tensor:
    """带干预的前向：跳过指定层 / 只用前 N 层 / 注意力温度缩放。

    全部用 forward hook 实现（跳层 = 整个 Block 输出替换为输入；温度 = 重算
    softmax(QK^T/温度) 后的注意力输出），对模型本体零改动，跑完摘除。

    Args:
        model: 已加载权重的 GPT 模型。
        token_ids: 输入序列 [1, T]。
        skip_layer: 跳过的层（1 起），0 表示不跳。
        depth: 只用前 depth 层（1 起），不小于模型层数时全用。
        attn_temp: 注意力温度，softmax(score/温度)；1.0 不干预，
            >1 更分散（钝化），<1 更尖锐（锐化）。

    Returns:
        干预后的 logits [1, T, vocab]。
    """
    import torch

    from core.model import rope as apply_rope

    n_layer = len(model.blocks)
    skipped = set()
    if 1 <= skip_layer <= n_layer:
        skipped.add(skip_layer - 1)
    skipped |= set(range(depth, n_layer))

    def skip_hook(module: Any, inputs: Any, output: Any) -> tuple[Any, None]:
        return inputs[0], None  # Block 输出替换为输入 = 该层（含残差）整体跳过

    def make_temp_hook(temp: float) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> tuple[Any, None]:
            h = inputs[0]
            B, T, _ = h.shape
            q = module.wq(h).view(B, T, module.n_head, module.head_dim)
            k = module.wk(h).view(B, T, module.n_kv_head, module.head_dim)
            v = module.wv(h).view(B, T, module.n_kv_head, module.head_dim)
            q, k = apply_rope(q, k, T, module.head_dim, h.device)
            k = module._repeat_kv(k)
            v = module._repeat_kv(v)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * (module.head_dim**-0.5) / temp
            causal = torch.tril(torch.ones(T, T, device=h.device))
            scores = scores.masked_fill(causal == 0, float("-inf"))
            y = torch.softmax(scores, dim=-1) @ v
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
            return module.proj(y), None

        return hook

    handles = []
    for i, blk in enumerate(model.blocks):
        if i in skipped:
            handles.append(blk.register_forward_hook(skip_hook))
        elif abs(attn_temp - 1.0) > 1e-9:
            handles.append(blk.attn.register_forward_hook(make_temp_hook(attn_temp)))
    try:
        with torch.no_grad():
            logits, _ = model(token_ids)
    finally:
        for h in handles:
            h.remove()
    result: torch.Tensor = logits
    return result


def top_tokens(logits: torch.Tensor, tok: TokenizerLike, k: int = 5) -> list[tuple[str, float]]:
    """取最后一个位置的 top-k 预测 token 及其概率。

    Args:
        logits: 模型输出 [1, T, vocab]。
        tok: 分词器。
        k: 返回数量。

    Returns:
        [(token 文本, 概率)]，按概率降序。
    """
    import torch

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    values, indices = torch.topk(probs, k)
    out = []
    for p, i in zip(values.tolist(), indices.tolist(), strict=True):
        t = display_token(tok.decode([i]), i)
        if len(t) > 6:  # BPE 长词组做标签会撑爆条形图版面
            t = t[:5] + "…"
        out.append((t, p))
    return out


def find_focus_token(matrix: np.ndarray[Any, Any]) -> tuple[int, float]:
    """找出「最被盯的字」：对每列求和（被关注度），取被关注度最大的列。

    排除两类位置：第 0 位是 BOS 起始符——模型会把多余注意力倾泻在它身上
    （attention sink 现象），教学上没有"关键词"含义；最后一位是正在生成
    的位置，它是注意力的消费方而非供给方。

    Args:
        matrix: 注意力矩阵 [T, T]（通常取聚焦最明显的中层、各头平均）。

    Returns:
        (列位置, 该列被关注度总和)。
    """
    col_sum = matrix.sum(axis=0)
    candidates = col_sum[1:-1] if len(col_sum) > 2 else col_sum
    offset = 1 if len(col_sum) > 2 else 0
    focus = offset + int(np.argmax(candidates))
    return focus, float(col_sum[focus])


def _barh(ax: Axes, pairs: list[tuple[str, float]], title: str, color: str) -> None:
    """画一组水平概率条。"""
    ys = np.arange(len(pairs))[::-1]
    ax.barh(ys, [p for _, p in pairs], color=color)
    ax.set_yticks(ys)
    ax.set_yticklabels([t for t, _ in pairs], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, max(0.05, max(p for _, p in pairs)))
    ax.tick_params(axis="x", labelsize=7)


def render_pipeline(
    tokens: list[str],
    attn_per_layer: list[np.ndarray[Any, Any]],
    baseline: list[tuple[str, float]],
    intervened: list[tuple[str, float]],
    skipped_layers: set[int],
    interventions: str,
    focus: tuple[int, float],
) -> Figure:
    """渲染管道总览图：token → Embed → 各层（内嵌注意力缩略图）→ 输出概率对比。

    Args:
        tokens: token 文本标签。
        attn_per_layer: 每层各头平均注意力 [T, T]。
        baseline: 无干预 top-k 预测。
        intervened: 干预后 top-k 预测。
        skipped_layers: 被跳过的层索引集合（0 起）。
        interventions: 干预描述文案（如 "跳过第 2 层"），无干预时为空串。
        focus: (最被盯的字的位置, 被关注度)。

    Returns:
        matplotlib Figure。
    """
    n_layers = len(attn_per_layer)
    fig = plt.figure(figsize=(16.5, 6.5))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    title = "Transformer 透视镜：数据从左到右流过每一层"
    if interventions:
        title += f"（当前干预：{interventions}）"
    ax.text(0.2, 6.25, title, fontsize=14, fontweight="bold", va="top")

    # 顶部：token 色块
    cmap = plt.cm.Set3  # type: ignore[attr-defined]
    x = 0.2
    for i, t in enumerate(tokens[:12]):
        w = max(len(t) * 0.28, 0.55)
        rect = mpatches.FancyBboxPatch(
            (x, 5.35), w, 0.5, boxstyle="round,pad=0.04", facecolor=cmap(i % 12), edgecolor="gray", linewidth=0.6
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 5.6, t, ha="center", va="center", fontsize=8)
        if i == focus[0]:
            ax.plot([x + w / 2], [5.95], marker="v", color="red", markersize=8)
        x += w + 0.1
    ax.text(0.2, 5.12, f"输入切成 {len(tokens)} 个 token（▼ 标记最被盯的字）", fontsize=8, color="gray")

    # Embed 条
    emb = mpatches.FancyBboxPatch(
        (0.25, 1.6), 0.7, 3.2, boxstyle="round,pad=0.05", facecolor="lightgreen", edgecolor="gray"
    )
    ax.add_patch(emb)
    ax.text(0.6, 3.2, "Embed\n256 维", ha="center", va="center", fontsize=8)

    # 各层方块 + 内嵌注意力缩略图
    for li in range(n_layers):
        x0 = 1.5 + li * 2.0
        skipped = li in skipped_layers
        face = "0.85" if skipped else "white"
        box = mpatches.FancyBboxPatch(
            (x0, 1.6),
            1.8,
            3.2,
            boxstyle="round,pad=0.05",
            facecolor=face,
            edgecolor="gray",
            linestyle="--" if skipped else "-",
            alpha=0.45 if skipped else 1.0,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.9, 4.5, f"第 {li + 1} 层", ha="center", fontsize=10, fontweight="bold")
        if skipped:
            ax.text(x0 + 0.9, 2.0, "已跳过", ha="center", fontsize=10, color="red", fontweight="bold")
        sub = fig.add_axes((x0 / 15 + 0.008, 2.35 / 6.5, 1.6 / 15, 1.6 / 6.5))
        sub.imshow(attn_per_layer[li], cmap="Blues", vmin=0)
        sub.axis("off")
        if li < n_layers - 1:
            ax.annotate(
                "",
                xy=(x0 + 2.0, 3.2),
                xytext=(x0 + 1.82, 3.2),
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5},
            )
    ax.annotate("", xy=(1.52, 3.2), xytext=(1.0, 3.2), arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5})

    # 输出概率：干预时原版/干预后并排对比，无干预时单图
    out_x = 1.5 + n_layers * 2.0
    ax.annotate(
        "",
        xy=(out_x + 0.02, 3.2),
        xytext=(out_x - 0.16, 3.2),
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5},
    )
    if intervened != baseline:
        _barh(fig.add_axes((0.615, 0.30, 0.085, 0.40)), baseline, "原版预测", "#4C72B0")
        _barh(fig.add_axes((0.78, 0.30, 0.085, 0.40)), intervened, "干预后预测", "#DD8452")
    else:
        _barh(fig.add_axes((0.62, 0.30, 0.16, 0.40)), baseline, "下一个字的预测概率", "#4C72B0")

    # 自动结论
    focus_tok = tokens[focus[0]] if focus[0] < len(tokens) else "?"
    conclusion = f"自动发现：「{focus_tok}」是最被盯的字——后面的字都在它身上找线索（总被关注度 {focus[1]:.1f}）。"
    if interventions:
        conclusion += (
            f"\n干预（{interventions}）后预测概率重排：对比左右两张条形图，看模型没了这层/变了注意力后「想法」怎么变。"
        )
    ax.text(
        0.25,
        1.3,
        conclusion,
        fontsize=9,
        color="darkblue",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    return fig
