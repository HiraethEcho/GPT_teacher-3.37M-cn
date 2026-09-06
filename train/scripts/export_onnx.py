"""导出 ONNX：把训练好的模型计算图交给 Netron 查看。

用法：
    uv run --with onnx python train/scripts/export_onnx.py
    uv run --with onnx --with onnxruntime python train/scripts/export_onnx.py  # 含数值验证

导出后用 Netron 打开（https://netron.app 网页拖入，或 pip install netron 本地起服务），
可以看到自己训练的模型的完整计算图：Embedding → 4 x (RMSNorm → GQA 注意力 →
残差 → RMSNorm → SwiGLU → 残差) → RMSNorm → LM Head。

说明：
    - 固定输入形状 [1, seq_len] 导出（计算图查看用途，非部署）；
    - 导出走 standard_attention（纯 matmul/softmax 算子，ONNX 完全覆盖），
      与推理时 Flash Attention 的差异仅在浮点求和顺序；
    - 不支持量化 checkpoint（量化算子不在 ONNX 标准算子集内）。
"""

import argparse
import warnings
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from core.infer import load_checkpoint, load_model_and_tokenizer
from core.model import GPT

# 固定形状导出是本脚本的设计（见模块 docstring），standard_attention 里
# `T > 1` 在 trace 时固化为常量属于预期行为，无需告警
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")


class LogitsOnly(nn.Module):
    """ONNX 导出包装：输入 token ids，输出 logits。

    GPT.forward 返回 (logits, kv_caches)，KV Cache 列表结构无法导出，
    这里只保留 teacher-forcing 前向（训练同款），计算图完整保留。
    """

    def __init__(self, gpt: GPT) -> None:
        """初始化包装层。

        Args:
            gpt: 已加载权重的 GPT 模型。
        """
        super().__init__()
        self.gpt = gpt

    def forward(self, idx: Tensor) -> Tensor:
        logits: Tensor = self.gpt(idx)[0]
        return logits


def export_onnx(ckpt_path: str, out_path: str, opset: int) -> tuple[LogitsOnly, GPT, int, int]:
    """从 checkpoint 加载模型并导出 ONNX。

    Args:
        ckpt_path: checkpoint 路径（不支持量化产物）。
        out_path: 输出 .onnx 路径。
        opset: ONNX opset 版本。

    Returns:
        (导出用的包装模型, 原模型, 词表大小, 序列长度)，供后续数值验证。

    Raises:
        SystemExit: 缺少 onnx 包、使用量化 checkpoint 或文件不存在。
    """
    try:
        import onnx  # noqa: F401  # torch 2.4 导出序列化 proto 时硬依赖 onnx
    except ImportError:
        raise SystemExit(
            "错误：导出 ONNX 需要 onnx 包。\n  完整命令: uv run --with onnx python train/scripts/export_onnx.py"
        ) from None

    ckpt = load_checkpoint(ckpt_path)
    if any("_packed_params" in k for k in ckpt["model"]):
        raise SystemExit("错误：量化 checkpoint 不支持导出 ONNX，请使用 best.pt")

    model, tok, _, _ = load_model_and_tokenizer(ckpt_path, device="cpu", use_flash=False)
    wrapper = LogitsOnly(model).eval()

    vocab_size = tok.vocab_size
    seq_len = model.seq_len
    dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=opset,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"已导出: {out_path}")
    print(f"  参数量: {n_params / 1e6:.2f}M | 输入形状: [1, {seq_len}] | opset: {opset}")
    return wrapper, model, vocab_size, seq_len


def verify_export(wrapper: LogitsOnly, out_path: str, vocab_size: int, seq_len: int) -> None:
    """用 onnxruntime 对比 ONNX 与 PyTorch 前向的数值一致性。

    Args:
        wrapper: 导出时的包装模型。
        out_path: ONNX 文件路径。
        vocab_size: 词表大小。
        seq_len: 序列长度。
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("跳过验证：未安装 onnxruntime")
        print("  完整命令: uv run --with onnx --with onnxruntime python train/scripts/export_onnx.py")
        return

    onnx.checker.check_model(onnx.load(out_path))
    print("onnx.checker: 通过")

    torch.manual_seed(42)
    dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    with torch.no_grad():
        ref: Any = wrapper(dummy).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": dummy.numpy()})[0]

    diff = float(np.abs(ref - got).max())
    print(f"onnxruntime vs PyTorch: 最大绝对误差 {diff:.2e}")
    assert diff < 1e-4, f"误差过大: {diff}"


def main() -> None:
    """命令行导出入口。"""
    ap = argparse.ArgumentParser(description="导出 ONNX 供 Netron 查看计算图")
    ap.add_argument("--ckpt", default="train/checkpoints/best.pt", help="checkpoint 路径")
    ap.add_argument("--out", default="train/checkpoints/model.onnx", help="输出 ONNX 路径")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    ap.add_argument("--skip-verify", action="store_true", help="跳过 onnxruntime 数值验证")
    args = ap.parse_args()

    wrapper, _, vocab_size, seq_len = export_onnx(args.ckpt, args.out, args.opset)
    if not args.skip_verify:
        verify_export(wrapper, args.out, vocab_size, seq_len)
    print("\n用 Netron 打开查看计算图:")
    print("  网页版: https://netron.app（拖入文件即可，模型不出本机）")
    print("  本地版: uv run --with netron netron train/checkpoints/model.onnx")


if __name__ == "__main__":
    main()
