# GPT Teacher 项目规范

## 代码规范

- 写任何 Python 代码前，先加载 `python-code-style` skill 并遵循其规范。
- 本项目已落地的工具链（配置见 `pyproject.toml`，提交前必须三清）：
  - `uv run ruff check .` + `uv run ruff format --check .`（120 列，E/W/F/I/B/C4/UP/SIM）
  - `uv run mypy .`（strict，tests 豁免 untyped-def）
- 公共 API 用 Google 风格 docstring；导入一律绝对导入。
- mypy strict 的深度标注模式（本项目实证过的坑）：
  - numpy 数组标注写全泛型参数 `np.ndarray[Any, Any]`，裸 `np.ndarray` 报 type-arg
  - `nn.Module.__call__` 的返回值是 Any：先落显式类型标注的局部变量再 return（直接 return 报 no-any-return）
  - `zip()` 一律带 `strict=`（B905）
  - 函数签名引用 torch/模型类型：模块顶部 `TYPE_CHECKING` 导入 + `from __future__ import annotations`
- 重依赖延迟导入：torch / matplotlib 等在函数体内 import（模块级 try/except + 友好提示），
  保证不装重依赖的纯 CLI 场景可降级；运行时真实使用的导入不得放进 TYPE_CHECKING。
- 可视化代码规范（走查实证）：
  - 中文字体：模块级设 `plt.rcParams["font.sans-serif"]`（Arial Unicode MS 系）
  - token 显示用 `core.visualize.display_token / reassembly_labels`（空白字符换 ↵/·，
    byte 碎片重组「词···」），不手写 decode
  - 热力图色标手动固定（数据含退化恒定值如对角 1.0 时，vmin/vmax 不设会被钉死）
  - 色标不放竖排中文 label（旋转小字号不可读），方向性说明写进面板标题

## 改代码红线（用户 2026-08-19 指定）

任何改动完成前，必须全部通过：

1. `uv run pytest core/tests/`（56 个测试）
2. `uv run python -m core.evaluate` 验收 6/6
3. `train/README.md` 里的示例命令（推理等）行为正常
4. 涉及生成行为改动时：`uv run python train/scripts/regression.py` 金样本 27 条全过（基线勿随意 `--update` 重建）

## 验收分工（用户 2026-08-20 确立）

上述红线验收由 Claude 在交付前完成并附证据，用户无需手工重跑训练和验收：

- 不涉及训练产物的改动：regression 零行为变化即证明，用户重训反而引入变量
- 涉及训练行为（数据/超参/tokenizer/结构）的改动：由 Claude 代跑重训（5000 步 MPS 约 5 分钟），交付 evaluate 结果 + regression 行为变化量 + 基线处置建议
- 用户只负责：公众号群发、最终效果拍板、交互体验类判断
