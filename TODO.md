# TODO

> 更新于 2026-08-20。改代码红线见 CLAUDE.md（56 测试 + evaluate 6/6 + 示例命令 + 金样本回归）。

## 🧪 项目迭代

- [ ] distill 新方向：中等词表 BPE tokenizer 实验（5K-10K，降低 embedding 占比）
  - 依据：第 9 篇蒸馏三连败的"不可能三角"结论；从 student 架构和 tokenizer 下手
  - 安全网已就位（2026-08-19 三件套）：
    - `train/scripts/regression.py` 金样本回归（27 条，量化"换 tokenizer 改变多少行为"；
      重建基线用 `--update`，当前基线锚定 best.pt md5=a7fbf79a223b，2026-08-20 重训后重建）
    - checkpoint 版本指纹（git hash + 数据 md5 + tokenizer md5 随权重落盘）
    - 加载时 tokenizer 一致性校验（词表漂移显式警告）
  - 前置加固已完成（2026-08-20）：prepare_data 已参数化（`--seq-len`/`--tokens-per-char` 联动 config.yml，换 tokenizer 时长度上限不再静默失效）
  - 实验产出可写系列第 11 篇
