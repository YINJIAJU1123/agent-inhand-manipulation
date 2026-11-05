# REVEAL：Agentic In-Hand Manipulation

面向同事的研究代码交接版。任务是根据语言指令，把手中物体上指定的字母／点数表面转到目标姿态。

**先读这份入口，再看 [详细说明](README_DETAILED.md)。** 本分支是从研究工作区整理出的独立源码快照，不携带历史实验分支的大文件或账号配置。

## 代码包含什么

- **Cube**：RGB 位姿估计、连续目标 latent student、等待／执行控制器、闭环 Agent 评测。
- **Die（骰子）**：带噪仿真位姿、连续目标 student、控制模块、三方法评测。
- **Duck / Hedgehog / Pig / Milk carton**：物体专属 student、字母标记资产、微调 grounding 和评测。
- **三种方法**：完整 Astra、裸 Fine-tuned VLM、裸 Rule VLM。
- **训练与分析**：数据采集、student／grounding／控制器拟合、中英文指令模板、结果报告脚本。

本分支不包含权重、运行资产、完整训练数据、完整实验日志或 API 凭据。训练和评测代码已按模块整理；它不是通用机械手 SDK，也不是六物体共用同一策略的零样本迁移模型。

## 从哪里读起

| 内容 | 入口 |
|---|---|
| Cube 主循环 | `scripts/pose_latent_student_v1/evaluate_agent.py` |
| 其他标记物体主循环 | `scripts/inhand4_q1/evaluate_agent.py` |
| 骰子主循环 | `scripts/dice_latent_v1/evaluate_agent.py` |
| 物体配置 | `configs/objects.json` |
| 等待／执行控制 | 各模块下的 `control_modules.py` |
| 模型视觉 grounding | 各模块下的 `grounding.py`、`semantic.py` |
| 交接启动入口 | `scripts/handoff/launch.py` |

## 最短使用流程

```bash
git clone --branch codex/reveal-colleague-handoff-20261106 --single-branch \
  git@github.com:YINJIAJU1123/agent-inhand-manipulation.git
cd agent-inhand-manipulation

# 先从项目负责人处取得独立 artifact 包并解压；这里用实际绝对路径。
export REVEAL_BUNDLE=/absolute/path/reveal_artifacts_20261106
python3 scripts/handoff/check_bundle.py --bundle "$REVEAL_BUNDLE"

# 默认只生成命令，不加载 GPU、不调用 API。
python3 scripts/handoff/launch.py --bundle "$REVEAL_BUNDLE" \
  --object duck --method vlm --gpu 0 --episodes 1 --output outputs/duck_dev_001
```

实际运行还需 **Python 3.12、匹配的 Isaac Lab / Newton / CUDA 环境、单独下载的 Qwen3-VL-4B-Instruct**。按详细说明先启动推理服务，再在评测命令末尾加 `--execute`。Astra 需要使用者自己的可用 API 账户；此分支不共享作者账号。

## 权重在哪里

独立交付目录为 `reveal_artifacts_20261106/`，内部按 `objects/cube`、`die`、`duck`、`hedgehog`、`pig`、`milk_carton` 分组，包含选定权重、LoRA、环境配置和运行依赖资产。`MANIFEST.json` 记录逐文件大小和 SHA-256；Git 中只有版本引用和检查工具。

## 验证范围

交接检查覆盖源码语法、文件依赖、六物体三方法命令配置和独立 artifact 完整性。整理过程没有另开 GPU 评测，也没有把这个新目录宣称为已完成新机器端到端复现。现有正式实验仍使用其冻结发布；交接入口默认是新种子的开发运行。
