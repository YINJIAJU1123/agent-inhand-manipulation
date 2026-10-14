# 文本 grounding v2：冻结 SigLIP2 + 学得的目标分类器 + RGB-D student

日期：2026-10-14。训练和全部 1,344 回合正式评测均已完成；另有 16 回合冒烟验证。

## 与 v1 的区别

v1 的 79.30% 使用规则颜色索引选择 canonical embedding，不能证明语言改写泛化；旧报告已更正。v2 推理链路为：

`完整文本 → 冻结 SigLIP2 文本向量 → 训练的线性分类器 → 预测颜色对应的 canonical embedding → 冻结 RGB-D/GRU 控制器 → 21 维动作`

分类器接口只有文本向量与模型权重，没有环境目标编号。评测器仍需要用环境目标生成指令、记录标签和评分，但这些标签不传入 grounding。canonical 模式直接使用环境原始指令；heldout 模式由评测器生成未见模板指令。RGB-D 是实时仿真相机输入，本体感知不含物体姿态、目标姿态或方向误差。

这是一版**六种英文颜色目标的模块化语言接口基线**，不是端到端语言动作联合训练，也不是 VLM verify/replan/recovery 闭环。

## 训练和语言测试

- 训练和编码缓存环境：22031 服务器 CPU，PyTorch 2.7.0+cu128、Transformers 4.57.6。
- 冻结编码器：`siglip2-base-patch16-224`，归一化 768 维文本特征。
- 训练：18 种模板 × 6 种颜色 = 108 条；仅训练 `Linear(768, 6)`。
- 优化：AdamW，学习率 0.02，weight decay 0.01，1000 个全批次步骤，seed 123。
- 留出：训练前固定 12 种不同模板 × 6 种颜色 = 72 条，不用于更新权重或选择 checkpoint。
- 结果：训练 108/108，留出 **72/72**；每种颜色 12/12。
- tokenizer 固定 padding 到 64 tokens；同一句单独编码与混合批次编码的最大绝对差为 `5.96e-8`。
- 改变文本排列时预测按相同顺序改变，未通过批次位置决定目标。

完整模板、混淆矩阵和逐条预测：[language_holdout.json](results/text-grounding-v2/language_holdout.json)。

本地 rollout 使用冻结编码器的**完整句子向量缓存**，以避免通过跳板机传输 1.4 GB 权重。缓存键为完整句子，值只有 SigLIP2 向量，不含颜色标签；分类器仍需从向量预测目标，未知句子会报错。使用完整模型运行时可省略 `--text-feature-cache`，通过 `--model` 指定本地模型。这里不报告在线文本编码延迟。

另外提供 `scripts/vision/infer_text_goal.py` 自由文本入口，直接加载完整编码器且不连接仿真。三个缓存外句子的推理冒烟得到 cyan、green、magenta 三个正确目标；这三例不计入预先固定的 72 条留出集，也不作为额外调参依据。

## 闭环协议

- 本地 RTX 3090，IsaacLab 2.3.2、Isaac Sim 5.1.0、PyTorch 2.7.0+cu126。
- round 10 student、teacher 定义、action scale 1.0 均冻结。
- 32 环境；RGB-D grid 16/16；vision stride 2；每回合最多 600 步；固定 goal yaw 0。
- 成功条件沿用当前协议：方向误差不超过 0.16 rad，且该回合没有掉落。`goal_hold_time_s=0`，因此不代表长期保持稳定。
- canonical：seed 123，384 回合。
- 未见模板：seed 123/456，各 384 回合。
- 错指令对照：seed 123，192 回合；把指令颜色循环偏移一位，但仍按原环境目标评分。该组验证语言对控制的影响，其“原目标成功率”不是对新指令的成功率。

每回合记录原评分目标、实际指令、预测颜色、成功、掉落和方向误差。训练/缓存和冻结 student 的校验值见 [artifact_manifest.json](results/text-grounding-v2/artifact_manifest.json)。

## 已完成的闭环结果

| 指令组 | seed | 成功 | 掉落 | 目标识别 |
| --- | ---: | ---: | ---: | ---: |
| canonical | 123 | 296/384（77.08%） | 70/384（18.23%） | 384/384 |
| 未见表达 | 123 | 309/384（80.47%） | 64/384（16.67%） | 384/384 |
| 未见表达 | 456 | 291/384（75.78%） | 74/384（19.27%） | 384/384 |
| 未见表达合计 | 123+456 | **600/768（78.13%）** | **138/768（17.97%）** | **768/768** |
| 错指令对照（按原目标评分） | 123 | **10/192（5.21%）** | 31/192（16.15%） | 对新指令目标 192/192 |

两个 seed 的 rollout 合计覆盖了全部 72 条留出指令。两个 seed 的成功率相差 4.69 个百分点，应报告两者，不选择较高者。目标识别正确后，各表达给冻结控制器的 canonical embedding 相同，因此不能将与 canonical 对照的数值差异解释为语言提升了控制性能。

错误颜色指令下，分类器 192/192 预测为文本中的新目标；按原目标评分的成功率降至 5.21%。该结果支持控制受文本目标影响，而非从评分目标索引获取指令。这不是“新指令完成率”的测量。

[汇总与分面结果](results/text-grounding-v2/summary.json) 和 [逐回合原始报告目录](results/text-grounding-v2/) 已保存。四组的 student、classifier、文本缓存、评测器 SHA256 和主要协议参数一致；逐条检查确认未见表达来自留出集合，汇总成功数与原报告一致。代码语法检查和 `git diff --check` 通过。

## 复现

在具备 IsaacLab 的环境中：

```bash
EVAL_PYTHON=/path/to/isaaclab/python \
STUDENT_CHECKPOINT=/path/to/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt \
GROUNDING_CHECKPOINT=/path/to/siglip2_text_classifier_v2.pt \
TEXT_FEATURE_CACHE=/path/to/siglip2_text_cache_v2.pt \
REPORT_DIR=outputs/visual_student_v3/reports/text_grounding_v2_newrun \
bash scripts/vision/run_text_grounding_eval.sh
```

重新训练用 `scripts/vision/train_text_grounding.py`；生成无标签的完整文本缓存用 `scripts/vision/cache_grounding_text.py`。分类器 checkpoint、文本缓存及冻结 student 已保存于本地和 22031 服务器的 `outputs/visual_student_v3/checkpoints/`。

小型复现包位于本地及远端项目的 `outputs/visual_student_v3/text_grounding_v2_release_20261014.tar.gz`，含 student、分类器、文本缓存、运行脚本和原始报告；完整 SigLIP2 权重继续保存在 22031 服务器的 `/home/jiaju/models/siglip2-base-patch16-224`。

## 适用范围

只验证固定物体、固定颜色布局、六类目标和这些英文表达。没有测试否定、多颜色歧义、颜色同义词、中文、空间关系、多物体或中途改指令。当前不能宣称通用语义推理、未知布局 grounding 或 agentic 恢复能力；这些需要独立实验。
