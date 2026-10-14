# 双路线 v4：学习的目标 latent 与 Astra 工具控制

开发始于 2026-10-14；训练与闭环检查跨至 2026-10-15。

## 方法与边界

两条路线复用冻结的 RGB-D student、21 维动作及 action scale=1.0。

| 路线 | 高层输入 | 高层输出 | 控制方式 |
|---|---|---|---|
| 微调小 VLM | RGB、由 RGB 计算的裁剪、实际语言指令 | 连续 768 维目标向量、可读性概率 | 预测向量直接输入 student；周期观察 |
| Astra | 相同 RGB/裁剪、实际语言、测量反馈和调用历史 | inspect / reorient / verify / stop / abstain 工具 | 选择已有技能及有界执行时长 |

小模型为 Qwen/Qwen3-VL-4B-Instruct，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。官方 ModelScope 下载的两个权重分片已逐一通过该 Hugging Face revision 的 LFS SHA-256 核对。采用 NF4、QLoRA r=8/alpha=16/dropout=0.05，训练语言层 attention q/v 投影和目标 head；视觉 student 冻结。

训练损失包含可读性 BCE、可见目标判别、向量余弦对齐和冻结 student 动作一致性。动作参考来自已有颜色技能，不是新增的状态 teacher 数据。训练期间保持 student eval；动作一致性使用原生 GRU 以允许输入梯度。

六个原有单位目标向量的两两余弦约为 0.969–0.990。第一版自由向量加普通 cosine softmax 预测集中到单一颜色。第二版预测已有技能均值附近五维子空间的有界连续坐标，映射为 768 维向量并归一化；判别采用向量距离，同时均衡采样目标颜色。运行时没有 argmax 后的原型替换，最近技能只用于诊断和评分。

这只学习六个技能附近的目标表示，768 维输出不代表 768 个自由度，不证明任意语义或连续姿态控制。每 90 步的观察时机仍由规则决定，尚未学习观察、验证或停止策略。

## 新数据

| 划分 | seeds | 帧数 | 实际不同布局 | 可读目标标签 | 模糊标签跳过 |
|---|---|---:|---:|---:|---:|
| 训练 | 32001–32004 | 170 | 24 | 129 | 203 |
| 图像验证 | 33001–33002 | 62 | 11 | 52 | 79 |

不使用旧 v3 验证数据。训练、验证和测试从互不相交的符号排列库抽样，seed 也分离。一帧对应多个指令样本，序列帧彼此相关；帧数不等于独立场景数。

监督使用同物理状态下显示/隐藏符号的 RGB 差异、投影区域、朝向和像素门限。开发探针中深度未可靠包含贴纸，不能只靠深度判断可读性。门限按实际图案面积修正，模糊或部分遮挡标签跳过。门限确定于开发探针，不用测试结果调参。此标签是渲染监督代理，不等同于人工保证可读。

布局、姿态、投影框、隐藏符号的对照图像仅用于标签或评分，不输入模型。两条路线共用基于 RGB 连通区域的裁剪；完整 RGB 也输入模型。

## 训练开发结果

固定随机子集包含 72 个独立划分的图像/指令样本，其中 17 个目标标记为可读。

| 版本 | 优化步数 | 可见目标定位正确 | 接受的预测 | 接受且正确 | 不可读样本错误接受 |
|---|---:|---:|---:|---:|---:|
| latent_v1 自由向量 | 200 | 3/17 | 17 | 3 | 2 |
| latent_v2 有界连续表示 | 600 | 15/17 | 17 | 15 | 2 |

两版同时改变输出约束、损失、采样及训练步数，不能把改善归因于一个模块。第一版失败检查点保留。第二版训练 2,949,120 个 VLM 参数和 662,278 个 head 参数，72 个 LoRA B 张量均发生非零更新；可见目标的平均目标向量余弦为 0.99788。第二版完成真实 QLoRA 训练；仅有梯度或 loss 下降不作为任务成功证据。图像验证用于开发，不是最终测试。

## Astra 接口及当前依赖

通过 Responses API 调用 `gpt-6-astra`，单次严格校验一个工具：观察/执行可指定 10–90 步，验证保持技能 20 步，stop 声明由独立评分器检查，abstain 不声明成功。后续请求包含实际执行后图像、图像与本体变化、前一技能及状态，使用 response/call ID 保留历史。

请求不含目标编号、布局、真实姿态、奖励、方向误差或评分成功标志。日志保存提示词、原始工具调用、模型标识、用量、图像哈希，不保存 key。

截至本报告，未配置 `OPENAI_API_KEY`，Astra 真实评测尚未运行。代码不会使用 Codex 登录凭据替代 API key。接口 mock 测试不算 Astra 性能证据。模型与工具规范参考 [Astra 官方文档](https://developers.openai.com/api/docs/models/gpt-6-astra) 和 [function calling](https://developers.openai.com/api/docs/guides/function-calling)；账户权限仍需实际请求确认。

## 闭环开发协议

固定 seed34001 标准指令、seed34002 改写指令，每组六个目标，每模式 12 回合。每回合最多 600 步、6 次高层调用；每个 seed/mode 启动新仿真进程。oracle 提供真值目标作为执行对照。

主成功指标为正确目标绑定且正在执行、完整姿态误差 <=0.16 rad、未掉落；无 hold 要求，评分器终止。Astra stop 请求另记误停，因此不能用主成功率声称自主完成认证。推理等待期间物理仿真暂停，视频不含等待。

必须检查配对初始物理状态、符号布局和共享代码指纹。v4 更换布局库、RGB 裁剪并刷新 reset 后颜色标记，旧 v3 22.22% 不直接用于比较。两路线同时改变表示和调度，是系统比较而非单模块消融。

## 复现

Isaac Lab 环境另需 transformers 4.57.6、peft 0.17.1、bitsandbytes 0.50.2、accelerate 1.15.0。本机新增依赖隔离于 outputs/vlm_dependencies 和 outputs/learning_dependencies。

```bash
# 在已安装 Isaac Lab 的 Python 环境执行；本机路径如下。
PYTHON=/home/yin/Grasping/Grasppose/env_isaaclab/bin/python
export PYTHONPATH="$PWD/outputs/vlm_dependencies:$PWD/outputs/learning_dependencies:$PWD/outputs/video_dependencies"
$PYTHON scripts/agentic/run_latent_data_v4.py --output outputs/agentic_v4/data/clean
$PYTHON scripts/agentic/prepare_latent_dataset_v4.py --input outputs/agentic_v4/data/clean --output outputs/agentic_v4/data/prepared
$PYTHON scripts/agentic/train_latent_goal_v4.py \
  --model outputs/agentic_v4/models/Qwen3-VL-4B-Instruct \
  --train outputs/agentic_v4/data/prepared/train --val outputs/agentic_v4/data/prepared/val \
  --output outputs/agentic_v4/checkpoints/latent_v2 --steps 600 --accumulation 2
$PYTHON scripts/agentic/run_two_track_v4.py --output outputs/agentic_v4/eval/development \
  --latent-checkpoint outputs/agentic_v4/checkpoints/latent_v2 --modes latent oracle --record-video
$PYTHON scripts/agentic/analyze_two_track_v4.py outputs/agentic_v4/eval/development
PYTHONPATH="$PYTHONPATH:scripts/agentic" $PYTHON -m unittest scripts/agentic/test_two_track_v4.py
```

安全配置 API 凭据后，使用 `--modes astra` 追加配对评测。已完成输出禁止覆盖；重新复现需新路径。权重、图像和视频留在 outputs；Git 归档小型报告和指纹。


## 完成的闭环结果（2026-10-15）

| 模式 | 标准指令组 | 改写指令组 | 总成功 | 最终绑定正确 | 掉落 | 高层调用 |
|---|---:|---:|---:|---:|---:|---:|
| 微调小 VLM + 连续 latent | 3/6 | 1/6 | **4/12，33.33%** | 5/12 | 2/12 | 54 |
| 真值目标 oracle | 6/6 | 6/6 | 12/12 | 真值给定 | 0/12 | 0 |
| Astra | 未运行 | 未运行 | 不可报告 | 不可报告 | 不可报告 | 0 |

两组的初始位置、姿态、本体状态、符号布局逐项完全相同；学生、目标 adapter、调用预算、步数和共享评测源代码指纹一致。12 回合不足以支持统计优越性结论。

latent 的 8 个失败回合中，6 个没有接受目标、1 个最终绑定错误、1 个绑定正确但未达到姿态阈值。两个掉落均发生在尚未接受目标的观察阶段。另外，改写组有 3 个回合偶然达到目标姿态但未完成语义目标绑定，因此不计成功。标准组与改写组使用不同 seed，不能仅凭 3/6 与 1/6 把差异归因于措辞。

按符号计：circle 2/2、triangle 1/2、cross 1/2，star/diamond/bars 均 0/2。尚未证明六类目标均能可靠完成。图像验证的 15/17 不能替代闭环测试结果，也不能称为稳定基线。

目前证据支持：小 VLM 确实完成了参数微调，连续预测向量确实直接驱动冻结 student，并完成部分未见场景。后续优先补足遮挡、主动观察后的新视角和改写指令数据，训练更可靠的证据接受及观察决策；同时在获得 API 凭据后完成 Astra 配对测试。正式方法研究还需拆分表示、观察调度、训练步数等因素，不能用本轮系统比较替代消融。

### 证据与录像

- [四组评测汇总与配对审计](results/two-track-v4/analysis.json)
- [第二版训练清单](results/two-track-v4/latent_v2/training_manifest.json)
- [第二版图像验证原始预测](results/two-track-v4/latent_v2/validation.json)
- [第一版失败验证](results/two-track-v4/latent_v1/validation.json)
- [标准组完整 latent 轨迹](results/two-track-v4/34001_latent/report.json)
- [改写组完整 latent 轨迹](results/two-track-v4/34002_latent/report.json)
- [归档 SHA-256](results/two-track-v4/sha256.json)

标准组六条连续录像保存在本地 `outputs/agentic_v4/eval/development/34001_latent/`。成功例 case1 为 circle、case2 为 triangle、case3 为 cross。失败例同样保留，不仅展示成功片段。模型路径为 `outputs/agentic_v4/checkpoints/latent_v2/`，包含 LoRA、goal head、训练来源快照及日志。

8 项接口测试通过；它们检查连续向量传递、梯度、数据划分、工具参数、调用预算及输入边界，不代表 Astra 性能。真实 latent 训练和 24 个闭环回合另有上述结果。
