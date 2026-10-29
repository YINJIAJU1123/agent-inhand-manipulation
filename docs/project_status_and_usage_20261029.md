# 当前项目、结果与使用方法

更新：2026-10-29。本页覆盖旧记录中的运行状态。这次发布是源码与说明快照，训练权重、原始数据和认证文件不随代码提交。

## 1. 当前方法

```text
指令 + 当前 RGB
 ├─ Astra 工具参数 → 冻结语义 latent bank
 ├─ 微调 VLM → 连续 latent
 └─ 规则 VLM → 语义技能 → 冻结 latent bank
                        ↓
RGB → 冻结 CNN → 当前 pose7
proprio/history109 + pose7 + latent768
                        ↓
循环 latent student → 21 维动作
                        ↓
Astra 等待/执行控制器 → Revo3 → 后续观察与执行反馈
```

student 使用 768→64 的目标投影和继承的 BeliefStudent core，不接收目标四元数或相对目标四元数。Astra 本身输出有界工具参数，经适配层变成 latent，并非直接生成 768 个浮点数。微调 VLM 直接输出连续 latent。

主比较中的 VLM 微调/规则均为裸基线，没有学习等待/执行控制器。三种方法共享测量位姿停止门控。仿真在模型思考期间持续推进。

## 2. 已核实的结果

### 正式 cube-v1

每条件 40 次、两个 seed、120 秒、最多六次高层调用。成功要求正确语义绑定、自主停止、完整四元数误差 ≤0.16 rad、此前没有掉落。仅到达姿态不是完整任务成功。

| 方法 | 原始成功 | 掉落 | 三方法共同服务有效集 |
| --- | ---: | ---: | ---: |
| Astra + W/E | 29/40，72.5% | 10 | 29/39 |
| 裸微调 VLM | 27/40，67.5% | 9 | 26/39 |
| 裸规则 VLM | 14/40，35.0% | 15 | 13/39 |

证据：[正式系统比较汇总](results/cube-v1-20261029/formal_system_comparison.json)。

Astra 对微调 VLM 的配对 McNemar p=0.60724，尚不能证明稳定领先。服务异常保留在原始 40 个位置中，共同有效集对所有方法排除相同位置，不替补样本。

VLM 审计确认正确 LoRA/head、连续 latent 实际执行、没有额外控制器，已知训练 capture 与正式请求没有完全相同图像。固定字母—颜色布局捷径仍未排除。见[审计与消融设计](experiments/vlm_debug_and_ablation_design_20261029.md)。

### 额外 student 训练已经完成

两种候选各追加 400 次 DAgger 更新，再在相同的 40 个新开发初始状态、实际 CNN 位姿输入下测试：

| 底层控制条件 | 到达目标 | 掉落 |
| --- | ---: | ---: |
| 同事原 pose-goal student | 39/40 | 1 |
| 冻结 latent student | 38/40 | 2 |
| 模拟位姿噪声追加 DAgger | 39/40 | 1 |
| 实际 CNN 位姿追加 DAgger | 40/40 | 0 |

这是底层到达测试，不能称为语言任务 100% 成功。CNN 候选通过了开发准入条件，但尚未替换正式 student。证据：[开发汇总](results/cube-v1-20261029/student_development.json)。

### 消融及暂停状态

- **W/E**：等待×执行四组各 40 次已完成，原始成功分别 23、26、28、29。配对分析尚未证明显著组件收益。
- **G**：有/无显式目标的匹配 W/E 四个训练均完成。三组在线消融队列仍在等空卡，现已暂停。
- **独立复验**：三方法共 120 次计划，暂停时该队列尚未完成任何新试验。
- **同状态不同目标**：本地 Newton 已能实际运行，但重复轨迹在第 60、90、119 步未能精确重建物理状态。暂未通过配对准入。
- **F**：一次初始计划的格式/执行时序测试通过，尚无正式在线结果或真实初始计划 API 结果。
- **学习执行速度**：仅候选代码，未训练验证。正式 E 的速度仍固定为 1，学习动作残差。
- **视觉语义重映射**：待实现，不能假设改 mesh texture 就改变 Gaussian 观察。

用户要求暂停后，三个自有排队进程均已停止，未改变冻结权重或历史结果。不要根据旧日志里的 PID 或“running”判断现在仍在运行。

## 3. 代码入口

| 路径 | 功能 |
| --- | --- |
| `scripts/pose_latent_student_v1/model.py` | 权重迁移、直接 latent 循环 policy |
| 同目录 `prepare.py`、`train.py`、`run_stages.sh` | 初始化检查、接口蒸馏、teacher DAgger |
| `perception.py`、`newton_runtime.py`、`goals.py` | 实际 RGB pose、运行合同、方块语义映射 |
| `grounding.py`、`semantic.py`、`astra_bridge.py` | 微调 VLM 与 Astra 接口 |
| `control_modules.py`、`train_modules.py` | W/E 模块与训练 |
| `evaluate.py`、`evaluate_agent.py` | 底层到达/完整语义任务两套不同指标 |
| `formal_release.py`、`freeze_release.py` | 固定代码、权重、物理参数和评分 |
| `scripts/pose_latent_studies_v2/` | 复验、目标消融、一次计划、状态重建 |
| `scripts/diagnostics/` | VLM 输入与新 checkpoint 审计 |
| `scripts/astra_transport_v2/` | 既有授权传输、调用预算与审计 |

工作树包含后续候选。重现正式结果要使用对应 release 中冻结的源码/权重哈希，不能直接把最新工作树当作旧版本，也不能重写 manifest 绕过校验。

## 4. 环境与文件准备

方块依赖同事 ViserDex commit `93317c16e3f6ab6b35c7d36174c5b6c7fe090e80`。IsaacLab v3.0.0 commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`，包元数据 28.0.0，Isaac Sim 6.2.0.0、Newton 1.6.1、Warp 1.17.0、Torch 2.11.0+cu130、Python 3.12。按对应仓库固定版本安装指南准备 CUDA 编译工具和 Python headers。新的 SuperMimic main 不能替代旧方块源码。

必须另外准备：

- 原方块 student、绑定的 `env.yaml` 和用于 DAgger 的 PPO teacher。
- 本项目已训练 latent student，或通过下述两阶段训练获得。
- 原方块 USD、Gaussian 外观、keypoints 与相机标定。
- 方块 CNN `cnn_6202_final.pth`。
- 原 `siglip2_text_cache_v2.pt`，六个颜色向量及其哈希不变。
- VLM backbone、对应 LoRA 和 latent/readability head。
- Astra 完整版 W/E 权重，以及已授权的模型服务。
- 精确复现还需要对应的冻结 release 与源码快照。

这些实验权重/数据没有包含在这次源码提交中。认证文件不入 Git。不允许缺文件时退回真值 pose 或静默换模型。

现有 shell runner 的目录约定：

```text
$RUNTIME_ROOT/
  bin/activate_runtime.sh       # 导出 VISERDEX_PYTHON 和固定依赖环境
  viserDex_revo3/               # 上述旧方块版本
  VLMmanipulation/              # 本仓库
  prepared_release_20261029/
    student/student_1000.pt + env.yaml
    teacher/model_4800.pt
  outputs/                     # 模型、服务文件、评测
```

不少研究 wrapper 引用日期目录和已存在的实验文件，迁移主机前须检查路径。`state_pair/prepare_local_runtime.py` 是当前主机的开发环境搬移工具，不是通用安装器。

## 5. 使用示例

以下命令是手动使用说明，不会恢复已暂停队列。每次使用新的输出目录。

### 审计 latent 权重初始化

```bash
export RUNTIME_ROOT=/absolute/path/to/prepared_runtime
source "$RUNTIME_ROOT/bin/activate_runtime.sh"
export PYTHONPATH="$RUNTIME_ROOT/viserDex_revo3/source/viserdex:${PYTHONPATH:-}"
"$VISERDEX_PYTHON" scripts/pose_latent_student_v1/prepare.py \
  --source "$RUNTIME_ROOT/prepared_release_20261029/student/student_1000.pt" \
  --output "$RUNTIME_ROOT/outputs/latent_initialization_audit"
```

此步骤验证权重映射、梯度遮罩和循环序列接口，不代表已经训练或保留了原成功率。

### 两阶段训练

```bash
export CUDA_VISIBLE_DEVICES=<available-GPU-index-or-UUID>
export RUN_ID=my_cube_migration
bash scripts/pose_latent_student_v1/run_stages.sh
```

依次执行 200 次 interface 更新和 400 次 teacher DAgger。额外实际 CNN 适配入口为 `run_pose_adaptation.sh`，需显式设置 `SOURCE_DIR`、`OUTPUT_DIR`、`POSE_SOURCE=gs_cnn` 并准备已有训练模型。检查 completion JSON，不能只看 checkpoint 是否存在。

### 查看评测参数

```bash
cd "$RUNTIME_ROOT/viserDex_revo3"
bash scripts/run_newton.sh \
  "$RUNTIME_ROOT/VLMmanipulation/scripts/pose_latent_student_v1/evaluate.py" --help
bash scripts/run_newton.sh \
  "$RUNTIME_ROOT/VLMmanipulation/scripts/pose_latent_student_v1/evaluate_agent.py" --help
```

实际视觉底层测试使用 `evaluate.py --pose_source gs_cnn`。完整语言任务使用 `evaluate_agent.py`，显式提供 provider、maintenance、stop 和全部模型路径。主表裸 VLM 必须为 `maintenance=continue`，不传学习 actor/execution/completion checkpoint，并保持冻结 measured gate。`run_agent_eval.sh` 的历史开发默认值不等于主表裸基线配置。正式运行必须通过 release guard。

新方块实验统一使用 `nconmax=4096`、`njmax=1024`，检查全部日志并拒绝容量溢出。保留每条件 40 次原始数据、异常和 seed 结果，不挑最佳批次。

### 仅检查同事新权重

```bash
python scripts/diagnostics/audit_supermimic_checkpoint.py \
  --repository /absolute/path/to/supermimic \
  --output /absolute/path/to/checkpoint_audit.json
```

需要 PyTorch 和同事源码依赖，使用 CPU、`weights_only=True` 校验 SHA、严格加载与前向。不启动仿真或训练。适配建议见[兼容性评估](experiments/supermimic_compatibility_20261029.md)。
