# SuperMimic RSLSphere65 兼容性评估

日期：2026-10-29。只读取源码、校验权重并运行 CPU 网络前向，没有启动新训练或仿真，也没有替换当前 student。

## 结论

**高层 agent 方法可以迁移，但这个 checkpoint 不能直接加载进现有 latent student。** 这次变化包括模型结构、对象、时序、观测历史和感知依赖。需要单独的 sphere latent-policy 适配版本，不能只改权重路径。

已读取 [supermimic](https://github.com/KeZhang19/supermimic) commit `c70f96d370821ebaec25ea43e07580cd80f17722`。`actor_iter19234.pt` 的实际 SHA256 为 `ec2890dfd885328e2c9aef11ad52390dd54eda91a80077924a89b100ae76eb37`，与清单一致。实际 `Actor.load_state_dict(strict=True)` 成功，CPU BF16 前向输出 `[2,21]` 且有限。均值输入只是网络检查，不是有效物理轨迹或成功率测试。

证据：[机器可读审计](../results/cube-v1-20261029/supermimic_checkpoint_audit.json)。可复现入口：`scripts/diagnostics/audit_supermimic_checkpoint.py`。

## 实际差异

| 项目 | 当前冻结方块版本 | 同事新版本 |
| --- | --- | --- |
| 物体 | RSLCube | RSLSphere65，直径 65 mm、标称 70 g |
| Actor | 继承 BeliefStudent 的循环网络 | SuperMimic MLP，2048→1024→512→256→21 |
| 输入 | proprio109、当前 pose7、latent768、hidden | 10 帧 × 124 维历史 |
| 目标 | 连续 latent 直接入网络 | 目标四元数4 + 相对目标四元数4 |
| 时间 | 30 Hz 控制 | 20 Hz 控制、120 Hz 物理 |
| 状态记忆 | recurrent hidden | 显式 10 帧窗口，约 0.5 秒 |
| 动作 | 21 维，当前环境绑定 | 21 维 tanh，原 joint-limit/EMA 映射，alpha≈0.98882 |
| checkpoint | `model`、`original_student_cfg` 等 | `actor`、`config`、`noise_config`、`metadata` |
| 感知 | 实际 RGB CNN 与方块 keypoints | 发布 Actor 使用带噪位姿，未提供球体 RGB predictor 权重 |

新输入每帧：21 关节位置 + 84 最近四步实际动作 + 4 目标四元数 + 3 基座位置 + 1 剩余时间 + 11 位姿/目标差。现有 `model.migrate()` 明确要求原 BeliefStudent 的 `student_cfg` 和 `student`，新文件不具备此格式。`NewtonRuntime` 还会检查方块资产哈希、原 student 身份和 30 Hz；直接替换会被拒绝，这些检查应保留。

同事发布的约 21 MB Actor 已包含推理所需网络及归一化。完整约 192 MB 训练 checkpoint 在其 GitHub Release，另含世界模型、critic、优化器。仅用 Actor 当冻结蒸馏教师不需要先下载完整训练状态；若继续原 SuperMimic 世界模型训练才需要完整包。本次未下载该完整包。

## 保持直接 latent 的建议路线

目标接口保持 `Astra/VLM → latent768 → 低层 policy`，不改成解码目标四元数再喂原 policy。

1. **以新 Actor 为初始化和冻结训练教师。** 独立验证它在原球体环境、20 Hz 与原噪声下工作。保留原归一化、关节顺序、动作限幅和 EMA。
2. **重新制作 latent Actor 输入。** 每帧删除 105:109 的目标四元数与 120:124 的目标差，保留 116 个物理/测量特征。加入例如 64 维学习目标特征，保留 10 帧窗口。实际目标切换时历史里的目标条件如何更新要明确，并保持训练部署一致。
3. **尽量保留权重。** 后续 MLP 层可逐项复制。第一层按每帧的物理通道重排复制，新增 latent 列单独初始化。只迁移兼容的归一化统计。首先做目标通道归零的等价性审计，再训练新目标列与投影，随后 DAgger 适配。复制权重不等于保留原成功率。
4. **补齐球体视觉。** 当前方块八角点 CNN 和 Gaussian 资产不适用于球体。需要球体有方向语义的 keypoints/纹理、渲染输入和对应 predictor。只向 policy 输入带噪真值 pose 的测试不能称为 RGB 完整系统。
5. **重新验证 W/E。** 可复用结构和训练流程，但球体动力学、20 Hz 时序和 MLP 历史都变了，现有控制器权重不能直接视为有效。等待时间/动作步数也要按秒重新核对。
6. **冻结新对象版本后比较。** 三种高层共用同一球体 student/predictor，按每条件 40 次、最多两个 seed 评测。复用成功定义和异常统计原则，明确记录 20 Hz 对应 120 秒为 2400 步。不要修改旧 cube-v1 的 3600 步合同。

如果仅想快速验证同事 Actor，直接给原目标姿态做底层诊断是合理的，但必须单列为 pose-goal reference，不能称为完成了我们的直接 latent 迁移。

## 能复用什么

- **直接复用设计/接口**：指令协议、Astra 服务、连续 latent 通道规范、日志/失败分析、40 次实验组织、W/E/F/G 消融思路。
- **可初始化但需验证**：VLM 适配器、latent head、控制器结构与部分权重。新球体视角/纹理会改变输入分布。
- **必须适配**：SuperMimic latent 输入与历史、球体语义到姿态映射、20 Hz 执行、球体视觉和绑定合同。
- **不能直接迁移**：方块成功率、Astra 优于 VLM 的数值结论、控制器收益或真机表现。

建议继续保留已有方块作为完整基线，把球体作为第二物体独立增加。当前实际 CNN 适配的方块候选已有 40/40 底层开发结果，旧链路仍有价值。此次没有改动或启用任何 sphere 训练。
