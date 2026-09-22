# State-teacher metrics contract v2

冻结日期：2026-09-22。这个合同用于 T0 Direct、T1 VisERDex-style 和 T2 Ours；后续训练和 checkpoint 选择不能更换指标定义。训练奖励可以变化，但评测条件、统计口径和主指标保持不变。

## 统一主指标：usable teacher rate

每个目标面 64 个 episode，六个面共 384 个 episode；seed `101`；cube 目标面随机，面内 yaw 随机；每个 episode 的目标保持不变；总 horizon 10 s。

- 接受阈值：姿态误差 `<= 0.16 rad`。
- 连续保持：在阈值内连续 `0.5 s`。
- `usable_teacher_rate`：episode 内完成连续 0.5 s 保持，并在 episode 结束时仍在阈值内。
- `drop_rate`：目标物掉落的 episode 比例。
- `instant_reach_rate`：episode 内至少一次进入阈值，只作诊断。

所有方法只用 `usable_teacher_rate` 作为主排序指标。`drop_rate` 是安全门槛，`instant_reach_rate` 用于解释“能不能到达但不能稳住”。不使用加权总分，避免通过牺牲掉落率换取到达率。

当前工程验收门槛暂定为：`usable_teacher_rate >= 75%` 且 `drop_rate <= 5%`。必须在 3 个独立训练 seed 上都满足，才认为 teacher 足够支持后续 student 训练，可以停止 Ours 优化。

## 辅指标：repeated reorientation

随机目标面和随机面内 yaw；seed `101`；128 个 episode；总 horizon 30 s；接受阈值仍为 `0.16 rad`；成功后立即采样下一个目标。

- `CS mean/std/min/max`：episode 内连续成功次数，明确标记 horizon-censored episode。
- `drop_rate`、`horizon_censored_rate`。
- `mean_time_to_first_success_s`。

这个指标衡量连续重定向能力，不能替代主指标。VisERDex 论文的 `0.4 rad` 实机部署指标另行报告，不混入本合同。

## 控制质量和诊断指标

所有 episode 同时记录：最终姿态误差、最小姿态误差、动作 L2、动作变化 L2、目标增量、关节速度 RMS、物体角速度、原始动作越界比例。它们只用于解释失败原因和检查 reward 单位，不参与主排序。

## 视频验收

固定保存 3 段 30 s qualitative rollout，evaluation seeds `101/102/103`，记录 checkpoint、目标面、是否掉落和视频 SHA-256。视频只用于确认动作是否自然、目标切换是否正常和失败模式，不替代数值主指标。

## 统计和报告要求

- 报告总数、成功数、比例和 95% Wilson 区间。
- 按六个目标面分别报告 `held_at_end_rate`，同时报告六面宏平均；不能只报告最好的面。
- 报告掉落、超时和右删失；CS 不是无限时长寿命统计。
- 正式结论至少使用 3 个独立训练 seed；evaluation seed 固定为 `101` 以保持版本间可比。
- 每个结果保存 checkpoint 路径、SHA-256、git revision、完整 argv、环境配置和 runtime。

对应命令入口：`scripts/rsl_rl/evaluate_state_baseline.py`。
