# Teacher 结果冻结记录：teacher-benchmark-v1

日期：2026-09-22。冻结的是已测结果、协议和 checkpoint 身份；后续新实验新建 v2，不覆盖 v1。当前 teacher 仍可优化，不代表已通过最终验收。

后续 T0/T1/T2 的统一评价合同已冻结在
[`docs/state_teacher_metrics_v2.md`](../../state_teacher_metrics_v2.md)：主指标是六面 balanced hold，辅指标是 repeated reorientation；v1 中的历史结果继续保留，不能与新合同下的结果直接混排。

## 三档比较的定义

目标是检验 **Direct < VisERDex-style < Ours（Revo3 工程适配）**，三档都在同一 Revo3 单 Cube state-teacher 任务下测量。它是研究目标，排序由数据决定。

| 固定方法 ID | 定义 | 当前状态 |
|---|---|---|
| T0 Direct | 原始 Revo3 state-PPO，只有任务/手型所需接口转换 | checkpoint 对应关系与同协议评估待完成 |
| T1 VisERDex-style | 移植论文 teacher 配方，逐项记录手型、奖励、观测、随机化和 curriculum 差异 | 完整复现和同协议评估待完成 |
| T2 Ours | T1 上加入明确记录的 Revo3 工程改进 | model1999 为当前候选，尚非最终版 |

注意：task 名中的 Direct 是环境/API 命名，不能单凭它判定论文 baseline。model500 来自 revo3_viserdex_cube_formal；目前将其固定命名为 **早期适配 checkpoint**。model1999 是它的 hold02 续训，不能把二者自动映射成 T0/T1。论文公开实机数字单列文献参考，不充当 T1 的 state-teacher 实测。

## 已测结果

Repeated 协议：随机目标面 + 面内 yaw，0.4 rad，30 s 总 episode，seed 101，128 episodes。存活到总时限的记录为右删失，CS 不是终生连续次数。± 表示 episode 内总体标准差（ddof=0），不是独立训练种子标准差。

| 结果 ID | checkpoint | dwell | CS mean ± SD | 掉落 | 时限删失 |
|---|---|---:|---:|---:|---:|
| model500_repeated30s | model500（早期适配） | 0 s | 7.3125 ± 5.513832 | 28/128（21.875%） | 100/128 |
| hold02_final_repeated0s | model1999（Ours 候选） | 0 s | 25.6640625 ± 11.062519 | 36/128（28.125%） | 92/128 |
| hold02_final_repeated02s | model1999（Ours 候选） | 0.2 s | 23.2890625 ± 7.434245 | 23/128（17.96875%） | 105/128 |

同为 zero dwell 时，CS 提升，但掉落率也增加 6.25 个百分点，不能概括成所有稳定性指标全面提升。0.2 s dwell 行采用不同成功条件，应单列。续训增加了训练预算；不能把改善单独归因于 hold shaping。

Balanced hold 协议：model1999，六面各 64 episodes，seed 101，0.16 rad，连续 0.5 s，10 s 总 horizon，随机面内 yaw，目标整段固定。

| 指标 | 计数 | 比例 |
|---|---:|---:|
| 曾瞬时到达 | 364/384 | 94.791667% |
| 曾连续保持 | 360/384 | 93.750000% |
| 终态保持 | 293/384 | 76.302083% |
| 掉落 | 18/384 | 4.687500% |

逐面 held-at-end（face 0–5）：49、51、45、49、51、48 / 64。其他控制指标、Wilson 区间和每条 episode 都保留于 raw/ 与 record.json，不只保留最好看的数字。

## 纠正 VisERDex 定义

[VisERDex 原文](https://arxiv.org/html/2604.11138) Appendix B-A1 / Table IX：teacher 成功奖励阈值为 0.1 rad；无成功 10 s、掉落或累计 50 次成功终止。Section IV-D / Table IV：**实机评估用 0.4 rad**，每个条件 5 runs，Cube nominal 35.4 ± 13.8，adversarial 25.6 ± 8.9。这是完整视觉系统的实机成绩，不能称为 state-teacher 成绩或直接比较优劣。之前会话将 0.1 rad 统称为论文评价阈值不准确，以本记录为准。

## 固定与追溯

- 原始 9 份 JSON 原样归档至 raw/；checkpoint 路径、SHA-256、argv、完整 env_cfg 和 runtime 随原始报告保留。
- record.json 从 episode 记录重新计算并核对汇总，不手填四舍五入后的数字。
- SHA256SUMS 检测文件变化。未来更正另写勘误/新版本，不静默改 v1。
- 两个 checkpoint 的二进制目前未在本地找到；已固定远端路径和报告中的哈希，权重本地备份仍待补齐。
- 原始报告 git_revision 为 null。local_source_snapshot/ 仅保存冻结时本地代码，不能冒充当时远端代码的精确版本。
- video_metadata/ 和 record.json 保存三段 30 s 视频的来源与哈希；它们是 state-teacher 定性回放。

## 后续比较合同

先补齐 T0/T1/T2 的实现差异、初始化、训练预算与 checkpoint 映射。训练阈值和评价阈值分别固定；将 0.1 rad teacher 条件和 0.4 rad deployment-style 条件分开。10 s 无成功窗口不等于 10 s 总 episode。对三档使用相同目标/重置分布、终止规则和评价种子，报告掉落、timeout、删失、CS 和 hold。正式方法结论需要多个独立训练种子；多个 evaluation seeds 不能代替训练重复。不能预先承诺排序。
