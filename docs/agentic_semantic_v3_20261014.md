# 多语义目标、主动观察与执行反馈：agent v3

状态：首版实现与冻结的 54 回合验证已完成（2026-10-14）。链路可运行，当前成功率不足以称为稳定多语义 agent；旧版固定星形任务和 900 回合报告保持独立。

## 控制接口

1. 完整用户指令先单独进入本地 Qwen2.5-VL-7B-Instruct，输出 `goal_symbol` 与 `goal_direction`。
2. 实际 RGB、完整指令、前一步解析出的语义目标随后进入同一 VLM；输出可见贴纸的 `color/symbol`。视觉判断不能改写语言目标。
3. agent 将语言目标与视觉证据匹配，选出目标颜色技能；未知、冲突与已观察表面进入证据记录。
4. 路由器从冻结的 `canonical_embeddings[6, 768]` 取一个 768 维向量。
5. 冻结 RGB-D student 输入：1060 维视觉特征历史、128 维本体特征历史（history=8）及目标向量；输出 21 维动作，action scale=1.0。

VLM 不直接生成目标 latent 或关节动作。当前 embedding 是六个已训练颜色技能的接口，不能据此声称任意语义向量均可控制。

例如，开发集真实成功回合的两个 VLM 输出为：

```json
{"goal_symbol":"cross","goal_direction":"up"}
```

```json
{"stickers":[{"color":"magenta","symbol":"cross"}]}
```

第二段为原始视觉清单中与目标匹配的一项，完整原始回答仍保存于逐回合 trace。路由器选择颜色技能 4（magenta），取 `canonical_embeddings[4]`；VLM 没有生成这一 768 维向量。

## 新任务

支持 star、circle、triangle、cross、diamond、bars 六类语义目标，方向暂限 up。目标从实际指令由 VLM 解析，不按环境索引或固定 star 选择。英文 canonical 与改写模板由评分器生成；布局独立于指令随机化。

同一 seed 改变 instruction offset 时保留物体初始状态及符号布局，只更改语言目标和评分目标。高层 agent 不能访问布局、目标编号、物体姿态、奖励或方向误差。所有 VLM 请求保存完整 prompt、原始回答、图像哈希（文本请求为文本哈希）及决策。每回合预算 6 次，包括 1 次语言解析及最多 5 次视觉调用。

## 主动观察与执行反馈

- 轻量反馈：相邻图像变化、距上次 VLM 观察的图像变化、测量本体状态变化、连续静止次数。
- `periodic`：每 90 控制步调用；每个观察动作最多 90 步。
- `feedback`：新视角出现或观察超时触发调用；观察动作停滞时允许提前切换；执行停滞且无法重新看到目标时，最多一次 20 步再观察动作，然后回到已接受目标。
- 两种多次观察模式共享语言解析、视觉证据记忆、目标绑定和技能库。
- 首次唯一可见目标可接受；冲突目标需要不同图像支持。未知观察不清除已接受目标。
- 不同图像不代表统计独立观察；图像静止不代表成功或滑移。当前反馈是可审计的启发式，尚无学习的抓持风险估计。

## 评分边界

沿用先前严格到达指标：正确绑定语言指定目标并正在执行它，完整姿态误差不超过 0.16 rad，且未掉落。固定 yaw=0；没有 hold 要求。成功由独立评分器停止回合，agent 尚不认证完成。

VLM 同步阻塞时物理仿真暂停；回放不包含推理等待。该版本不证明实时部署能力。再观察动作的触发也不等于已证明恢复成功。

## 运行

```bash
python scripts/agentic/run_semantic_eval_v3.py \
  --output outputs/agentic_v3/validation \
  --protocol configs/agentic_semantic_v3_validation.json \
  --model /path/to/Qwen2.5-VL-7B-Instruct
python scripts/agentic/analyze_semantic_eval_v3.py outputs/agentic_v3/validation
PYTHONPATH=scripts/agentic python -m unittest scripts/agentic/test_semantic_agent_v3.py -v
```

启动器只管理自己启动的服务，保存代码、模型配置、学生与 adapter 指纹；每组使用新仿真进程。报告包含实际指令、语言解析、视觉目标、反馈、状态转换、原始 VLM 请求及录像。

## 开发中修复的接口问题

最初把语言解析与视觉清点合并，真实模型在 6 回合中仅 2 回合正确解析目标，且会根据可见符号改写指令或在目标隐藏时输出未知。这轮开发失败保存在 `outputs/agentic_v3/dev_v1`，不计入验证结果。随后将文本解析与视觉定位拆开，保持同一个 VLM、记录两种实际请求，并共同计入调用预算。

## 验证设计与解释

独立小规模验证预先固定为 54 回合：2 个新种子、3 个场景/指令配置、feedback / periodic / oracle 各 18 回合。seed31101 的两种配置保持初始物理状态和布局，只改变目标及措辞；seed31102 使用改写指令。单组仅 18 回合，不报告统计显著性或论文级泛化结论。

另做 15 次真实 VLM 纯文本诊断（独立于 rollout 调用预算）：6 标准指令、6 改写、3 歧义或不支持指令。完整保存失败例。它不参与修改冻结验证期间的提示词。

两种多次观察模式的比较覆盖调用时机、观察动作时长及有界再观察的整体差异，不是单一模块消融。正式方法研究仍需拆开各因素。


## 冻结验证结果

9 个独立仿真进程全部正常结束，共 54 回合。代码/学生/adapter 指纹、完整指令进入实际请求、调用预算、每组 6 个唯一回合均通过审计。同一 seed 各模式及指令 offset 的初始物体位置、姿态、本体状态、符号布局最大差值均为 0。两种 seed 共有 12 个不同初始场景；每种模式的 18 条指令实例不等同于 18 个独立初始场景。

| 模式 | 成功 | 掉落 | 语言解析正确 | 最终视觉目标绑定正确 | VLM 调用（文本 / 视觉） |
|---|---:|---:|---:|---:|---:|
| 执行反馈 feedback | 4/18，22.22% | 4/18 | 16/18 | 5/18 | 66（18 / 48） |
| 周期观察 periodic | 4/18，22.22% | 3/18 | 16/18 | 5/18 | 86（18 / 68） |
| 真值目标 oracle | 15/18，83.33% | 2/18 | 不适用 | 真值直接给定 | 0 |

反馈版总调用少 23.26%，视觉调用少 29.41%；本轮没有成功率提升，且多发生 1 次掉落。18 回合/模式、2 个 seed 不支持统计优越性结论，调用节省也需要更大验证。oracle 绕过语言和视觉绑定；原始逐回合 `language_correct=true` 为评分器占位，汇总中置为 null，不能当作 VLM 语言准确率。

两种非 oracle 模式按目标的成功数相同：star、triangle、cross、diamond 各 1/3，circle、bars 各 0/3。这表明实现已覆盖六个目标接口，但尚未证明六类目标都能可靠完成。

### 语言与同场景换指令诊断

- 纯文本标准指令：6/6 正确。
- 改写指令：5/6 正确。bars 指令的原始输出为 `{"goal_symbol":"bars","goal_direction":"top"}`；严格接口只接受 `up`，因而拒绝执行。这里是输出枚举不符合约定，不能直接归因于不理解方向。两个 rollout 中重复出现这一拒绝。
- 歧义/不支持指令：2/3 正确拒绝。`Make the sticker without a star face upward.` 被错误解析为 `star/up`，否定指令尚不可靠。
- 同场景改变目标的 6 对指令：两种模式均有 5/6 对语言解析同时正确，但仅 1/6 对最终视觉绑定同时正确，0/6 对都成功。反馈版有 3/6 对改变最终技能，周期版为 4/6；仅改变技能不代表选对目标。

因此，当前最明显的问题是符号与贴纸颜色的视觉绑定，以及错误绑定之后的证据更新。底层 oracle 成绩较高，优先重训 student 的依据不足。本实验也不能直接与旧版固定 star 的 29.33% 比较，因为任务分布和调用预算不同。

### 一个真实纠错回合

独立验证 seed31101 / canonical / feedback / case3，实际指令为 `Turn the cube so the cross sticker faces upward.`，目标贴纸为 red。

1. step 70：误绑定 green，开始执行。
2. step 180：执行停滞触发一次有界再观察。
3. step 200：返回先前目标，继续收集证据。
4. step 220：出现 red 冲突候选；step 260 在另一张图像支持后修正为 red。
5. step 340：独立评分器判为成功，未掉落。

录像为 `outputs/agentic_v3/validation/seed31101_offset0_canonical_feedback/31101_feedback_case3_continuous.mp4`，170 帧 / 15 fps，约 11.3 秒仿真时间；推理等待不在视频内。该例展示已运行的纠错过程。周期版在同一回合也成功，不能据此断言再观察带来因果收益。反馈版共触发 3 次有界再观察，其中 2 回合成功。

## 证据与下一步

- [冻结协议与指纹](results/semantic-agent-v3/freeze.json)
- [完整汇总与成对状态检查](results/semantic-agent-v3/analysis.json)
- [真实语言诊断及失败回答](results/semantic-agent-v3/language_probe.json)
- [纠错案例所在组的全部原始回合](results/semantic-agent-v3/seed31101_offset0_canonical_feedback/report.json)
- [归档文件校验值](results/semantic-agent-v3/sha256.json)

每个场景/模式目录均归档原始报告与初始状态，共 9 组。视频保留在本地 outputs，未将模型权重或视频提交到 Git。开发集结果未混入验证汇总。11 项新接口测试和 7 项原有 agent 测试通过；这些使用 mock 的测试只验证代码契约，不是模型性能证据。

下一版本优先修复可接受方向词的规范化与不支持指令拒绝，改进小贴纸视觉定位/局部图像质量，并在执行前积累更可靠的匹配证据；之后以固定调用预算拆分调用时机、观察动作和纠错的消融。自主视觉成功认证、停止与抓持风险估计仍是后续研究内容。
