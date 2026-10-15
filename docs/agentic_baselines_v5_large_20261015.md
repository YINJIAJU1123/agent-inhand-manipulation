# 规则与微调 latent：900 回合冻结评测

状态：2026-10-15 已启动有限后台实验，最终结果尚未生成。运行目录为 `outputs/agentic_v4/large_eval_20261015`；不创建 scheduled task。

## 对照定义

| 模式 | 模型与决策 | 每模式回合数 |
|---|---|---:|
| Baseline1 / periodic | 冻结 Qwen2.5-VL-7B 解析语言、清点可见符号，再由原 v3 周期观察与证据规则选择已有技能 | 300 |
| Baseline2 / latent | 冻结 Qwen3-VL-4B QLoRA latent_v2，直接执行预测的连续 768 维目标向量 | 300 |
| Oracle | 直接提供正确目标，检查冻结 student 的执行表现 | 300 |

Baseline1 是含 VLM 的符号输出与规则控制系统，并非完全不使用模型的纯规则。继承原来的语言/视觉提示词、证据路由与调用分配，只把局部裁剪统一为与 latent 一致的 RGB 裁剪。没有用新的测试结果调整模型或阈值。

两条系统使用不同基座、训练方式、目标表示与高层规则。因此本轮可评价系统效果，不能单独归因于微调、模型大小或 latent 表示。要隔离学习收益，后续还需要同基座的未微调/仅 head/LoRA 对照。

## 协议

- seeds35001–35025，共 25 个新种子；每个种子六类目标。
- 每个物理场景测试同目标的 canonical 与 paraphrase 指令，初始状态和布局保持不变。
- 每种模式 300 回合，对应 150 个不同初始物理场景；不是 300 个独立场景。
- 三种模式共 900 回合、150 个独立仿真进程。
- 同一冻结 student、action scale=1.0、RGB-D 历史、六个技能及成功评分器。
- 每回合最多 600 控制步、6 次高层调用。Baseline1 的一次语言解析也计入预算，因此最多五次视觉调用；latent 每次联合输入语言与图像。
- 训练/验证/测试的符号布局库互斥；新评测不用于补采或模型更新。
- 主成功仍要求正确语义绑定、正在执行正确目标、完整姿态误差 <=0.16 rad 且未掉落。无 hold 条件，由评分器停止，不证明 agent 自主停止。
- 推理期间仿真暂停；本实验不证明实时部署。

统计按符号、措辞、目标绑定、掉落、调用和推理时间展开；比较逐场景配对结果。95% 区间以整个 seed 为重采样单位，包含该 seed 的全部目标和措辞，避免把重复措辞当作独立样本。过程中的 `analysis.json` 明确标记 partial，最终才标 complete。

## 可复现与运行状态

[冻结协议](../configs/agentic_baselines_v5_large.json)；[启动器](../scripts/agentic/run_baselines_v5.py)；[统计脚本](../scripts/agentic/analyze_baselines_v5.py)。

```bash
PYTHON=/home/yin/Grasping/Grasppose/env_isaaclab/bin/python
$PYTHON scripts/agentic/run_baselines_v5.py --output outputs/agentic_v4/large_eval_20261015
$PYTHON scripts/agentic/analyze_baselines_v5.py outputs/agentic_v4/large_eval_20261015
```

启动器冻结协议、源代码、student、adapter、LoRA、goal head、规则模型权重和符号资产指纹；每组完成后审计回合和配对初始状态，写入 analysis.json。支持同版本断点恢复，拒绝代码、协议或关键检查点变化；文件锁防止同目录重复运行。只管理自己启动的模型服务与仿真进程。单组 watchdog 为 1800 秒。

查看 `progress.json`、`launcher.log`、各组 `.log`。seed35001 的所有模式和两种措辞均录像，成功和失败例全部保留。大模型及图片保留在 outputs。

## Astra 账号说明

当前进程未配置 OPENAI_API_KEY，尚未对该账号进行真实 Astra API 请求。官方文档明确区分 ChatGPT 订阅登录与 API key 的按量计费方式；一般 OpenAI API 调用使用 Platform API key。当前 Codex 登录状态不能直接证明脚本拥有 API key、额度或模型权限。

如果用户在同一账号的 OpenAI Platform 项目中已有 API key，且项目可使用 gpt-6-astra 并具备额度，就可用于这条接口；是否具备这些权限仍需实际 API 请求核实。不读取或转换 Codex 登录 token，不在聊天中提供密钥。

来源：[官方认证说明](https://learn.chatgpt.com/docs/auth)、[Astra 模型文档](https://developers.openai.com/api/docs/models/gpt-6-astra)。
