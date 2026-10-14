# 视觉语义 agent 原型：随机符号、证据记忆与主动观察

状态：开发中的仿真原型。语言 v2 基线保持冻结；这里是单独的新任务，不复用原语言成功率作为 VLM 结果。

## 任务与边界

- 固定英文任务：`Turn the cube so the star sticker faces upward.` 当前接口只支持星形目标，不代表开放词汇语言理解。
- 六个原有颜色贴纸上添加 star / circle / triangle / cross / diamond / bars；每回合随机排列符号，平衡星形所在的六种颜色。
- VLM 读取实际 RGB 全图和由 RGB 饱和度区域产生的裁剪。请求不包含目标编号、布局、姿态、奖励、误差或带标签文件名。
- 渲染器使用仿真姿态绘制符号；评分器使用真值计算结果；oracle 对照直接获得正确颜色。三者均不属于 agent 输入。
- 当前 VLM 为本地 Qwen2.5-VL-7B-Instruct，NF4 4-bit、greedy decoding。这是已有可运行模型的原型，不是尚未测试的 Qwen3-VL 实验。
- 底层复用冻结 RGB-D student 和其 canonical language embeddings。512 RGB-D 缩到 256 后采用原有特征函数，history 8、视觉 stride 2、21 维归一化动作、scale 1.0。新增贴纸和渲染分辨率改变了视觉分布，因此必须保留同场景 oracle 对照。

## 已实现的方法机制

1. **先观察再绑定目标。** VLM 列出可见小贴纸的颜色和符号，不向它暗示每张图都必须找到星形。唯一星形候选才进入目标绑定流程。
2. **符号证据记忆。** 按颜色保存不同观察图像支持的符号记录；unknown 不清空已确认信息。两个不同图像支持的非星形面可以暂时跳过。
3. **假设驱动主动观察。** 首次星形检测先作为候选，调用已训练的对应颜色控制技能，将其转到便于确认的视角。获得第二张不同观察的支持后才接受目标。目标冲突也需要两次支持。
4. **事件触发调用。** 初次观察、候选待确认、未解决目标的新视角、可见性下降、图像变化停滞触发 VLM。设置调用冷却时间和每回合最多 6 次调用；周期对照共享同一预算和观察技能。
5. **视觉验证与有界恢复。** 目标已知且图像停滞时，询问该颜色小贴纸是否在上表面。若回答可见但不朝上，最多执行一次 20 控制步的替代面动作，然后回到原目标。验证只提供粗粒度诊断，不能认证 0.16 rad 精度，也不会直接宣告任务成功。

这些机制是可检验的工程假设，尚不能据此宣称算法创新或性能提高。不同图像哈希仅防止完全重复图像投票，不等于统计独立视角；“可见颜色像素质量”也只是粗糙的运动代价代理。

## 对照与评分

| 模式 | 高层策略 | 是否使用目标真值 |
| --- | --- | --- |
| oracle | 直接调用正确颜色技能 | 是，仅作控制上限 |
| once | 初始 VLM 一次，无法定位则弃权 | 否 |
| periodic | 固定间隔重看，共享主动观察技能 | 否 |
| event_memory | 事件触发、证据记忆、确认、验证/恢复 | 否 |

periodic 也保留已经接受的目标、非星形证据记录和主动观察技能。event_memory 额外启用双图确认、事件调度与验证恢复。因此这里是系统策略比较，不是单独的记忆消融。

v2 评分要求：agent 已接受正确目标、正在执行该目标、姿态误差 ≤ 0.16 rad、没有掉落。搜索动作中碰巧经过正确姿态单独记作 incidental reach，不能算成功。每回合最多 600 控制步；成功时由独立评分器停止该槽位，因此当前结果仍不是自主视觉停止能力的证据。没有持续 hold 要求。当前 VLM 请求是同步阻塞的，等待期间仿真不推进；因此不能据此宣称真实机器人上的实时性或推理延迟鲁棒性。

每个 seed/mode 在新仿真进程中运行。记录初始姿态、位置、proprio 和布局，并检查各模式相同初始状态。GPU 物理与渲染可能仍有非确定性，因此小样本只能用于原型诊断。

开发种子：811。已在冻结代码后使用独立种子 912、1013，完成每模式 12 回合。这只是 pilot，不替代论文级多种子大规模评测。

## 已发现的问题

- 256 分辨率直接询问容易把其他符号误认成星形，并错误声称可见。512 分辨率改善了部分清晰例子，但没有解决遮挡和全部误认。
- VLM 有时返回 JSON 数组而不是外层对象；解析器已兼容数组，保留原始文本审计。
- 第一版开发评分会在搜索偶然经过目标姿态时提前结束；其 periodic 的 2/6 几何到达均没有正确 grounding，因此不能作为语义成功。该评分已替换，旧报告仅保留为诊断。
- 第一次在同一仿真进程连续切换模式时，复位阶段长时间没有进展；改用每模式独立进程。

## 运行入口

```bash
# 使用已安装 torch / transformers / IsaacLab 的 Python。
# bitsandbytes 与 accelerate 可安装到单独 target；不要覆盖仿真依赖。
PYTHONPATH="$PWD/outputs/vlm_dependencies" python scripts/agentic/vlm_server.py \
  --model /path/to/Qwen2.5-VL-7B-Instruct --quantize

python scripts/agentic/run_pilot.py --output outputs/agentic_v1/pilot \
  --seeds 912 1013 --modes oracle once periodic event_memory
python scripts/agentic/summarize_pilot.py outputs/agentic_v1/pilot
python scripts/agentic/verify_snapshots.py outputs/agentic_v1/pilot \
  --output outputs/agentic_v1/pilot/verification.json
PYTHONPATH=scripts/agentic python -m unittest scripts/agentic/test_visual_agent.py -v
```

模型权重不写入 Git。接口失败、预算用尽、弃权和失败回合均必须保留在结果中。

完整实测数据见[首轮结果报告](agentic_symbol_results_20261014.md)。
