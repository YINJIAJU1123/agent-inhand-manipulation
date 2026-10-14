# 随机符号 VLM：扩展评测与录像

## 已完成的评测

在本地 RTX 3090 上运行同一个固定星形指令任务。冻结学生、VLM、提示词与策略参数，不在评测过程中训练或调参。

- 新种子 20001–20050，共 50 个；每个种子平衡六种目标颜色。
- 三种策略：periodic、once、oracle；每组 300 回合，总计 900 回合。
- 每回合最多 6 次 VLM 调用、600 控制步。
- 主指标：正确接受目标后执行到达姿态阈值且不掉落。偶然经过目标不算成功。
- 次指标：掉落、正确绑定、弃权、调用次数、动作步数、分目标颜色结果。
- 以 seed 为单位进行配对 bootstrap，保留每个 seed 内的六个目标，报告 95% 区间。未完成的组不混入完整评测结论。
- 各组检查初始姿态、位置、proprio 和符号布局一致。
- event_memory 暂留开发集修正调用分配与确认延迟，不以扩大测试集代替方法修正。

这是对原 pilot 的分布内扩展，并不证明新物体、开放词汇指令或新视觉外观上的泛化。

## 最终结果

2026-10-14 北京时间 16:59 完成，150/150 个任务、900/900 回合。

| 策略 | 成功 | 成功率 | 掉落 | 正确目标绑定 |
| --- | ---: | ---: | ---: | ---: |
| periodic | 88/300 | 29.33% | 130/300 | 97/300 |
| once | 14/300 | 4.67% | 7/300 | 19/300 |
| oracle | 249/300 | 83.00% | 45/300 | 300/300 |

periodic 相对 once 提高 24.67 个百分点；配对 seed cluster bootstrap 95% 区间为 +20.00 至 +29.33 个百分点。periodic 成功率区间为 25.00–33.67%。50 个种子的初始物理状态与布局审计一致；跨进程 RGB 渲染不保证逐像素一致。

once 有 267/300 回合初始弃权，故上述差异包含更多观察、调用与运动的联合收益，不能单独归因于事件触发或记忆。

实际 `inventory` 分支让 VLM 清点符号，程序固定筛选 star；完整用户任务文本没有传入该视觉请求。因此这是固定语义目标的 VLM 视觉基线，不是多目标自然语言评测。后续多语义版本见 [agent v3](agentic_semantic_v3_20261014.md)。

最终机器可读统计位于 `outputs/agentic_v1/large_eval_v2/analysis.json`，保留全部失败与弃权回合。

## 录像

历史 pilot 只保存了关键帧。`render_keyframe_replays.py` 将其制作为保持时间间隔的回放：不插值、不生成中间动作，始终标注“非连续录像”。报告中的成功或掉落结论与最后一张保存的图像分开标注。

扩展评测对第一个种子 20001 的三个策略各六个回合保存连续 RGB 流。复用每两步已经采集的 RGB，15 fps，时间轴为模拟时间，不含同步 VLM 推理等待；没有新增渲染或物理步骤。文件为每个回合的 `*_continuous.mp4`，报告附帧数、fps 和路径。末帧是最后可用的动作前图像，终止后环境复位画面不属于该回合。

全部首批回合均录制，既有成功也有失败；展示用视频不参与统计筛选。

## 运行与恢复

```bash
python scripts/agentic/run_large_eval.py \
  --output outputs/agentic_v1/large_eval_v2 \
  --protocol configs/agentic_symbol_large_eval_v2.json \
  --model /path/to/Qwen2.5-VL-7B-Instruct
python scripts/agentic/analyze_large_eval.py outputs/agentic_v1/large_eval_v2
```

Python 环境需要 IsaacLab、torch、transformers；项目的 `outputs/vlm_dependencies` 提供 bitsandbytes/accelerate，`outputs/video_dependencies` 提供 imageio-ffmpeg。

启动器会启动并最终关闭自己拥有的 VLM 服务。每个 seed/mode 使用新仿真进程，单任务超时 900 秒；出错则停止并写出原因，不将缺失回合算成成功。重新运行相同命令时验证代码/协议指纹并跳过完整结果。它是当前机器上的一个进程，不是 scheduled task。

状态文件：`outputs/agentic_v1/large_eval_v2/progress.json`；逐组日志和报告也位于该目录。`running_counts.json` 是中间计数，最终不确定性与配对分析由 `analyze_large_eval.py` 输出。
