# Visual student DAgger round 9

日期：2026-10-13  
分支：`experiment/viserdex-revo3-cube`

## 本轮变更

- 修正 `scripts/vision/collect_dagger_rollouts.py`：当 student 的 `language_mode=vlm` 时，采集器使用本地 SigLIP2 文本编码（768 维）生成六个固定 marker prompt 的语言条件，不再错误地写入 6 维 one-hot。
- teacher、freeze manifest、action scale 和评测协议保持冻结。teacher 为 `model_1999.pt`，SHA256 为 `193129f874f81bca34a42e309c77612eb11ea29d7e2b4a60a5e8870536f9f707`；action scale 为 `1.0`。

## 采集与训练

- 在原始 8×RTX 5090 主机上运行 targeted DAgger：180 episodes，target faces `1,3,5`，32 environments，vision stride `2`，RGB-D semantic grid `16×16`，depth grid `16×16`。
- 新增 shard：10,732 samples；与历史缓存合并后共 163,389 samples。
- evidence student 训练 40 epochs，batch size 512，learning rate `3e-4`，seed `5`，最终 train MSE `0.10775`，validation MSE `0.39539`。
- 最终 checkpoint SHA256：`65945cec10552eb646382c374217a31d714d9d50ad3353e7e3d04f153a35ffd9`。

## 384 集闭环评测

固定协议：RGB-D semantic camera、SigLIP2 language mode、vision stride `2`、action scale `1.0`、32 environments、384 episodes。

- success rate：**70.5729%**（271/384）
- drop rate：19.5313%
- per-face success：face0 69.23%，face1 77.61%，face2 71.70%，face3 75.00%，face4 70.91%，face5 59.72%
- 相比之前 120 集基线 68.33%，本轮提升约 2.24 个百分点；face5 仍是主要短板。

## NAS 归档

- checkpoint：`/mnt/data_nas/jiaju/viserdex_siglip_eval/outputs/visual_student_v3/dagger_target135_r9_sigliptext_evidence_seed5_student.pt`
- DAgger shard：`/mnt/data_nas/jiaju/viserdex_siglip_eval/outputs/visual_student_v3/dagger_target135_r9_sigliptext_evidence_seed5.pt`
- 评测报告：`/mnt/data_nas/jiaju/viserdex_siglip_eval/reports/siglip_seed5_4090_r9_large384.json`
- 评测报告 SHA256：`25242f75af7a80f0b0766001deef730a0e3fac942226831cfdc0e3e38519345c`

