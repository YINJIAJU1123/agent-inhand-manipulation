# Visual student DAgger round 10

日期：2026-10-13  
分支：`experiment/viserdex-revo3-cube`

## 采集与训练

- 针对 face5 运行 240 episodes，冻结 teacher、SigLIP2、action scale、freeze manifest 和采集协议。
- 新增 12,819 samples；合并后特征缓存共 176,208 samples。
- evidence student 训练 40 epochs，batch size 512，learning rate `3e-4`，seed `5`。
- 最终训练 checkpoint SHA256：`28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7`。

## 两个独立 seed 的闭环评测

协议保持一致：RGB-D semantic camera、SigLIP2 `language_mode=vlm`、vision stride `2`、action scale `1.0`、32 environments、每个 seed 384 episodes。

| seed | success rate | drop rate |
|---:|---:|---:|
| 123 | 80.73% (310/384) | 15.10% |
| 456 | 79.43% (305/384) | 16.67% |
| 合计 | **80.08% (615/768)** | **15.89%** |

face5 在两个 seed 上分别为 86.67% 和 83.61%，说明针对性补采解决了上一轮的主要短板。当前结果相对 round 9 的 70.57%（单个 384 集评测）提升约 9.5 个百分点；后续论文表格应继续使用多 seed、分面均衡的协议。

## NAS 归档

- checkpoint：`/mnt/data_nas/jiaju/viserdex_siglip_eval/outputs/visual_student_v3/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt`
- DAgger shard：`/mnt/data_nas/jiaju/viserdex_siglip_eval/outputs/visual_student_v3/dagger_target135_r10_face5_sigliptext_evidence_seed5.pt`
- seed 123 报告：`/mnt/data_nas/jiaju/viserdex_siglip_eval/reports/siglip_seed5_4090_r10_seed123.json`
- seed 456 报告：`/mnt/data_nas/jiaju/viserdex_siglip_eval/reports/siglip_seed5_4090_r10_seed456.json`
- seed 123 报告 SHA256：`f3156a22bf9c86f5108347d49f3b1eb7a631cf4efb2945546f69dcc838912668`
- seed 456 报告 SHA256：`326aeb0f03b1febe914a630d982ee2a55b412dc4fdc7532918a5c5da7486ad5a`
