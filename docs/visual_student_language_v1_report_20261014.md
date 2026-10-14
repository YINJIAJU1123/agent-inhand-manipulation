# Visual student rule-routed baseline v1 (corrected)

日期：2026-10-14  
分支：`experiment/viserdex-revo3-cube`

## 结论更正（2026-10-14）

本轮 79.30% **不能作为学得的语言 grounding 或改写泛化证据**。实际执行时，评测器从原始指令解析颜色，并把颜色索引传入 `_canonicalize_prompt_features`，直接选择 canonical embedding；改写后的 SigLIP2 文本向量被绕过。下面的数值保留为规则颜色路由下的视觉控制结果。此前将其描述为“语言训练已完成”不准确。

## 版本定义

这一版把语言接入放在冻结的 RGB-D student 之上：

- teacher、RGB-D frontend、GRU、action head、action scale `1.0` 和评测协议保持不变；
- SigLIP2 `siglip2-base-patch16-224` 只负责把指令编码成 768-D 文本向量；
- `siglip2_prompt_canonicalizer_v1` 存储模板向量，但本轮实际通过规则解析的颜色索引选择 canonical embedding；
- canonical embedding 和评测脚本使用同一个 slow tokenizer 和同一模板 batch，避免 SigLIP2 文本 pooling 的 batch padding 偏差；
- 控制器参数未改；本轮属于规则颜色路由的控制基线，不能证明学得的语言理解。

模板集合：

1. `show the {color} marker`
2. `reorient to the {color} marker`
3. `turn toward the {color} marker`
4. `bring the {color} marker to the top`
5. `move the {color} marker into view`
6. `place the {color} marker upward`

## 闭环评测

统一配置：RGB-D semantic camera（RGB grid 16、depth grid 16）、vision stride 2、32 environments、384 episodes、action scale 1.0、seed 123/456。

| prompt style | seed | success | drop |
| --- | ---: | ---: | ---: |
| canonical | 123 | 77.34% (297/384) | 18.23% |
| `reorient to ...` | 123 | 78.91% (303/384) | 15.63% |
| `reorient to ...` | 456 | 79.69% (306/384) | 17.45% |
| **rewrite aggregate** | **123+456** | **79.30% (609/768)** | **16.54%** |

各 face 的 rewrite success rate：

- seed 123：face0 73.13%、face1 74.63%、face2 85.25%、face3 77.27%、face4 86.15%、face5 77.59%；
- seed 456：face0 80.00%、face1 78.46%、face2 66.67%、face3 85.00%、face4 85.71%、face5 86.15%。

canonical 与 rewrite 对同一颜色实际提供相同的控制器输入，因此两者的差异不能用于判断语言改写效果。round 10 原始基线为 80.08%（615/768）；这里只保留控制成功率的数值比较。

## 文件与校验

远端项目：`/home/jiaju/src/RevoLab`

- base student：`outputs/visual_student_v3/checkpoints/dagger_target135_r10_face5_sigliptext_evidence_seed5_student.pt`  
  SHA256 `28f62f3e1f8590cd57479a89aff5d36f054b6c49d19e544a7820d72d10f8d6f7`
- prompt canonicalizer：`outputs/visual_student_v3/checkpoints/siglip2_prompt_canonicalizer_v1.pt`  
  SHA256 `945c3ff59f51c6fabe5ce9f560f11f61edddb1ae9515953854db5b8dc96c4b3a`
- canonical report：`outputs/visual_student_v3/reports/dagger_target135_r10_face5_sigliptext_promptcanon_v1_eval_canonical_seed123.json`  
  SHA256 `1c31d8351219b39dfe758f9a152118a7f4ef132f588e6f8a2d3bcbc871e9d75a`
- rewrite seed123 report：`outputs/visual_student_v3/reports/dagger_target135_r10_face5_sigliptext_promptcanon_v1_eval_reorient_seed123.json`  
  SHA256 `ebc34590b1d9d2a3997f55c5b44d7def186c149709f02f22a3e2e843de86e4a2`
- rewrite seed456 report：`outputs/visual_student_v3/reports/dagger_target135_r10_face5_sigliptext_promptcanon_v1_eval_reorient_seed456.json`  
  SHA256 `ae47a32995852d45b1a19628d7e7622a1906fde63aedf0e74f85fd7329361c89`

NAS mount 在本次结束时未挂载（`/mnt/data_nas` 不存在），因此本轮文件暂时保留在远端项目 outputs 中；重新挂载 NAS 后可按上述相对路径直接归档，校验值已记录。
