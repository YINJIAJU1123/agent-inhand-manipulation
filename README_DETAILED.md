# REVEAL 代码交接详细说明

这份说明供接手实验的同事使用。短入口见 [README](README.md)。源码整理自当前研究工作区；`configs/source_snapshot.json` 记录保留文件的原始哈希。交接工具是新增辅助层，核心控制、评分和权重张量未重写。

## 1. 系统怎样工作

语言指令和当前图像进入上层语义系统，输出检查动作、目标绑定和目标技能。公共工具参数通过固定 goal bank 变成连续 latent；微调 VLM 则直接预测 latent。底层 student 将当前测量、历史和目标 latent 转成关节目标。Astra 请求等待期间仍推进物理与本地控制；回复到达后再更新语义目标。最后由停止规则和独立真实姿态评分判断成功。

三种方法共享对应物体的目标表示和 student，但其历史、检查策略和控制模块不同。完整 Astra 与裸基线的比较是**完整系统比较**，不能只解释为上层模型大小的作用。

| 物体 | 语义图像 | 底层位姿 | 逻辑控制频率 | Astra 停止配置 |
|---|---|---|---:|---|
| Cube | 原生 120×120 RGB | 冻结 RGB CNN | 30 Hz | measured，0.16 rad |
| Die | 512×512 RGB | 带噪仿真位姿 | 20 Hz | 物体专属 learned correction |
| Duck / Hedgehog / Pig / Milk carton | 512×512 RGB | 带噪仿真位姿 | 20 Hz | 物体专属 learned correction |

裸基线不加载额外 actor / completion 权重。Cube 的等待和执行 actor 是不同权重；不要回退到历史 completion gate。不同物体的 student、LoRA、goal bank、actor 与 stop 权重不可混搭。成功要求正确语义绑定、终止时真实 canonical quaternion 误差不超过 0.16 rad，且没有终止掉落；仅中途经过目标不算成功。

## 2. 目录和阅读顺序

```text
scripts/
  pose_latent_student_v1/       Cube 模型、CNN 位姿、控制、训练、评测
  dice_latent_v1/               骰子 runtime、feature student、控制、训练、评测
  inhand4_q1/                   四个标记物体的 runtime、训练、评测
  agentic/                     Qwen backbone、crop、latent head、HTTP 推理
  astra_transport_v2/          既有 Responses transport 和 loopback client
  multilingual_eval_v1/        中英文与措辞模板生成
  pose_latent_studies_v2/      核心依赖：路径适配和视觉干预 renderer
  diagnostics/                配置审计与完成后的语言结果汇总
  handoff/                    新交接入口：校验、命令生成、路径适配、API bridge
configs/
  objects.json                六物体的相对路径和方法配置
  bundle_reference.json       本分支对应 artifact manifest 的哈希
  source_snapshot.json        核心源码来源哈希
  runtime_versions.json       实验环境的主要版本记录
```

建议先读 `evaluate_agent.py` 的主循环，再读 `runtime.py` / `newton_runtime.py`、`semantic.py`、`control_modules.py`，最后看 `grounding.py` 和训练脚本。脚本保留研究模块的命名，不必从历史实验目录猜依赖。

## 3. 独立 artifact 包

```text
reveal_artifacts_20261106/
  BUNDLE.json
  MANIFEST.json
  objects/
    cube/          source student、latent student、pose CNN、相机标定、goal cache、两个 actor、grounding
    die/           source、latent student、环境、goal bank、actor、completion、grounding
    duck/          同上，另有 marked_assets 和原始 relocation 声明
    hedgehog/
    pig/
    milk_carton/
  runtime/
    cube/          该版本 ViserDex 运行代码和资产
    die/           骰子的运行代码和资产
    inhand4/       四物体共享的原始来源目录
    IsaacLab/      对应实验版本的源码和依赖锁
    inhand4_source_inventory.json
```

每个物体的 `OBJECT.json` 说明对应文件。`grounding/goal_head.pt` 与 `grounding/lora/` 必须一起使用。没有打包 Qwen 基础模型、Python 虚拟环境、CUDA 驱动、训练数据或作者登录信息。上游运行目录的历史 README 可能讨论别的开发目标；本交接说明和 `OBJECT.json` 才定义本次部署用途。

包解压在 Git 仓库外，随后执行：

```bash
export REVEAL_BUNDLE=/absolute/path/reveal_artifacts_20261106
python3 scripts/handoff/check_bundle.py --bundle "$REVEAL_BUNDLE"
```

校验会读取全部文件并检查本分支绑定的 manifest。路径适配只在 `outputs/handoff_cache/` 生成派生配置／视觉资产；原始权重、source inventory 和 canonical 环境定义不改。部分原始 artifact 保留历史绝对路径作为哈希合同标识，不能手工全局替换；启动器负责加载时的正反向映射。

## 4. 环境准备

已运行实验使用 Linux、Python 3.12.13、PyTorch 2.11.0+cu130、Isaac Sim 6.2.0.0、Isaac Lab 包版本 28.0.0、Newton 1.6.1、Warp 1.17.0、Transformers 5.10.4、PEFT 0.21.2、bitsandbytes 0.50.2。Isaac Lab 源码 revision 在 `BUNDLE.json` 中。历史文档中的 Isaac Lab 3.0/Newton 名称不应被理解为可以任意安装另一个同名版本。

优先复用同事已经准备好的匹配环境，并设置：

```bash
export VISERDEX_PYTHON=/absolute/path/compatible_env/bin/python
export UV_BIN=/absolute/path/uv
export REVEAL_BUNDLE=/absolute/path/reveal_artifacts_20261106
export QWEN_MODEL=/absolute/path/Qwen3-VL-4B-Instruct
```

新建环境可从包内 `runtime/IsaacLab/pyproject.toml`、`uv.lock` 和对应 runtime 的 `requirements/` 开始，使用该锁文件与匹配的 CUDA 13 wheel；需要满足上游许可证/EULA及驱动要求。不要把 `pip install isaaclab` 当作完整安装方式，也不要把已有虚拟环境直接复制到不同路径后就认为所有 editable 安装和 CUDA 链接都有效。

本次未在空白机器上重新安装整套环境，所以这里记录的是**已使用的版本与依赖入口**，不是已验证的一键安装承诺。`run_newton.sh` 会核对主要版本，启动器为所选物体选择相应 ViserDex 源码。底层可视化／Gaussian renderer 仍依赖上游扩展，具体安装说明随 runtime 源码保留。

基础模型使用 `Qwen/Qwen3-VL-4B-Instruct`；须从你有权访问的模型来源单独取得完整目录（配置、processor/tokenizer、分片权重）。不能只提供 LoRA。没有悄悄下载模型或回退其他模型的逻辑。

## 5. 跑一个开发回合

以下两条命令分别在两个终端执行。先确认 GPU 是你有权使用的卡，且留有足够显存。每个终端使用同一环境，服务端口应独占。

**终端 A：启动 Duck 的微调 VLM。**

```bash
"$VISERDEX_PYTHON" scripts/handoff/launch.py \
  --bundle "$REVEAL_BUNDLE" --object duck --method vlm --mode service \
  --model "$QWEN_MODEL" --gpu 0 --port 19081 --execute
```

等服务日志出现 ready，再执行终端 B：

```bash
"$VISERDEX_PYTHON" scripts/handoff/launch.py \
  --bundle "$REVEAL_BUNDLE" --object duck --method vlm \
  --gpu 0 --port 19081 --seed 990601 --episodes 1 \
  --output outputs/duck_vlm_dev_001 --execute
```

去掉 `--execute` 只检查路径并打印命令，不加载 GPU、不发 API 请求。输出目录必须是新的；入口拒绝覆盖。支持的物体名是 `cube`、`die`、`duck`、`hedgehog`、`pig`、`milk_carton`。切换物体时，两边的 `--object` 都要改，换一个端口或关闭你自己的旧服务。Rule 方法同样先以 `--method rule --mode service` 启服务，再启动 `--method rule` 评测。

GPU 编号对两个终端使用的是同一个可见宿主编号，启动器设置 CUDA 和 headless GL 设备；不要先套一层不同的 `CUDA_VISIBLE_DEVICES` 后再混用物理编号。所有 HTTP 服务只绑定 loopback。没有自动抢卡、SSH 登录或终止别人进程的代码。

## 6. Astra 的 API bridge

必须使用你自己的凭据。API 路由保留实验原来的 `gpt-6-astra` 模型标识，账户需要实际支持它；不可用时会失败，不会静默换模型。不同服务版本、认证方式和延迟可能改变结果，不能保证重现原成功率。

```bash
# 通过你自己的凭据管理方式设置 OPENAI_API_KEY；不要写入仓库。
# OPENAI_BASE_URL 可选，默认 https://api.openai.com/v1；必须为 HTTPS。
"$VISERDEX_PYTHON" scripts/handoff/api_bridge.py \
  --object duck --port 19080 --run-dir outputs/duck_api_bridge_001
```

另一个终端：

```bash
export ASTRA_BRIDGE_URL=http://127.0.0.1:19080
export ASTRA_BRIDGE_KEY_FILE="$PWD/outputs/duck_api_bridge_001/bridge.key"
"$VISERDEX_PYTHON" scripts/handoff/launch.py \
  --bundle "$REVEAL_BUNDLE" --object duck --method astra --gpu 0 \
  --seed 990601 --episodes 1 --output outputs/duck_astra_dev_001 --execute
```

Bridge key 只是本地服务的随机访问口令，不是 API key。bridge 的图像合同按物体选择：Cube 为 120×120，其余为 512×512。日志可能包含实验图像、指令和模型响应；不要把 `outputs/` 提交到 Git。

## 7. 训练、语言评测与正式复现

- Cube：`prepare.py` / `train.py` 对 student 做适配；`collect_control.py`、`train_modules.py`、`train_completion.py` 处理本地模块；`grounding.py` 训练语义 latent head / LoRA。
- Dice / 四物体：`collect_rgb.py`、`label_rgb.py`、`grounding.py` 处理视觉 grounding；`fit_features.py`、`collect_modules.py`、`train_modules.py`、`train_completion.py` 处理 student / 控制模块。
- 训练数据、teacher rollout 和正式候选选择历史没有全部打包。训练代码的参数需要对应采集清单，不能把一个推理权重目录当作完整训练集。
- `multilingual_eval_v1/protocol.py` 保留中英文标准／改写模板。旧脚本默认种子属于历史研究；新实验要建立独立 manifest，避免与旧结果混合。
- `diagnostics/report_aligned_cube_language.py` 和 `report_aligned_object_language.py` 针对对应完整 720 次目录做配对和 Holm 汇总。它们不是任意开发输出目录的通用计分器。

当前交接入口只生成开发评测命令，不自动复跑整个正式队列。正式复现要冻结新机器的源码、路径适配、所有权重、运行配置、初态/图像配对、种子和服务版本。原正式 release 的 source hash 与整理分支可能不一致；不能删除校验或伪造通过来复用旧 release。服务失败保留，不按结果挑样本、重跑失败直到成功，也不把不同队列直接合并。

## 8. 常见问题

- **源码/权重 hash 不匹配**：先运行 `check_bundle.py`，确认 branch 与 bundle 是同一版；不要用文件名代替哈希确认。
- **路径仍指向旧机器**：从交接入口启动，不直接手工改权重或环境 YAML；查看派生 relocation 和 object metadata。
- **`viserdex` 或 renderer 缺失**：使用匹配的模拟环境和所选物体 runtime 源码。三个 runtime 版本不能随意互换。
- **服务 backend / frame_size 不匹配**：服务与评测物体不一致，或端口连到了旧服务；Cube 与其他物体的图像尺寸不同。
- **CUDA/OpenGL interoperability 警告**：可能回退 CPU 拷贝；检查指定 GPU、输出图像和时延，不能仅凭进程存活判断正常。
- **接口失败或超过 120 秒**：记录服务问题，不能解释为机械控制失败，也不能静默替换该正式样本。
- **重跑输出目录已存在**：使用新的目录，保留原记录。

## 9. 当前交接验证

已对核心 Python 文件做语法检查，对六物体 × 三方法的开发命令和关键停止配置做离线验证，并校验全部独立 artifact 文件。没有启动额外 GPU 仿真，没有进行同事账户 API 调用，也没有宣称这个整理分支已经在新机器端到端验证。现有正式实验使用的工作目录、冻结代码和运行进程不受本次交接整理影响。

离线复核命令（不启动仿真或 API）：

```bash
REVEAL_TEST_BUNDLE="$REVEAL_BUNDLE" python3 -m unittest discover -s tests -v
```
