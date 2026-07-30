# Online Multi-Expert Time-Series Forecasting

本项目面向概念漂移场景下的在线多变量时间序列预测。核心方法 `multi_expert` 将 FSNet 与 FSNet-Time 专家、通道级 MoE 路由和 Temporal Smoothness Buffer（TSB）结合，在统一的滚动延迟反馈协议下完成离线预训练和在线适应。

仓库还保留了 DynaME、PatchTST-DGrad、ER、FSNet、OneNet、DSOF 和 PROCEED 的本地适配实现，用于在相同数据接口和在线协议下进行比较。这里的 baseline 是嵌入本项目结构后的实现，不是对应上游仓库的完整镜像。

> 本 README 只描述未被 `.gitignore` 排除的源码和实验入口。数据、checkpoint、日志和结果文件均为本地运行产物，不纳入版本控制。

## 核心方法

默认的 Multi-Expert 包含：

- 2 个 FSNet 专家，建模跨变量表示；
- 2 个 FSNet-Time 专家，建模每个变量的时间模式；
- channel 或 factorized horizon-channel Router；
- 可选 top-k 路由与逐 horizon/channel 的快速 correction `z`；
- version-aware Stable/Recovery memory 与 Recovery replay；
- 责任条件化 prediction-head capability subspace；
- `plain`、`tsb`、`subspace`、`hybrid` 四种 Expert 更新策略；
- 专家与路由器独立的离线、在线学习率和梯度裁剪。

主要实现位于：

```text
exp/exp_multi_expert.py
models/ts2vec/fsnet.py
models/ts2vec/fsnet_.py
```

## 滚动延迟反馈

预测起点始终每次前进一个时间点。只启用 `--delay_fb` 时保留旧的完整窗口延迟协议。额外启用 `--progressive_fb` 后，origin `s` 会在预测之前释放当前输入最后一个观测点，并将其分配给所有满足 `h=s-o` 的历史预测：

```text
origin 0: predict origin 0
origin 1: release origin 0 / h=1, then predict origin 1
origin 2: release origin 0 / h=2 and origin 1 / h=1, then predict origin 2
```

这意味着：

- record 不保存尚未成熟的完整未来标签；
- 每个成熟位置立即更新快速 Router correction `z` 和 Neural Router 的局部 credit；
- Expert 仍只在 record 的全部 `H` 个 target 成熟后更新；
- 测试结束不人为 flush 尚未成熟的 record；
- 延迟模式不会再通过 `index * pred_len` 跳过中间窗口；
- 所有 capability 对比和 memory 重评 forward 都禁止更新 FSNet 持久状态。

在线测试建议固定 `--test_bsz 1`。完整 progressive origin 顺序为：先衰减 `z`，释放成熟位置，更新 `z` 和 Neural Router；完整 record 成熟后计算责任与版本对齐，用旧 subspace 完成 supervised + Recovery Expert 更新，再将该 record 写入 memory；随后周期刷新 memory/subspace，最后预测当前 origin。当前预测窗口的未来真值仅用于离线指标，不进入任何更新或 record。

Subspace 第一版的保护范围严格限制为 prediction head：`ExpertNet.regressor` 与 `FSNetTimeExpertNet.regressor_time` 的 weight；bias 和 encoder 参数不投影。对应配置固定为 `--subspace_scope regressor`。

## 已实现方法

| 方法 | `--method` | 本地实现 |
|---|---|---|
| Multi-Expert | `multi_expert` | `exp/exp_multi_expert.py` |
| DynaME | `dyname` | `exp/exp_dyname.py`, `models/dyname.py` |
| PatchTST-DGrad | `patchtst_dgrad` | `exp/exp_patchtst_dgrad.py`, `models/patchtst_dgrad.py` |
| Experience Replay | `er` | `exp/exp_er.py`, `models/er.py` |
| FSNet | `fsnet` | `exp/exp_fsnet.py`, `models/fsnet.py` |
| OneNet | `onenet` | `exp/exp_onenet.py`, `models/onenet.py` |
| DSOF | `dsof` | `exp/exp_dsof.py`, `models/dsof.py` |
| PROCEED | `proceed` | `exp/exp_proceed.py`, `models/proceed.py` |

ER、FSNet、OneNet、DSOF 和 PROCEED 共用 `exp/exp_stream_baselines.py` 中的训练与在线测试框架。

## 环境安装

推荐 Python 3.10。当前环境验证版本为 Python 3.10.20 和 PyTorch 2.11.0+cu128。

```bash
conda create -n online python=3.10 -y
conda activate online
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

`requirement.txt` 不固定 PyTorch 的 CUDA 构建后缀。使用 GPU 时，可以先安装与本机驱动匹配的 PyTorch，再安装其余依赖。已安装的 `2.x` PyTorch 会满足依赖约束。

## 数据准备

数据 CSV 不纳入版本控制，需要自行放入 `data/`。主要实验使用以下文件名：

```text
data/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── WTH.csv
└── ECL.csv
```

CSV 第一列必须是 `date`，其余列为数值变量。`main.py` 中的主要数据映射为：

| 数据集 | 文件名 | M 模式变量数 | 默认目标列 |
|---|---|---:|---|
| ETTh1 | `ETTh1.csv` | 7 | `OT` |
| ETTh2 | `ETTh2.csv` | 7 | `OT` |
| ETTm1 | `ETTm1.csv` | 7 | `OT` |
| ETTm2 | `ETTm2.csv` | 7 | `OT` |
| WTH | `WTH.csv` | 12 | `WetBulbCelsius` |
| ECL | `ECL.csv` | 321 | `MT_320` |

数据划分定义在 `data/data_loader.py`：

- WTH、ECL 等 `Dataset_Custom` 数据使用 20%/5%/75% 的 train/validation/test 时间划分；
- 标准化统计量只由训练段拟合；
- ETT 数据使用 loader 中定义的固定时间边界；
- train/validation 不受 `delay_fb` 影响；test 始终以 stride=1 生成窗口。

## 运行 Multi-Expert

仓库保留的实验入口为：

```bash
bash scripts/run_multi_expert.sh
```

当前默认运行 4 个数据集 × 3 个预测步长，共 12 个任务：

| 配置 | 默认值 |
|---|---|
| 数据集 | ETTh2、ETTm1、WTH、ECL |
| `seq_len` | 60 |
| `pred_len` | 1、24、48 |
| 特征模式 | M |
| 训练 batch size | 32 |
| 测试 batch size | 1 |
| 训练轮数 | 15 |
| early-stopping patience | 3 |
| 在线模式 | full |
| 延迟反馈 | 开启 |
| 专家数 / top-k | 4 / 4 |
| 预训练模式 | retrain |
| 单 GPU 默认并发 | 2 |

脚本优先使用 `PYTHON` 指定的解释器。为了确保使用当前 Conda 环境，推荐：

```bash
PYTHON="$CONDA_PREFIX/bin/python" \
GPU_IDS=0 MAX_PER_GPU=2 \
bash scripts/run_multi_expert.sh
```

多 GPU 时使用逗号分隔：

```bash
PYTHON="$CONDA_PREFIX/bin/python" \
GPU_IDS=0,1 MAX_PER_GPU=1 \
bash scripts/run_multi_expert.sh
```

## 直接调用 `main.py`

下面的示例在 WTH 上运行 `pred_len=24` 的完整 Multi-Expert：

```bash
python -u main.py \
  --method multi_expert \
  --root_path ./data/ \
  --data WTH \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 24 \
  --batch_size 32 \
  --test_bsz 1 \
  --train_epochs 15 \
  --patience 3 \
  --learning_rate_expert 1e-3 \
  --learning_rate_router 1e-3 \
  --online_lr_expert 5e-5 \
  --online_lr_router 1e-5 \
  --online_learning full \
  --num_experts 4 \
  --top_k 4 \
  --tsb_alpha 0.5 \
  --tsb_buffer_size 8 \
  --delay_fb \
  --pretrain_mode retrain \
  --itr 1
```

运行 baseline 时替换 `--method`，并按需要设置对应参数。例如：

```bash
python -u main.py \
  --method patchtst_dgrad \
  --root_path ./data/ \
  --data WTH \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 24 \
  --batch_size 32 \
  --test_bsz 1 \
  --train_epochs 15 \
  --online_learning full \
  --delay_fb \
  --patch_len 16 \
  --stride 8 \
  --d_model 32 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 128 \
  --revin 1 \
  --dgrad_online_lr 1e-3 \
  --itr 1
```

`--method` 应显式指定为上表中的实现之一。

## 运行 Progressive Credit + Subspace

第三阶段使用独立入口，不覆盖原有脚本：

```bash
bash scripts/run_progressive_credit_subspace.sh
```

默认启用 progressive feedback、`horizon_channel` Router、在线 correction、capability sketch、Stable/Recovery memory，并显式选择 `subspace` Expert 更新策略。支持 `DATASETS`、`LENS`、`GPU_IDS`、`MAX_PER_GPU`、`PRETRAIN_MODE`、`ROUTER_GRANULARITY`、`CORRECTION_LR`、`LOCAL_CREDIT_WEIGHT`、`STABLE_BUFFER_SIZE`、`RECOVERY_BUFFER_SIZE`、`SUBSPACE_RANK`、`SUBSPACE_LAMBDA` 和 `EXPERT_UPDATE_STRATEGY` 等环境变量。例如：

```bash
DATASETS="WTH ECL" LENS="24 48" GPU_IDS=0,1 MAX_PER_GPU=1 \
EXPERT_UPDATE_STRATEGY=hybrid bash scripts/run_progressive_credit_subspace.sh
```

消融实验通过参数组合完成，无需复制实现：

| 消融 | 关键参数（其余沿用新脚本默认值） |
|---|---|
| Neural Router only | `--disable_expert_online_update --disable_online_correction --stable_buffer_size 0 --recovery_buffer_size 0 --expert_update_strategy plain` |
| Neural Router + progressive z | 上述配置移除 `--disable_online_correction` |
| Progressive correction，无 version awareness | `--disable_version_awareness` |
| Single Stable，无 Recovery | `--disable_recovery` |
| Stable + Recovery，无 subspace | `--expert_update_strategy plain` |
| 普通 per-Expert subspace | `--disable_credit_weighted_subspace --expert_update_strategy subspace` |
| Responsibility-conditioned subspace | `--disable_version_awareness --expert_update_strategy subspace` |
| Version-aware responsibility-conditioned subspace | `--expert_update_strategy subspace` |
| TSB | `--expert_update_strategy tsb` |
| Subspace | `--expert_update_strategy subspace` |
| Hybrid | `--expert_update_strategy hybrid` |

## 预训练与 checkpoint

默认 `--pretrain_mode retrain` 会先训练模型，再执行在线测试。加载已有权重时使用：

```bash
python -u main.py \
  ... \
  --pretrain_mode load \
  --pretrained_checkpoint /path/to/checkpoint.pth
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--skip_test` | 只预训练并保存 checkpoint |
| `--checkpoint_tag` | 为结构或实验变体添加标识 |
| `--itr` | 重复实验次数 |
| `--online_learning` | `none`、`full` 或 `regressor` |
| `--n_inner` | 每个在线样本的内部更新次数 |

## Multi-Expert 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--num_experts` | 4 | 专家数量 |
| `--top_k` | 4 | 每个通道保留的专家数 |
| `--lambda_div` | 0.0 | 离线专家多样性损失权重 |
| `--online_lr_expert` | 1e-4 | 专家在线基础学习率 |
| `--online_lr_router` | 1e-5 | 路由器在线基础学习率 |
| `--tsb_alpha` | 0.5 | 当前梯度与参考梯度的平滑系数 |
| `--tsb_buffer_size` | 8 | 已释放反馈样本缓冲区大小 |
| `--expert_grad_clip` | 1.0 | 专家梯度裁剪阈值 |
| `--router_grad_clip` | 0.5 | 路由器梯度裁剪阈值 |
| `--router_temperature` | 2.0 | 路由 softmax 温度 |
| `--router_entropy_weight` | 0.001 | 路由熵正则权重 |
| `--progressive_fb` | false | 启用逐时间点成熟反馈 |
| `--router_granularity` | channel | `channel` 或 factorized `horizon_channel` |
| `--correction_lr` | 0.1 | 快速 Router dual correction 学习率 |
| `--correction_decay` | 0.01 | 每个 origin 执行一次的 correction 衰减 |
| `--local_credit_temperature` | 1.0 | `(h,c)` 局部责任温度 |
| `--sample_credit_temperature` | 1.0 | record 级 Expert 责任温度 |
| `--local_credit_weight` | 0.1 | partial Router local-credit loss 权重 |
| `--min_credit_eps` | 1e-8 | credit、熵和对数计算下界 |
| `--capability_sketch_dim` | 32 | 每个 Expert 的能力 sketch 维数 |
| `--capability_sketch_seed` | 2025 | 固定随机投影 seed |
| `--responsibility_threshold` | 0.3 | Expert memory 最低 sample responsibility |
| `--alignment_threshold` | 0.8 | Stable/Recovery 准入分界 |
| `--credit_top_k` | 1 | 每个完整 record 最多归属的 Expert 数 |
| `--stable_buffer_size` | 32 | 每个 Expert 的 Stable 容量 |
| `--recovery_buffer_size` | 32 | 每个 Expert 的 Recovery 容量 |
| `--buffer_duplicate_threshold` | 0.98 | Stable sketch 重复判定阈值 |
| `--recovery_failure_penalty` | 0.5 | Recovery 尝试次数惩罚 |
| `--max_recovery_attempts` | 3 | Recovery 淘汰前最大失败次数 |
| `--buffer_storage_dtype` | fp16 | CPU memory tensor 的 `fp16` 或 `fp32` 存储 |
| `--memory_refresh_interval` | 100 | 完整 record 数量上的 memory 重评周期 |
| `--promote_alignment_threshold` | 0.9 | Recovery 升回 Stable 的 alignment 阈值 |
| `--promote_loss_threshold` | 1.0 | Recovery 升回 Stable 的当前损失阈值 |
| `--recovery_batch_size` | 2 | 每个 Expert 每次更新采样的 Recovery 数量 |
| `--recovery_loss_weight` | 0.1 | Recovery replay 总损失权重 |
| `--recovery_sketch_weight` | 1.0 | replay 内 sketch distillation 权重 |
| `--subspace_scope` | regressor | 当前唯一支持的保护范围 |
| `--subspace_rank` | 0 | 固定 rank；0 表示按累计能量选择 |
| `--subspace_max_rank` | 32 | subspace 最大 rank |
| `--subspace_energy_threshold` | 0.95 | 自动 rank 的累计能量阈值 |
| `--subspace_refresh_interval` | 100 | subspace 刷新周期 |
| `--subspace_min_samples` | 4 | 刷新所需最少 Stable 样本数 |
| `--subspace_eps` | 1e-8 | covariance、rank 与投影数值下界 |
| `--subspace_lambda` | 1e4 | 稳定证据保护强度 |
| `--subspace_gamma_min/max` | 0.0 / 1.0 | 软投影 gamma 裁剪范围 |
| `--expert_update_strategy` | tsb | `plain`、`tsb`、`subspace` 或 `hybrid` |
| `--disable_online_correction` | false | progressive 协议保留但禁用 `z` |
| `--disable_version_awareness` | false | alignment 在 credit/memory 中视为 1 |
| `--disable_recovery` | false | 只使用 Stable buffer，不做 replay |
| `--disable_credit_weighted_subspace` | false | subspace 样本等权 |
| `--disable_expert_online_update` | false | 冻结在线 Expert，仅更新 Neural Router |
| `--disable_tsb` | false | 完整关闭 TSB |
| `--online_log_interval` | 500 | 在线诊断日志间隔 |
| `--credit_diagnostic_buffer_size` | 10000 | 内存中保留的 record 级 credit 诊断上限 |

`--disable_tsb` 会关闭参考梯度、梯度平滑、冲突投影、TSB buffer 和与 TSB 绑定的自适应在线步长，而不只是将 `tsb_alpha` 设为 0。

Router 始终先产生所有 Expert 均为正且归一化的 dense prior。Progressive 模式先计算 `softmax(log(prior) + z)`，再对修正后的结果执行 top-k 并重新归一化；因此快速 correction 可以把原本不在 prior top-k 内的 Expert 提升为有效路由。局部责任学习监督 dense prior，mixture prediction 使用稀疏 effective weights。Router 熵奖励在 partial 与完整反馈更新中均为 `loss - router_entropy_weight × entropy`；权重为 0 时不额外构建熵的反向图。

### Stable/Recovery memory 生命周期

完整 record 使用预测时 Expert 输出计算 sample responsibility，并用预测时/当前 capability sketch 的 cosine alignment 做 version-aware credit transport。高责任且高 alignment 的样本进入 Stable；高责任但低 alignment 的样本进入 Recovery。两类 buffer 都是固定容量：Stable 会执行 sketch 重复检测并按 `responsibility × alignment` 替换，Recovery 按带失败次数惩罚的 recovery score 替换。

周期重评时，alignment 下降的 Stable 样本会先从 Stable 删除，再尝试迁往 Recovery；若 Recovery 已满且拒绝该样本，样本直接淘汰，不会错误地留在 Stable。每次完整 Expert 更新会使用同一个排除集合从所有 Recovery buffer 无重复采样，因此相同 sample ID 在一个 online step 最多 replay 一次；历史 prediction-time sketch 强制 stop-gradient，当前 Expert 分支保留梯度。更新后重新计算 loss/alignment，满足阈值则升回 Stable，超过最大 replay attempts 且未恢复则淘汰。所有 replay、memory 和 subspace 额外 forward 都禁用 FSNet 持久状态写入。

### Expert update strategies

`plain` 使用原始梯度；`tsb` 保留原有参考梯度平滑和冲突投影；`subspace` 不计算 TSB reference，只过滤 prediction-head weight；`hybrid` 先执行 TSB，再执行 prediction-head subspace filtering。`--disable_tsb` 保留兼容：与 `tsb` 冲突时降为 `plain`，与 `hybrid` 冲突时降为 `subspace`，启动时会打印 warning。

Subspace 使用 Stable head features 构建非中心化加权二阶矩 `M = Hᵀ diag(w) H / sum(w)`，对称化后调用 `torch.linalg.eigh`。不做均值中心化，因此重复出现的同一 prediction-head activation 方向仍会被保护。实现只构建 `[320,320]` 矩阵，不会构建 `[B*C,B*C]` 矩阵；负权重会显式报错，非有限或证据不足的刷新会保留旧 basis。FSNet-Time 的 channel feature 都作为观测，但同一样本的权重平均分配给所有 channel。其主要额外开销是周期性 read-only feature forward、每个 Expert 一个 320 维特征协方差和至多 `subspace_max_rank` 个 basis 向量；当前不保护 encoder 层。

ECL 在 `seq_len=60`、`pred_len=48`、321 通道、sketch 维数 32 时，每个 memory item 约保存 35,152 个浮点值：FP16 约 68.7 KiB。默认每个 Expert 32 个 Stable 加 32 个 Recovery、4 个 Expert 全部满载的上界约为 17.2 MiB；FP32 约为 34.3 MiB。Python 对象和少量标量开销未计入，`credit_top_k=1` 会限制单个 record 的复制数量。

查看全部参数：

```bash
python main.py --help
```

## 输出

以下目录由运行过程生成，并已通过 `.gitignore` 排除：

```text
checkpoints/<setting>/
├── checkpoint.pth
└── optimizer.pth

log/<timestamp>/
└── *.out

result/resultsN/<setting>/
├── metrics.npy
├── mae.npy
├── mse.npy
├── preds.npy
├── trues.npy
├── online_diagnostics.npz
├── credit_diagnostics.npz
└── online_diagnostics_summary.json
```

`metrics.npy` 依次包含 MAE、MSE、RMSE、MAPE、MSPE 和运行时间；`mae.npy`、`mse.npy` 保存在线累计曲线。

延迟反馈启动时会输出：

```text
[DELAY_FB] rolling origins; feedback delay=24 steps
```

Progressive Multi-Expert 诊断按 `online_log_interval` 聚合，包含 MSE、prior/effective entropy、`z` norm、Expert 权重与独立 MSE、责任与版本漂移、buffer 生命周期、Recovery 成功率、subspace rank/energy/drift、平行/正交梯度、gamma、TSB conflict rate 和 Router Gap。控制台只保留稀疏摘要。区间数组写入 `online_diagnostics.npz`；bounded record 级 JS divergence、ranking reversal、alignment、confidence 和 Expert update count 写入 `credit_diagnostics.npz`；summary 同时报告累计 record 数、实际保留数与估算覆盖数。

## 目录结构

仅列出未被 `.gitignore` 排除的项目文件：

```text
.
├── data/
│   └── data_loader.py
├── exp/
│   ├── exp_basic.py
│   ├── exp_multi_expert.py
│   ├── exp_dyname.py
│   ├── exp_patchtst_dgrad.py
│   └── exp_stream_baselines.py
├── models/
│   ├── ts2vec/
│   ├── dyname.py
│   ├── patchtst_dgrad.py
│   ├── er.py
│   ├── fsnet.py
│   ├── onenet.py
│   ├── dsof.py
│   └── proceed.py
├── scripts/
│   ├── run_multi_expert.sh
│   └── run_progressive_credit_subspace.sh
├── tests/
│   ├── test_progressive_online_order.py
│   ├── test_recovery_learning.py
│   └── test_subspace_protection.py
├── utils/
│   ├── expert_memory.py
│   ├── online_diagnostics.py
│   ├── recovery_learning.py
│   ├── subspace_protection.py
│   ├── metrics.py
│   ├── timefeatures.py
│   └── tools.py
├── main.py
├── requirement.txt
└── README.md
```

## 常见问题

### 进程显示 `Killed`

通常表示主机内存或 GPU 显存不足。先降低单卡并发：

```bash
MAX_PER_GPU=1 bash scripts/run_multi_expert.sh
```

### 延迟反馈为什么比旧的分块方式更慢

当前实现仍评估每一个相邻预测起点，只延迟标签释放，不再跳过 `H-1` 个窗口，因此测试迭代数与完整滚动预测一致。

### 脚本没有使用预期的 Conda 环境

显式设置：

```bash
PYTHON="$CONDA_PREFIX/bin/python" bash scripts/run_multi_expert.sh
```

### CUDA 或 PyTorch 版本不匹配

先根据本机 NVIDIA 驱动安装对应的 PyTorch CUDA 构建，再安装 `requirement.txt` 中的其他依赖。

## 上游项目

本仓库的模型设计和本地适配参考了：

- [OneNet](https://github.com/yfzhang114/OneNet)
- [FSNet](https://github.com/salesforce/fsnet)
- [OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF)
- [DSOF](https://github.com/yyalau/iclr2025_dsof)
- [DynaME](https://github.com/shhong97/DynaME)

使用相关方法进行论文实验时，请同时引用对应的原始论文和代码仓库。
