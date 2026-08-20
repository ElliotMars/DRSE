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

### 结构性 ablation

Expert composition 保持 Expert 数量、Router 输出与 capability projection 形状不变。默认 `mixed` 沿用 legacy allocation，4 个 Expert 时严格为 2 个 FSNet 加 2 个 FSNet-Time：

```bash
# 4 x FSNet
python -u main.py --method multi_expert --num_experts 4 --top_k 4 --expert_composition fsnet

# 4 x FSNet-Time
python -u main.py --method multi_expert --num_experts 4 --top_k 4 --expert_composition fsnet_time

# 2 + 2 heterogeneous，默认行为
python -u main.py --method multi_expert --num_experts 4 --top_k 4 --expert_composition mixed
```

TSB smoothing/conflict filter 的 2×2 组合如下；没有 disable flag 的子步骤保持开启：

```bash
# smoothing off, filter off
python -u main.py --method multi_expert --disable_tsb_smoothing --disable_tsb_conflict_filter

# smoothing on, filter off
python -u main.py --method multi_expert --disable_tsb_conflict_filter

# smoothing off, filter on
python -u main.py --method multi_expert --disable_tsb_smoothing

# smoothing on, filter on，默认行为
python -u main.py --method multi_expert
```

Controller 可独立选择，不依赖 TSB 是否开启：

```bash
python -u main.py --method multi_expert --adaptive_controller fixed
python -u main.py --method multi_expert --adaptive_controller dynamic
```

轻量脚本默认只打印 pretraining、三种自动 oracle、TSB 2×2、Expert composition、controller 和四种 Expert update strategy 的结构性 ablation 命令，不启动实验；显式设置 `EXECUTE=1` 后才执行。运行 load ablation 时通过 `PRETRAINED_CHECKPOINT` 指定 checkpoint：

```bash
bash scripts/run_structural_ablations.sh \
  --method multi_expert --root_path ./data/ --data WTH --pred_len 24

EXECUTE=1 bash scripts/run_structural_ablations.sh \
  --method multi_expert --root_path ./data/ --data WTH --pred_len 24
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

真实数据短程 smoke test 可以复用同一脚本，不改变完整实验默认行为：

```bash
DATASETS=ETTh2 LENS=24 PRETRAIN_MODE=load MAX_ONLINE_STEPS=200 \
STRICT_ONLINE_CHECKS=1 ONLINE_LOG_INTERVAL=50 \
bash scripts/run_progressive_credit_subspace.sh
```

`MAX_ONLINE_STEPS=-1` 表示不限制；严格检查只建议在短程 smoke test 中开启。

### Recommended Full DRSE / Full Method Configuration

CLI global defaults 为了兼容旧实验仍是 `progressive_fb=false`、
`router_granularity=channel` 和 `expert_update_strategy=tsb`；它们不等于
论文 Full Method 配置。所有正式 paper online results 必须显式启用因果反馈协议：
`--progressive_fb` 或 `--delay_fb`。在线学习开启但两者均关闭时，程序会
在创建 experiment 前 fail-fast，因为 rolling-origin 下的完整 future-window
immediate update 是非因果的。推荐直接使用
`scripts/run_progressive_credit_subspace.sh`，该脚本显式启用 progressive
feedback、`horizon_channel` Router 和 `subspace` Expert 更新。正式配置的
关键参数如下：

| 组件 | Full Method 配置 |
|---|---|
| Progressive / Router | `--progressive_fb --router_granularity horizon_channel` |
| Expert | `--expert_update_strategy subspace --expert_composition mixed --num_experts 4 --top_k 4` |
| Online correction | 默认启用；`--correction_lr 0.1 --correction_decay 0.01` |
| Capability / credit | `--capability_sketch_dim 32 --capability_sketch_seed 2025 --responsibility_threshold 0.3 --alignment_threshold 0.8 --credit_top_k 1` |
| Stable / Recovery | `--stable_buffer_size 32 --recovery_buffer_size 32 --memory_refresh_interval 100 --recovery_batch_size 2 --recovery_loss_weight 0.1 --recovery_sketch_weight 1.0` |
| Recovery lifecycle | `--recovery_failure_penalty 0.5 --max_recovery_attempts 3 --promote_alignment_threshold 0.9 --promote_loss_threshold 1.0` |
| Subspace | `--subspace_rank 16 --subspace_max_rank 32 --subspace_energy_threshold 0.95 --subspace_refresh_interval 100 --subspace_min_samples 4 --subspace_lambda 1e4 --subspace_evidence_mass_scale 8.0 --subspace_gamma_min 0 --subspace_gamma_max 1` |

脚本参数可通过前文列出的同名环境变量覆盖。修改 Full Method 配置时应显式记录
覆盖项，不要把 CLI global defaults 当作论文配置。

每个 iteration 的结果目录都会在长 test 开始前保存 `run_config.json`。该文件使用
paper-critical 参数白名单，记录实际 causal protocol、pretrain mode、Expert composition、
adaptive controller、TSB 开关、Stable/Recovery 与 subspace 配置、comparator 设置、seed、
device，以及可用时的 git commit/dirty 状态；不会写入 tensor、optimizer state 或 checkpoint
内容。重复运行只会覆盖该 iteration 目录内自己的 metadata。

消融实验通过参数组合完成，无需复制实现：

| 消融 | 关键参数（其余沿用新脚本默认值） |
|---|---|
| Neural Router only | `--disable_expert_online_update --disable_online_correction --stable_buffer_size 0 --recovery_buffer_size 0 --expert_update_strategy plain` |
| Neural Router + progressive z | 上述配置移除 `--disable_online_correction` |
| Progressive correction，无 version awareness | `--disable_version_awareness` |
| Single Stable，无 Recovery | `--disable_recovery` |
| Plain / No Subspace | `--expert_update_strategy plain` |
| Unweighted Stable Subspace（仅 covariance 等权） | `--disable_credit_weighted_subspace --expert_update_strategy subspace` |
| Full Responsibility-Conditioned Stable Subspace | `--expert_update_strategy subspace` |
| Legacy TSB | `--expert_update_strategy tsb` |
| Hybrid | `--expert_update_strategy hybrid` |

## 预训练与 checkpoint

默认 `--pretrain_mode retrain` 会先训练模型，再执行在线测试。加载已有权重时使用：

```bash
python -u main.py \
  ... \
  --pretrain_mode load \
  --pretrained_checkpoint /path/to/checkpoint.pth
```


完全跳过 offline supervised pretraining 和 checkpoint 加载时使用：

```bash
python -u main.py ... --pretrain_mode none
```

`none` 的严格语义是 random initialization + online adaptation only。首次 `test()` 会在任何在线更新前捕获随机初始化 model 与 optimizer 状态；同一 `Exp` 后续重复测试仍恢复到该相同 initial state。

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
| `--expert_composition` | mixed | `mixed`、`fsnet` 或 `fsnet_time` Expert 组成 |
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
| `--max_recovery_attempts` | 3 | 最多允许的 Recovery 失败次数；第 `max_recovery_attempts` 次失败后淘汰 |
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
| `--subspace_lambda` | 1e4 | 归一化稳定证据的保护强度 |
| `--subspace_evidence_mass_scale` | 8.0 | Stable evidence mass 饱和归一化尺度 |
| `--subspace_gamma_min/max` | 0.0 / 1.0 | 软投影 gamma 裁剪范围 |
| `--expert_update_strategy` | tsb | `plain`、`tsb`、`subspace` 或 `hybrid` |
| `--disable_online_correction` | false | progressive 协议保留但禁用 `z` |
| `--disable_version_awareness` | false | alignment 在 credit/memory 中视为 1 |
| `--disable_recovery` | false | 只使用 Stable buffer，不做 replay |
| `--disable_credit_weighted_subspace` | false | 仅让 Stable-subspace covariance 的样本权重等权；memory admission、生命周期和 evidence-adaptive gamma 保持启用 |
| `--disable_expert_online_update` | false | 冻结在线 Expert，仅更新 Neural Router |
| `--disable_tsb` | false | 完整关闭 TSB smoothing、conflict filter 与 reference |
| `--disable_tsb_smoothing` | false | 单独关闭 TSB gradient smoothing |
| `--disable_tsb_conflict_filter` | false | 单独关闭 TSB conflict projection |
| `--adaptive_controller` | dynamic | `fixed` 或独立于 TSB 的 `dynamic` 在线 LR controller |
| `--online_log_interval` | 500 | 在线诊断日志间隔 |
| `--credit_diagnostic_buffer_size` | 10000 | 内存中保留的 record 级 credit 诊断上限 |
| `--dynamic_comparator` | false | 显式启用 offline empirical K-switch comparator |
| `--dynamic_comparator_max_switches` | 1 | K-switch comparator 允许的最大切换次数 |
| `--dynamic_comparator_max_points` | 64 | K-switch 二次复杂度求解使用的最大有序时间块数 |
| `--max_online_steps` | -1 | smoke test 最大 online origin 数；-1 不限制 |
| `--strict_online_checks` | false | 开启高开销在线有限性、归一化、生命周期和顺序检查 |
| `--seed` | 0 | iteration 基础随机种子；第 `ii` 次使用 `seed + ii` |

`--disable_tsb` 保留 legacy 总开关与 strategy 降级 warning：它关闭 TSB reference、gradient smoothing、conflict projection 和 TSB buffer，但 adaptive controller 由 `--adaptive_controller` 独立选择。`fixed` 始终使用 Expert/Router base LR 与 base `tsb_alpha`；`dynamic` 即使 TSB 关闭也会根据 error ratio 缩小 LR，并保证不超过 base LR。只有 TSB smoothing 启用时，dynamic controller 才会动态增大 `tsb_alpha`。

Router 始终先产生所有 Expert 均为正且归一化的 dense prior。Progressive 模式先计算 `softmax(log(prior) + z)`，再对修正后的结果执行 top-k 并重新归一化；因此快速 correction 可以把原本不在 prior top-k 内的 Expert 提升为有效路由。局部责任学习监督 dense prior，mixture prediction 使用稀疏 effective weights。Router 熵奖励在 partial 与完整反馈更新中均为 `loss - router_entropy_weight × entropy`；权重为 0 时不额外构建熵的反向图。

### Stable/Recovery memory 生命周期

完整 record 使用预测时 Expert 输出计算 sample responsibility，并用预测时/当前 capability sketch 的 cosine alignment 做 version-aware credit transport。高责任且高 alignment 的样本进入 Stable；高责任但低 alignment 的样本进入 Recovery。令 sample confidence 为 `κ`、prediction-time responsibility 为 `r`、当前 capability alignment 为 `a`。长期 Stable credit 为 `q_stable = κ × r × a`，Recovery credit 为 `q_recovery = κ × r × (1-a)`。两类 buffer 都是固定容量：Stable 会执行 sketch 重复检测并直接按 `q_stable` 替换；Recovery priority 为 `q_recovery / (1 + failure_penalty × attempts)`。`responsibility_threshold` 与 `credit_top_k` 的准入行为保持不变，confidence 只调节长期 credit。

周期重评时，alignment 下降的 Stable 样本会先从 Stable 删除，再尝试迁往 Recovery；若 Recovery 已满且拒绝该样本，样本直接淘汰，不会错误地留在 Stable。每次完整 Expert 更新会使用同一个排除集合从所有 Recovery buffer 无重复采样，因此相同 sample ID 在一个 online step 最多 replay 一次；历史 prediction-time sketch 强制 stop-gradient，当前 Expert 分支保留梯度。更新后重新计算 loss/alignment：尚未恢复的样本留在 Recovery；恢复成功且 Stable 接收时迁入 Stable；恢复成功但 Stable 拒绝时直接淘汰，不会再次 replay；第 `max_recovery_attempts` 次 replay 失败后即淘汰。周期 refresh 使用完全相同的 Stable-or-Discard 语义。所有 replay、memory 和 subspace 额外 forward 都禁用 FSNet 持久状态写入。

当前 capability sketch 是 pooled representation snapshot 的固定随机投影：
ExpertNet 使用 temporal-mean representation，FSNet-Time 使用 channel-mean
representation。record-level `capability_alignment_existing` 明确表示这一
现有 coarse sketch 的 alignment；`capability_alignment` 作为向后兼容别名
保留。它不是完整 prediction-head capability，也不应解释为 exact head-feature
alignment。当前 record 没有保存 prediction-time head feature；后续若增加
`head_feature_alignment`，应作为 evaluation-only 诊断单独设计存储开销，
不能改变 admission、Recovery loss 或 subspace。

### 重复在线测试状态恢复

同一个 `Exp` 第一次调用 `test()` 时，会在任何在线更新前捕获当前 test-start baseline（预训练 checkpoint 或 `pretrain_mode=none` 的随机初始化）、FSNet 注册状态、Expert/Router optimizer state 和参数 `requires_grad`。后续调用 `test()` 会先恢复该 CPU 快照，再清除 gradient、Router correction、TSB buffer、Stable/Recovery memory、subspace、diagnostics 和临时计数。`grads`、`f_grads`、`q_ema`、`trigger` 已注册为 buffer，`W` 属于 model parameter，因此均包含在 `state_dict()` 中。当前不恢复 RNG state；Recovery 采样是确定性优先级排序，重复流测试使用固定输入和 seed。

### Expert update strategies

`plain`（Plain / No Subspace）使用原始梯度；`tsb`（Legacy TSB）保留参考梯度平滑和冲突投影；`subspace` 使用 Full Responsibility-Conditioned Stable Subspace，只过滤 prediction-head weight；`hybrid` 先执行启用的 TSB 子步骤，再执行 prediction-head subspace filtering。`--disable_credit_weighted_subspace` 仅得到 Unweighted Stable Subspace，即 covariance geometry 对 Stable samples 等权，仍保留 responsibility-controlled admission、version-aware lifecycle 和 evidence-adaptive gamma；当前没有实现或声称 no-responsibility GPM baseline。TSB 两个子步骤都关闭时不计算 reference，结果等于 raw current gradient。summary 中的 `tsb_gradient_diagnostics` 以 O(1) running sum/count 保存 `mean_grad_cosine`、`conflict_rate`、`mean_tsb_modification_ratio` 和可选 projection removal ratio，不保存逐参数梯度 tensor。

Subspace 使用 Stable head features 构建非中心化加权二阶矩 `M = Hᵀ diag(w) H / sum(w)`，对称化后调用 `torch.linalg.eigh`。不做均值中心化，因此重复出现的同一 prediction-head activation 方向仍会被保护。实现只构建 `[320,320]` 矩阵，不会构建 `[B*C,B*C]` 矩阵；负权重会显式报错，非有限或非空但证据不足的刷新会保留旧 basis。Subspace 的样本权重直接使用 memory 中的 `item.stable_credit`；FSNet-Time 的 channel feature 都作为观测，但同一样本的 stable credit 平均分配给所有 channel，因此 channel 权重和仍等于该样本的 stable credit。Stable evidence mass 是当前 Stable 样本 `stable_credit` 的累积和 `m = Σq_stable`，保护质量使用 `m / (m + subspace_evidence_mass_scale)` 饱和归一化；随后 `lambda_eff = subspace_lambda × protection_mass`，`gamma = 1 / (1 + online_lr × lambda_eff)`。Stable buffer 变空时保留 cached basis geometry，但立即把 evidence mass 和 protection mass 置 0、gamma 置 1，因此 filtered gradient 与 raw gradient 相同。其主要额外开销是周期性 read-only feature forward、每个 Expert 一个 320 维特征协方差和至多 `subspace_max_rank` 个 basis 向量；当前只保护 `ExpertNet.regressor` 与 `FSNetTimeExpertNet.regressor_time`，不保护 encoder 层。

ECL 在 `seq_len=60`、`pred_len=48`、321 通道、sketch 维数 32 时，每个 memory item 约保存 35,152 个浮点值：FP16 约 68.7 KiB。默认每个 Expert 32 个 Stable 加 32 个 Recovery、4 个 Expert 全部满载的上界约为 17.2 MiB；FP32 约为 34.3 MiB。Python 对象和少量标量开销未计入，`credit_top_k=1` 会限制单个 record 的复制数量。

查看全部参数：

```bash
python main.py --help
```

## Synthetic concept-drift benchmark

Synthetic benchmark 与 ECL/ETT/WTH loader 完全分离：generator 位于 `data_provider/synthetic_drift.py`，不会改变 `data/data_loader.py` 的默认行为。它使用固定 seed 的 causal AR、正弦项、trend 和 channel mixing 生成多变量序列，并为每个 timestamp 输出 `regime_id`、transition alpha、已知 drift events 与 regime intervals。支持：

- `abrupt`：A 在明确 change point 后立即切换为 B；
- `gradual`：transition window 内以单调 alpha 从 A 混合到 B；
- `recurring`：明确记录 `first_A`、`B`、`recurring_A` 的 A→B→A；
- `shock`：A 中临时注入 level/noise shock，窗口结束后恢复 A；
- `--synthetic_drift_channels 0,2`：只在指定 channel 上施加 drift，其余 channel 继续使用 A。

runner 复用生产代码的 `ProgressiveFeedbackManager`、`OnlineRoutingCorrection` 和 `ExpertMemoryManager`。每个 origin 严格执行 release matured observation → progressive update/memory lifecycle → predict → evaluation-only target access；新建 record 的 future target 始终为空。轻量 Expert 只用于快速机制 benchmark，不加入或替换真实模型。

脚本默认 dry-run，列出 `plain`、`tsb`、`subspace`、`hybrid` 四种策略：

```bash
bash scripts/run_synthetic_drift.sh

EXECUTE=1 \
DRIFT_TYPES="abrupt gradual recurring shock" \
STRATEGIES="plain tsb subspace hybrid" \
SYNTHETIC_DRIFT_CHANNELS="0,2" \
bash scripts/run_synthetic_drift.sh
```

可用 `DISABLE_RECOVERY=1`、`DISABLE_VERSION_AWARENESS=1`、`DISABLE_Z_CORRECTION=1` 做机制关闭对照。单次直接运行示例：

```bash
python -m utils.synthetic_drift_benchmark \
  --drift_type recurring \
  --strategy hybrid \
  --seq_len 12 --pred_len 3 --channels 3 --total_length 120 \
  --synthetic_drift_channels 0,2 \
  --output_dir result/synthetic_drift
```

每次运行保存 `predictions_and_mse.npz`、`drift_events.json`、`drift_metrics.json`、`memory_diagnostics.json` 和 `synthetic_timeline.npz`。Drift metrics 包含 pre-drift baseline、early post-drift error、peak、recovery time、cumulative excess error，以及 recurring A 的 reacquisition time 和 old-mode retention error ratio。Recovery time 是 drift 后第一个连续 `recovery_window` 的平均 MSE 不超过 `baseline × (1 + tolerance)` 的偏移；未恢复时写 JSON `null`。Memory 输出包含每个 Expert 的 Stable/Recovery occupancy timeline、按生成 regime 分解的 evidence、Stable→Recovery、Recovery→Stable、dropped-after-success、attempts-exhausted 和 mean recovery attempts。核心模块只输出数据，不生成论文图片。

## 输出

以下目录由运行过程生成，并已通过 `.gitignore` 排除：

```text
checkpoints/<setting>/
├── checkpoint.pth
└── optimizer.pth

log/<timestamp>/
└── *.out

result/resultsN/<setting>/
├── itr_0/
│   ├── metrics.npy
│   ├── mae.npy
│   ├── mse.npy
│   ├── preds.npy
│   ├── trues.npy
│   ├── online_diagnostics.npz
│   ├── credit_diagnostics.npz
│   ├── specialization_diagnostics.npz
│   ├── comparator_diagnostics.npz
│   ├── comparator_diagnostics.json
│   └── online_diagnostics_summary.json
├── itr_1/
│   └── ...
├── metrics.npy
├── mae.npy
├── mse.npy
├── preds.npy
├── trues.npy
├── aggregate_metrics.npz
└── aggregate_diagnostics_summary.json
```

每个 iteration 都写入稳定编号的 `itr_N` 目录，`itr=1` 也使用
`itr_0`。目录内沿用原来的预测文件名；顶层 `metrics.npy` 等文件继续
保存所有 iteration，供旧的结果读取代码使用。`aggregate_metrics.npz`
保存指标 mean/std；`aggregate_diagnostics_summary.json` 只读取各
iteration 的 summary JSON 汇总，不加载逐步 NPZ。随机种子为
`iteration_seed = seed + ii`，并写入对应 summary。

延迟反馈启动时会输出：

```text
[DELAY_FB] rolling origins; feedback delay=24 steps
```

Progressive Multi-Expert 诊断按 `online_log_interval` 聚合，包含 MSE、prior/effective entropy、`z` norm、Expert 权重与独立 MSE、责任与版本漂移、buffer 生命周期、Recovery 成功率、subspace rank/energy/drift、平行/正交梯度、gamma、TSB conflict rate 和 Router Gap。控制台只保留稀疏摘要。区间数组写入 `online_diagnostics.npz`。

Bounded `credit_diagnostics.npz` 为每个完整 record 保存 origin、prediction/current responsibility `[E]`、prediction/current Expert MSE `[E]`、pooled-sketch capability alignment/L2 distance `[E]`、JS divergence、ranking reversal、sample confidence、`expert_update_delta`、prediction-time mixture MSE 和 Router Gap。它还使用 prediction-time 保存的 Expert/mixture outputs 与完整 matured target 计算 evaluation-only Hard Expert、Top-2 convex 和 All-Expert convex hindsight oracle；新增字段为 `oracle_all_expert_mse`、`oracle_all_expert_weights [E]` 与 `gap_to_all_expert_oracle`。三种 oracle 都不重新运行 prediction-time model，不参与 Router、Expert、`z`、memory 或 subspace 更新，也不是 across-time constrained regret comparator。

`specialization_diagnostics.npz` 使用 streaming sum/count，状态空间为 `O((H+C)E)`，不会保存逐 record 的 `H×C×E` tensor。字段为 `mean_router_weight_by_horizon [H,E]`、`mean_router_weight_by_channel [C,E]`、`expert_mse_by_horizon [H,E]`、`expert_mse_by_channel [C,E]`，以及 winning Expert rate 的同形状数组。evaluation target 只在 `_update_prediction_diagnostics()` 中更新这些统计，不参与任何训练操作。

`comparator_diagnostics.npz/json` 使用 prediction-time Expert prediction 和测试完成后的 target 做 offline hindsight evaluation；target 不参与实际 Router prediction 或更新。Per-sample Hard/Top-2/All-Expert oracle 与下列 across-time constrained comparator 使用严格不同的名称和字段，oracle gap 不是 regret。

Global Static Convex Comparator 在整个 time、horizon 和 channel 上共享一个固定 `w∈Δ^E`，最小化 `sum_t f_t(w)`。测试期间 streaming 累计 `E×E` Gram、`E` cross term、target norm 和 Router loss，空间为 `O(E²)`；测试结束后用带 tolerance 和 iteration cap 的 simplex projected gradient 求解。保留兼容字段 `static_comparator_loss`、`static_regret`、`average_static_regret`、`static_comparator_weights [E]`，并同时输出显式的 `global_static_*` 字段。

Horizon-Channel Static Comparator 为每个固定 `(h,c)` 独立求一个跨时间不变的 `u_{h,c}∈Δ^E`。它批量求解 `H×C` 个小型 `E` 维 simplex quadratic，不构造跨 `(h,c)` 的巨大 Hessian；streaming 状态空间为 `O(HCE²)`。输出 `hc_static_comparator_loss`、`hc_static_regret`、`hc_average_static_regret` 和 `hc_static_comparator_weights [H,C,E]`。

K-switch Dynamic Comparator 是默认关闭的 empirical hindsight comparator。显式传入 `--dynamic_comparator` 后，它把时间保留为至多 `M=dynamic_comparator_max_points` 个有序块，预计算所有 `O(M²)` segment 的 best fixed global mixture，再用 dynamic programming 找到至多 `K+1` 个 segment；因此二次问题规模受 `M` 硬限制，且不允许逐 sample 任意切换 mixture。输出 `dynamic_comparator_loss`、`dynamic_regret`、`average_dynamic_regret`、`switch_points` 和 `segment_weights`。K=0 与 Global Static Comparator 定义一致。

以上结果分别是 empirical static regret 与 empirical K-switch dynamic regret，不构成或声称 Router 的 theoretical no-regret proof。`evaluate_path_length_comparator()` 仍是明确的 TODO，并抛出 `NotImplementedError`，不会用近似值冒充 exact path-length comparator。

Delayed Credit-Capability 分析无需重跑模型：

```bash
python -m utils.credit_diagnostic_analysis \
  result/resultsN/SETTING/itr_0/credit_diagnostics.npz \
  result/resultsN/SETTING/itr_0/credit_analysis.npz
```

输出按 `expert_update_delta` 和 record mean alignment 分桶，包含 mean JS、reversal rate、mean alignment 和 mean Router Gap。基础绘图示例：

```python
import matplotlib.pyplot as plt
import numpy as np

with np.load("credit_analysis.npz") as data:
    edges = data["alignment_bin_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, data["alignment_mean_js"], marker="o")
    plt.xlabel("Mean capability alignment")
    plt.ylabel("Mean JS divergence")
    plt.show()
```

跨 iteration summary 对存在的字段计算 mean/std，包括 online MSE、
Router Gap、JS divergence、ranking reversal、capability alignment、
Stable/Recovery 平均大小、Recovery 成功率与生命周期计数、subspace
rank/energy、`z` norm 以及适用时的 TSB conflict rate。策略不产生的
字段保留为 `null`，不会用 0 伪造。

### Progressive 真实数据 smoke test

独立 smoke 脚本不会改变正式实验脚本的默认参数：

```bash
DATASET=ETTh2 \
PRED_LEN=24 \
MAX_ONLINE_STEPS=200 \
STRICT_ONLINE_CHECKS=1 \
bash scripts/smoke_test_progressive.sh
```

默认使用小 Stable/Recovery buffer、较低 subspace rank、`subspace`
更新、`horizon_channel` Router、top-k 4、200 个 origin 和 `itr=1`。
所有配置均可由同名环境变量覆盖。`PRETRAIN_MODE=load` 时必须存在
兼容 checkpoint，可用 `CHECKPOINT=/path/to/checkpoint.pth` 显式指定；
旧的 channel Router checkpoint 应同时设置
`ROUTER_GRANULARITY=channel`。`PRETRAIN_MODE=retrain` 会明确打印无
checkpoint 模式。

脚本以前台方式运行并写独立日志，完成后自动调用
`scripts/check_smoke_results.py`。检查器支持输入顶层结果目录或单个
`itr_N` 目录，验证必需文件、有限数值、熵、`z`、alignment/JS/ranking
范围、buffer 容量、subspace rank/energy、成熟 record 理论上界和
strict failure count；成功时输出 `Smoke test diagnostics: PASS`。

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
