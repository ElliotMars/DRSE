# Online Multi-Expert Time-Series Forecasting

本项目面向概念漂移场景下的在线多变量时间序列预测。核心方法 `multi_expert` 将 FSNet 与 FSNet-Time 专家、通道级 MoE 路由和 Temporal Smoothness Buffer（TSB）结合，在统一的滚动延迟反馈协议下完成离线预训练和在线适应。

仓库还保留了 DynaME、PatchTST-DGrad、ER、FSNet、OneNet、DSOF 和 PROCEED 的本地适配实现，用于在相同数据接口和在线协议下进行比较。这里的 baseline 是嵌入本项目结构后的实现，不是对应上游仓库的完整镜像。

> 本 README 只描述未被 `.gitignore` 排除的源码和实验入口。数据、checkpoint、日志和结果文件均为本地运行产物，不纳入版本控制。

## 核心方法

默认的 Multi-Expert 包含：

- 2 个 FSNet 专家，建模跨变量表示；
- 2 个 FSNet-Time 专家，建模每个变量的时间模式；
- 通道级路由器，为每个变量动态生成专家权重；
- 可选 top-k 路由；
- TSB 在线更新机制，通过历史已释放样本的参考梯度进行平滑和冲突投影；
- 专家与路由器独立的离线、在线学习率和梯度裁剪。

主要实现位于：

```text
exp/exp_multi_expert.py
models/ts2vec/fsnet.py
models/ts2vec/fsnet_.py
```

## 滚动延迟反馈

启用 `--delay_fb` 后，预测起点仍然每次前进一个时间点。设预测长度为 `H=pred_len`，在线时序为：

```text
predict origin 0
predict origin 1
...
predict origin H-1
release/update origin 0
predict origin H
release/update origin 1
predict origin H+1
...
```

这意味着：

- 当前预测窗口的完整未来标签不会立即参与更新；
- origin `t` 的标签只会在 origin `t+H` 到达后释放；
- 延迟模式不会再通过 `index * pred_len` 跳过中间窗口；
- 延迟和非延迟模式评估相同的预测起点，但可用于模型更新的信息不同。

在线测试建议固定 `--test_bsz 1`。

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
| `--disable_tsb` | false | 完整关闭 TSB |
| `--online_log_interval` | 500 | 在线诊断日志间隔 |

`--disable_tsb` 会关闭参考梯度、梯度平滑、冲突投影、TSB buffer 和与 TSB 绑定的自适应在线步长，而不只是将 `tsb_alpha` 设为 0。

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
└── trues.npy
```

`metrics.npy` 依次包含 MAE、MSE、RMSE、MAPE、MSPE 和运行时间；`mae.npy`、`mse.npy` 保存在线累计曲线。

延迟反馈启动时会输出：

```text
[DELAY_FB] rolling origins; feedback delay=24 steps
```

Multi-Expert 在线诊断还包含 routed/uniform/expert MSE、最差通道、路由熵、动态学习率和梯度范数。

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
│   └── run_multi_expert.sh
├── utils/
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
