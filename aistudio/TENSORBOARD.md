# QuaSiD TensorBoard 日志

当前 [Logger](../engine/logger.py) 直接将文本日志和 TensorBoard 事件写入 `experiments/<mode>/<model_name>/<dataset>/<experiment>/log/logs/`。参数和配置保存在同一运行目录的 `log/configs/args.txt` 与 `log/configs/config.yaml`。DDP 由主进程记录日志。

## 开启记录

使用 `--tensorboard` 开启事件记录。以下是服务器单进程示例，要求已设置一张空闲设备编号 `QUASID_DEVICE`、准备 stocks 数据并确认环境：

```bash
cd /ossfs/workspace/QuaSiD
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES="${QUASID_DEVICE:?请先设置一张空闲设备编号}" \
  /opt/conda/bin/conda run -n base --no-capture-output python main.py \
  --train --mode uncond --model_name quasid_sde \
  --config_file config/unconditional/stocks.yaml --name stocks_v5 --tensorboard
```

事件目录为 `experiments/uncond/quasid_sde/stocks/stocks_v5/log/logs/`。两卡 DDP 启动方式见 [开发工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。`scripts/experiments/train_eval.sh` 默认 `TENSORBOARD=true`，通过 `TENSORBOARD=false` 关闭。

## 指标与输出

[训练器](../engine/solver.py) 在每个优化器步骤记录 `train/loss`、`train/lr`、`train/grad_norm`、`train/step_time_s`、`train/elapsed_time_s`、`train/throughput_samples_per_s` 和 `train/samples_seen`，横轴为优化器更新次数。初始化时还记录可训练参数量与总参数量。

checkpoint 保存到运行目录的 `ckpt/`。评估结果写入 CSV，图位于 `eval/` 或指定的 `eval/<eval_tag>/`。多 seed 评估及其 CSV 汇总方式见 [项目 README](../README.md)。

## 查看已有本地事件

从本地工程根目录读取已经位于本地的事件文件：

```bash
cd "/Users/rycai/Workspace/TPAMI QuaSiD Code"
conda run -n rycai python -m tensorboard.main \
  --logdir experiments --host 127.0.0.1 --port 6006
```

启动后访问 [本地 TensorBoard](http://127.0.0.1:6006)。该命令只读取本地文件，不建立服务器通路。

## 服务器读取服务

服务器从 QuaSiD 工程根目录启动，仅监听回环地址：

```bash
cd /ossfs/workspace/QuaSiD
/opt/conda/bin/conda run -n base --no-capture-output python aistudio/tensorboard_server.py \
  --logdir experiments --host 127.0.0.1 --port 6006 --reload_interval 5
```

[读取服务脚本](tensorboard_server.py) 为 TensorBoard 2.14 的旧 NumPy 别名和新版 protobuf 参数提供进程内兼容处理。它只影响读取服务，不改变训练日志的写入路径。多个日志根目录可通过 `--logdir_spec` 指定。

用户明确要求本地实时查看时，可复用用户维护的 ssctl 端点，建立本机 `127.0.0.1:6006` 到服务器 `127.0.0.1:6006` 的 SSH 本地转发。转发只绑定本机回环地址，不使用反向转发或公网监听。连接认证、失败处理与数据下载遵循 [服务器工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。
