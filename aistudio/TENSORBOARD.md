# QuaSiD TensorBoard 日志

当前 [Logger](../engine/logger.py) 将 TensorBoard 实时事件写入服务器本地临时目录 `/tmp/quasid-tensorboard/<experiment>-<唯一后缀>/`，可通过 `QUASID_TENSORBOARD_ROOT` 指定本地磁盘目录。启动日志打印准确路径。DDP 由主进程记录，避免网络文件系统写入填满事件队列、阻塞 rank 0 并拖慢其他 rank。

文本日志写入 `experiments/<mode>/<model_name>/<dataset>/<experiment>/log/logs/`，参数与配置保存在 `log/configs/args.txt` 和 `log/configs/config.yaml`。正常关闭后，TensorBoard 事件复制到 `log/logs/` 归档，本地事件保留；异常退出时应从启动日志中的实时目录恢复事件。

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

实时事件位于 `/tmp/quasid-tensorboard/stocks_v5-<唯一后缀>/`，正常结束后归档至 `experiments/uncond/quasid_sde/stocks/stocks_v5/log/logs/`。两卡 DDP 启动方式见 [开发工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。`scripts/experiments/train_eval.sh` 默认 `TENSORBOARD=true`，通过 `TENSORBOARD=false` 关闭。

## 指标与输出

[训练器](../engine/solver.py) 默认在第 1 步、每 50 步及最后一步记录 `train/loss`、`train/lr`、`train/grad_norm`、`train/step_time_s`、`train/elapsed_time_s`、`train/throughput_samples_per_s` 和 `train/samples_seen`，`--log_interval` 可调整间隔。横轴为优化器更新次数，各标量对应记录当步的值。初始化时还记录可训练参数量与总参数量。进度条输出限频为每秒最多一次。

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
/opt/conda/bin/conda run -n base --no-capture-output python -m tensorboard.main \
  --logdir /tmp/quasid-tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

服务器直接使用其 Python 环境中的 TensorBoard，不依赖本地 `aistudio/` 目录。启动前确认 TensorBoard、NumPy 和 protobuf 版本兼容；多个日志根目录可通过 `--logdir_spec` 指定。本地保留的 [兼容读取脚本](tensorboard_server.py) 不属于服务器部署内容。

用户明确要求本地实时查看时，可复用用户维护的 ssctl 端点，建立本机 `127.0.0.1:6006` 到服务器 `127.0.0.1:6006` 的 SSH 本地转发。转发只绑定本机回环地址，不使用反向转发或公网监听。连接认证、失败处理与数据下载遵循 [服务器工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。
