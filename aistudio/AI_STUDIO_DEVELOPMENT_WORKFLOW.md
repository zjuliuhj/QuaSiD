# QuaSiD AI Studio 开发工作流

本文约定当前 QuaSiD 工程的认证、源码同步和服务器执行方式。模型、数据与命令行接口见 [项目 README](../README.md)，交付步骤见 [Git 与同步流程](GIT_PUSH_AND_SERVER_SYNC_SOP.md)。

## 工程与环境

| 项目 | 约定 |
| --- | --- |
| 本地源码 | `/Users/rycai/Workspace/TPAMI QuaSiD Code`，同步脚本从自身位置解析工程根目录 |
| 服务器工程 | `/ossfs/workspace/QuaSiD` |
| 运行目录 | `/ossfs/workspace/QuaSiD`，与本地工程保持相同相对路径 |
| 已有 ssctl 端点 | `cairongyao.cry@127.0.0.1:58598` |
| 本地检查与同步环境 | `conda run -n rycai python` |
| 服务器 Python 约定 | `/opt/conda/bin/conda run -n base --no-capture-output python`，运行前确认环境与依赖 |
| PPU 环境 | 在使用 PPU 的远端 shell 中加载 `/usr/local/PPU_SDK/envsetup.sh` |
| 模型 | `quasid_sde`、`quasid_ddpm`，骨干为 `dit_solver_v5` |
| 任务 | `uncond`、`cond`、`forecast` |
| 运行产物 | `experiments/<mode>/<model_name>/<dataset>/<experiment>/` |

源码只在本地修改，服务器用于获授权的运行、测试和评估。本地与服务器根目录直接包含 `main.py`、`models/`、`config/`、`data/` 和 `aistudio/`。同步仅传输白名单源码，不迁移数据、checkpoint、日志或实验产物。单独获授权的数据同步写入 `data/datasets/`，运行前检查所选配置要求的数据是否已在服务器准备好。

当前代码将 checkpoint 写入运行目录下的 `ckpt/`，文本日志和 TensorBoard 事件写入 `log/logs/`，参数与配置写入 `log/configs/`，评估图写入 `eval/` 或 `eval/<eval_tag>/`，评估指标写入 CSV。所有路径相对服务器工程根目录。当前实现直接写入工程目录，没有独立 checkpoint 根目录或临时 TensorBoard 归档机制。

## 唯一通道与认证

ssctl 由用户维护。用户确认“ssctl 已连接”后，该授权在当前会话内持续有效；Codex 作为客户端完成明确授权的任务，不自行启动、停止、重启或替换隧道。

SSH/SFTP 只连接 `127.0.0.1:58598`，用户固定为 `cairongyao.cry`。不得使用 root、直连、其他跳板、代理、ProxyJump、ProxyCommand、ControlMaster、control socket 或其他持久连接，也不得让客户端配置隐式切换路线。禁止反向转发和公网暴露。仅在用户明确要求实时查看 TensorBoard 时，允许 [TensorBoard 文档](TENSORBOARD.md) 中的本机回环地址转发。

认证包装器仅在内存中从本地忽略文件 `aistudio/connect_ai_studio.sh` 读取所需认证字段。不得使用 cat、sed、rg 或调试打印输出该脚本，不得将密码写入新文件、命令、日志、工具输出、Git 或服务器。

密码只在 SSH/SFTP 客户端关闭回显的密码提示中发送，非交互包装器不得调用 stty。认证结束后恢复正常任务输出，以命令退出、EOF 或 SFTP 文件操作结果判断完成，不等待非交互 SSH 的 shell prompt。

连接阶段 30 秒内没有认证成功或远端响应时，停止并报告直接错误或“30 秒无响应”。任何连接失败后，未经用户重新确认或要求重试，不得重试、探测其他端口、接管 GUI 或更换认证与连接方式。本地权限拒绝不代表隧道断开；没有取得远端输出时，报告远端状态未确认。

## 本地编辑与验证

1. 检查当前工程是否有 `.git`。有 Git 时盘点状态、已跟踪差异和未跟踪文件，保留用户已有改动；源码快照使用文件清单检查。
2. 使用 apply_patch 修改源码、配置和文档。
3. 默认只对本轮改动文件执行静态检查：Python 使用 AST 解析、Shell 使用 `bash -n`、YAML 使用 `yaml.safe_load`，不导入模型或收集测试。有 `.git` 时执行 `git diff --check`。可运行 `conda run -n rycai python aistudio/sync_server.py --check-only --files <改动路径...>`。
4. 用户授权同步后，按 [Git 与同步流程](GIT_PUSH_AND_SERVER_SYNC_SOP.md) 上传明确选择的白名单文件。仅在用户明确要求行为测试时运行最相关的现有用例，使用 `--maxfail=1 --durations=3`。

本地不自动运行 pytest、模型前向、CPU 冒烟或训练。全量测试和完整训练/评估须由用户明确要求；日常修改不新增重复覆盖的测试。服务器失败时回到本地修复，再按新一轮同步流程交付。缺少依赖或环境不符时报告直接错误，不静默安装或切换环境。纯文档修改使用本地检查验收。

## 源码同步契约

同步固定使用 [sync_server.py](sync_server.py)（本地 `rycai` 环境 + 系统 `/usr/bin/sftp`），所有上传路径须列于 [SYNC_FILES.txt](SYNC_FILES.txt)；具体命令与交付报告格式见 [Git 与同步流程](GIT_PUSH_AND_SERVER_SYNC_SOP.md)。

正式同步只建立一条 SFTP 会话：先进入 `/ossfs/workspace/QuaSiD` 并优先探测最深层父目录（仅在返回不存在时逐级 `mkdir`），用 `pwd` 核验规范路径仍在工程内；文件上传至同目录 `.upload-<PID>` 临时路径后再以 `rename` 原子替换；全部成功后 `bye` 退出，不为同步额外建立 SSH 或触发远端校验。任何非预期目录、删除、上传或响应错误立即中止本轮并输出脱敏诊断，不建立第二个传输会话。

## 服务器运行

启动前检查实际设备占用与环境。两卡 DDP 使用本次选定的两张空闲设备和 `--nproc_per_node=2`，不默认复用上一轮设备。`dataloader.batch_size` 是全局 batch，两卡各分一半；有效优化器 batch 还乘以 `solver.grad_accum_every`。保留所选 YAML 的实验参数，变更 batch 时显式传入覆盖值。

非同步的有界任务通过同一端点使用一条合并的非交互 SSH。每个服务器写入命令都先进入工程根目录，结束后等待退出和 EOF。后台训练启动和后续状态检查属于分别授权的有界任务。

使用 PPU 时先在同一远端 shell 加载 SDK。下面的训练示例要求已将 `QUASID_DEVICES` 设置为本次检查确认的两张空闲设备编号，例如 `2,3`；该示例使用已包含在本地发布包中的 stocks 数据，其服务器副本仍需单独准备：

```bash
cd /ossfs/workspace/QuaSiD
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES="${QUASID_DEVICES:?请先设置两张空闲设备编号}" \
  /opt/conda/bin/conda run -n base --no-capture-output python -m torch.distributed.run \
  --standalone --nproc_per_node=2 main.py --train --mode uncond \
  --model_name quasid_sde --config_file config/unconditional/stocks.yaml \
  --name stocks_v5 --tensorboard
```

两卡训练不传 `--gpu`。当前 `main.py` 仅支持单进程 checkpoint 评估：用普通 Python 启动，省略 `--train`，通过 `--milestone` 选择 checkpoint，并保留训练时的模型与配置覆盖。`scripts/experiments/train_eval.sh` 和 GPU 队列按单卡任务运行，不能通过设置 `NPROC_PER_NODE` 将它们切换为 DDP；队列的空闲设备探测依赖 `nvidia-smi`。

`--model_name` 控制输出路径，`model.target` 选择模型实现。改为 DDPM 时同时设置 `--model_name quasid_ddpm` 和尾部覆盖 `model.target models.quasid.quasid_ddpm.QuaSiD_ddpm`。训练精度由 `solver.mixed_precision` 控制。算子、flash SDPA 和 PPU 的兼容性以实际服务器检查为准；仅识别到设备不代表环境完整。

## Git 与报告

当前目录没有 `.git` 时按源码快照处理，不自动初始化仓库或选择 remote。存在 `.git` 且用户要求 push 时，盘点全部本地修改和非忽略的未跟踪文件，逐路径暂存、创建必要提交并推送当前分支的既有目标。保留既有 remote 和认证方式，不硬编码其他项目的分支，不生成 SSH key 或读取 Keychain 来解决推送问题。

Git 失败不阻止已授权的源码同步。报告分别说明本地修改、静态检查、服务器同步、服务器测试和 Git 操作的实际结果。同步报告列出上传数量与失败路径、删除数量与缺失路径，以及未执行的操作。
