# QuaSiD Git 与服务器同步交付流程

Git SSH 首次配置、未来工程复用与认证排错见 [AntCode Git SSH 配置](GIT_SSH_SETUP.md)。AI Studio 认证边界、SFTP 协议约束与服务器运行约定见 [AI Studio 开发工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。本文给出本地静态检查、源码同步与 Git 交付的标准命令和报告项。

## 1. 范围与前置条件

- `sync` 上传列于 [SYNC_FILES.txt](SYNC_FILES.txt) 的源码、配置、测试与工程文档（新增源码文件时同步更新白名单）；整个 `aistudio/` 目录仅供本地使用，不上传。数据文件、编辑器配置、checkpoint 与日志均不入白名单。
- `commit` 与 `push` 独立于服务器同步，Git 不是 `sync` 的前置条件。当前无 `.git` 时按源码快照处理；在有 `.git` 的工作区执行 `push` 时，先检查 `git status --short`、差异与既有 remote，覆盖全部待同步改动并推送当前分支既有目标。
- `--delete <旧路径>` 仅在用户明确要求清理、本地文件已移除且当前仓库具备可核验的 `.git` 历史时使用；无 `.git` 的快照会直接拒绝删除请求。

## 2. 本地静态检查

```bash
cd "/Users/rycai/Workspace/TPAMI QuaSiD Code"
conda run -n rycai python aistudio/sync_server.py --check-only --files main.py
```

将 `--files` 参数替换为本轮实际改动的白名单路径（或用 `--all` 检查全量清单）。该模式对 `.py` 解析 AST、对 `.sh` 执行 `bash -n`、对 `.yaml` 执行 `yaml.safe_load`，有 `.git` 时同时执行 `git diff --check`。

## 3. 计划检查与正式同步

先运行 `--dry-run` 核对本地/远端路径映射与目录清单（不读取认证字段、不连接服务器）：

```bash
# 按需同步本轮改动文件
conda run -n rycai python aistudio/sync_server.py --dry-run \
  --files main.py config/unconditional/stocks.yaml

# 仅在获授权全量部署时检查完整清单
conda run -n rycai python aistudio/sync_server.py --dry-run --all
```

用户授权同步并确认 `ssctl` 已连接后，以相同参数执行正式同步：

```bash
conda run -n rycai --no-capture-output python aistudio/sync_server.py \
  --files main.py config/unconditional/stocks.yaml
```

若有获授权的远端旧文件清理，在 `--dry-run` 与正式同步中传入相同的 `--delete <旧路径>`。任一步骤报错即停止本轮，待本地修正并重新通过 `--dry-run` 后再按用户指示重试。

## 4. 交付报告

分别报告以下实际结果（未执行的环节显式注明未执行）：

- 本地改动路径与 `--check-only` 静态检查结果；
- 服务器同步的上传成功数与失败路径、删除成功数与已缺失旧路径；
- Git 提交哈希、分支与 `push` 状态（若执行）；
- 独立获授权的服务器命令及其直接输出（若执行）。
