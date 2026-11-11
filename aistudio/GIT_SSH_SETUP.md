# AntCode Git SSH 配置与工程复用

本说明记录已验证的 AntCode SSH 配置，供当前工程维护和未来工程复用。同一台机器、同一 AntCode 账号下，新工程可以复用已登记的密钥，只需配置各自的仓库地址。换机器时为新机器配置独立密钥。

Git 访问代码平台使用 `code.alipay.com`；AI Studio 服务器连接使用单独的 ssctl 通道，遵循 [AI Studio 开发工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。两者分别配置和验证。

## 当前工程的有效配置

| 项目 | 配置 |
| --- | --- |
| Git SSH 主机与用户 | `git@code.alipay.com` |
| 远程仓库 | `git@code.alipay.com:cairongyao.cry/QuaSiD.git` |
| 本地分支与上游 | `master` → `origin/master` |
| 本机私钥 | `~/.ssh/id_ed25519_antcode` |
| 本机公钥 | `~/.ssh/id_ed25519_antcode.pub` |
| 平台公钥标题 | `rycai-mac-antcode` |
| SSH 配置文件 | `~/.ssh/config` |

2026 年 11 月 11 日，平台登记公钥后，实际通过上述 SSH 地址运行 `git push -u origin master`，返回 `Everything up-to-date`，退出码为 0，全程没有密码或钥匙串授权提示。

私钥只保留在本机，权限为 `600`。仓库文档只记录路径和配置方法，不保存私钥、身份令牌或完整机器配置。

## 首次配置一台机器

### 检查已有配置

```bash
git remote -v
git branch --show-current
ssh -G git@code.alipay.com | rg '^(hostname|user|identityfile|identityagent|identitiesonly) '
ssh-add -l
ls -la ~/.ssh
```

已有可用密钥时，保留并核对其公钥登记情况，不覆盖或重复生成。`ssh-add -l` 只列出 agent 中的身份；通过 `IdentityFile` 直接使用磁盘上的私钥时，agent 为空仍可正常登录。

### 按需生成密钥

以下命令仅在专用密钥和公钥文件均不存在时生成 Ed25519 密钥，避免覆盖已有文件：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
test ! -e ~/.ssh/id_ed25519_antcode && \
  test ! -e ~/.ssh/id_ed25519_antcode.pub && \
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_antcode \
    -C 'my-machine-antcode' -N ''
```

本示例与当前机器一致，不设置私钥口令，依靠本机账户和文件权限保护私钥。需要为私钥设置口令时，去掉 `-N ''`，按提示设置，并通过 SSH agent 管理解锁状态。密钥生成后检查权限与指纹：

```bash
chmod 600 ~/.ssh/id_ed25519_antcode
ssh-keygen -lf ~/.ssh/id_ed25519_antcode.pub
```

### 登记公钥

```bash
cat ~/.ssh/id_ed25519_antcode.pub
```

在 [AntCode SSH 公钥设置](https://code.alipay.com/profile/keys) 中，将输出的完整一行粘贴到 Key，填写能识别机器的标题并保存。只复制 `.pub` 文件。平台列表加载完成后，核对新增标题和指纹；页面加载中的空列表不能作为账号没有公钥的证据。若平台显示冒号分隔的 MD5 指纹，可用以下命令比对：

```bash
ssh-keygen -E md5 -lf ~/.ssh/id_ed25519_antcode.pub
```

### 配置 SSH 使用该密钥

将以下配置合并到 `~/.ssh/config`，具体主机配置放在通用 `Host *` 和其 `Include` 之前，保留已有的其他主机与企业客户端配置：

```sshconfig
Host code.alipay.com
  User git
  IdentityFile ~/.ssh/id_ed25519_antcode
  IdentitiesOnly yes
```

```bash
chmod 600 ~/.ssh/config
ssh -G git@code.alipay.com | rg '^(hostname|user|identityfile|identitiesonly) '
```

检查有效输出指向指定私钥，且 `identitiesonly yes`。当前机器通过文件直接使用密钥，无需额外启动 ssh-agent。

## 新工程复用

先查看工程的现有 remote 和分支。首次配置 `origin` 时使用代码平台提供的 SSH 地址；将下例的路径替换为实际仓库路径：

```bash
git remote -v
git branch --show-current
git remote add origin git@code.alipay.com:namespace/project.git
```

如果 `origin` 已存在且确需修改，使用 `git remote set-url origin git@code.alipay.com:namespace/project.git`，不要重复添加。新工程沿用自身分支，不因复用 SSH 配置而自动改名为 `master`。

整理改动、检查差异并提交后，在非 detached HEAD 的工作分支上建立上游：

```bash
git push -u origin "$(git branch --show-current)"
```

以后直接运行 `git push`。推送成功回执同时验证该仓库的 SSH 认证与写权限，无需为重复确认再发起一组远程检查。只检查读取权限时可用 `git ls-remote origin HEAD`；读取成功不能代替写权限验证。

## 按实际失败环节排查

### Permission denied publickey

先核对 SSH 实际选用的身份、对应私钥是否存在，以及平台是否登记了匹配公钥。Git 中的 `user.name` 和 `user.email` 是提交署名，不能代替 SSH 登录身份。

企业环境还需检查 Git 是否覆盖了默认 SSH 命令。以下命令只列出配置项名称与来源，避免输出可能含身份令牌的配置值：

```bash
git config --show-origin --name-only --get-regexp '^(core\.sshcommand|http\.extraheader|credential\.)'
```

Git 配置的 SSH 命令可能与直接运行 `ssh` 的行为不同，最终以实际 Git 请求为准。诊断输出中的 `Authentications that can continue: publickey` 表示服务器要求公钥认证；同时缺少可用磁盘密钥和 agent 身份时，需要配置已获用户授权的 SSH 身份。不要从单个默认文件缺失推断整个认证系统不存在，也不要自动改成 HTTPS。

### 企业身份信息无法刷新

本机曾出现一个独立故障：`/opt/homebrew/etc/gitconfig.lock` 是 2026 年 8 月 6 日留下的空锁文件，企业客户端日志反复记录无法锁定 Git 系统配置，导致身份信息无法更新。确认没有进程占用后，将该锁移到备份路径，企业客户端便自行刷新了配置。随后 SSH 仍返回公钥认证失败，直到本机公钥登记完成才通过。

因此，企业身份信息刷新和 SSH 登录分别验收。不能将身份标记存在、刷新成功或文件更新时间变化当作 SSH 已可登录。

遇到同类日志时，先检查具体锁文件的时间、大小及占用进程。只处理已确认遗留的锁，采用保留备份的方式；正在使用的锁不移动。不要复制企业日志中的身份令牌到文档、命令示例或新配置，也不要为了刷新身份而停用企业安全客户端。

### macOS 钥匙串弹窗

弹窗明确请求 `git-credential-osxkeychain` 读取 `code.alipay.com` 时，检查是哪一个 HTTPS Git 请求触发了凭据助手，以及编辑器是否在后台自动 fetch。它属于 HTTPS 凭据访问授权；本说明的 SSH 路径使用本机私钥。

关闭自动 fetch 只会减少后台请求。已经成功配置 SSH 时，保留正常的编辑器自动 fetch 设置，不以关闭功能代替认证修复。

## 日常交付

在用户已授权整理和推送的工程中，依次检查状态、整理本轮文件、执行必要验证、暂存已审阅路径、检查 `git diff --cached --check`、提交并推送现有目标。遵守忽略规则，保留无关改动，不使用强制推送。完成后报告提交、分支和推送结果；后续重新配置密钥、换认证方式或改其他工程必须符合对应任务授权。

当前工程的具体交付流程见 [Git 与服务器同步流程](GIT_PUSH_AND_SERVER_SYNC_SOP.md)。该说明为本地开发文档，无需加入 AI Studio 服务器上传白名单。
