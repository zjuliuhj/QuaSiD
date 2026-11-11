# AI Studio SSH 直连配置与排障指南

本文介绍 `ssctl` 首次安装、通用实验连接参数与本地配置排障。当前 QuaSiD 工程的日常连接命令见 [每日登录速查](AI_STUDIO_DAILY_LOGIN.md)。

## 1. 安装与验证 ssctl

```bash
curl -fsSL https://artifacts.antgroup-inc.cn/artifact/repositories/ant-common/paas/ssctl/scripts/vup.sh | bash
export PATH=~/.ssctl/bin:$PATH
ssupgrade check --format json
```

## 2. 通用连接参数与模板

在 AI Studio“环境配置 → 直连配置”中获取实验 ID，替换以下占位符：

- `<YOUR_ALIAS>`：公司邮箱前缀（本人花名，禁止使用 `root`）
- `<TOKEN>`：手机端蚁家显示的六位数字令牌（每日首次连接需刷新）
- `<EXPERIMENT_ID>`：实验 ID
- `<LOCAL_SSH_PORT>`：本地空闲端口，例如 `58598`

Terminal 1 登录并建立隧道（保持运行）：

```bash
ssctl login -login-name <YOUR_ALIAS> -alilang-token <TOKEN>
ssctl connect \
  -service-type ssh \
  -ais_exp_id <EXPERIMENT_ID> \
  -container-name main \
  -listen-port <LOCAL_SSH_PORT> \
  -target-port 22
```

Terminal 2 通过本机回环端口登录（输入密码时不回显字符）：

```bash
ssh -p <LOCAL_SSH_PORT> <YOUR_ALIAS>@127.0.0.1
```

退出时先在 Terminal 2 执行 `exit`，再回到 Terminal 1 按 `Ctrl+C` 关闭通道。

## 3. 本地配置文件缺失排障

若 `ssctl` 提示 `~/.ssh/known_hosts`、`~/.ssh/config` 或 `~/.ssctl/ssh_config` 不存在，执行：

```bash
mkdir -p ~/.ssh ~/.ssctl
chmod 700 ~/.ssh
touch ~/.ssh/known_hosts ~/.ssh/config ~/.ssctl/ssh_config
chmod 600 ~/.ssh/known_hosts ~/.ssh/config ~/.ssctl/ssh_config
```

## 安全约束与参考文档

只允许通过 `ssctl` SSH 通道直连实验，禁止搭建反向代理、使用 `ssh -R`、将服务暴露到公网或改用其他绕行路径。

- [AI Studio 参考文档 1](https://yuque.antfin.com/aii/aistudio/owwngeue7edilxqh)
- [AI Studio 参考文档 2](https://yuque.antfin.com/aii/aistudio/bd0tgh71bgulrmnh)
