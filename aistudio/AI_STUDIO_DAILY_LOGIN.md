# AI Studio 每日登录速查

本文供用户手动登录和维护当前 QuaSiD 实验隧道（实验 ID `61980069`，本地端口 `58598`）。首次安装 `ssctl` 或排查本地配置见 [SSH 直连指南](AI_STUDIO_SSH_CONNECTION.md)；客户端开发与同步遵循 [开发工作流](AI_STUDIO_DEVELOPMENT_WORKFLOW.md)。

## 1. Terminal 1：登录并建立 SSH 通道

推荐使用本地辅助脚本（自动释放旧 `ssctl` 隧道端口并建立连接）：

```bash
# 每日首次连接（传入手机端蚁家六位数字令牌，或省略参数按提示输入）
./aistudio/connect_ai_studio.sh <TOKEN>

# 当日已登录时仅重连隧道
./aistudio/connect_ai_studio.sh --reconnect
```

等价手动命令：

```bash
export PATH=~/.ssctl/bin:$PATH
ssctl login -login-name cairongyao.cry -alilang-token <TOKEN>
ssctl connect \
  -service-type ssh \
  -ais_exp_id 61980069 \
  -container-name main \
  -listen-port 58598 \
  -target-port 22
```

保持 Terminal 1 运行。

## 2. Terminal 2：SSH 登录与进入工程

```bash
ssh -p 58598 cairongyao.cry@127.0.0.1
cd /ossfs/workspace/QuaSiD
```

退出时先在 Terminal 2 执行 `exit`，再回到 Terminal 1 按 `Ctrl+C` 关闭隧道。
