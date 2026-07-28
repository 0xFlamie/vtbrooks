#!/bin/bash
# vtbrooks 部署脚本 — ccvps 服务器
set -e

APP_DIR="/opt/vtbrooks"
REPO="https://github.com/flamie/vtbrooks.git"
VENV="$APP_DIR/venv"
SERVICE="vtbrooks.service"

echo "=== vtbrooks 部署 ==="

if [ ! -d "$APP_DIR" ]; then
    echo "克隆仓库..."
    git clone "$REPO" "$APP_DIR"
fi

cd "$APP_DIR"
echo "拉取最新代码..."
git pull origin master 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "创建 venv (Python 3.11)..."
    python3.11 -m venv "$VENV"
fi

echo "安装依赖..."
source "$VENV/bin/activate"
pip install -q numpy pandas

# 密钥只保存在服务器本地 .env, 不入库
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "WARN: $ENV_FILE 不存在，已创建空模板，请填入真实密钥后重启服务"
    cat > "$ENV_FILE" << 'ENVEOF'
VT_TELEGRAM_TOKEN=
VT_TELEGRAM_CHAT=
VT_DS_API_KEY=
ENVEOF
    chmod 600 "$ENV_FILE"
fi

# 安装 systemd 服务
cat > "/etc/systemd/system/$SERVICE" << SYSTEMD
[Unit]
Description=VT Brooks Signal Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/venv/bin/python vt_vote_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD

systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

echo ""
echo "=== 部署完成 ==="
systemctl status "$SERVICE" --no-pager -l | head -15
echo ""
echo "查看日志: journalctl -u $SERVICE -f"
echo "重启服务: systemctl restart $SERVICE"
