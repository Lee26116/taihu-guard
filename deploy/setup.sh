#!/bin/bash
# TaihuGuard 一键部署脚本
# 用法: sudo ./deploy/setup.sh

set -e

echo "============================================"
echo "  TaihuGuard 部署脚本"
echo "============================================"

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "项目目录: $PROJECT_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "安装 Python3..."
    apt-get update && apt-get install -y python3 python3-pip python3-venv
fi

# 检查 Chrome (Selenium 需要)
if ! command -v google-chrome &> /dev/null && ! command -v chromium-browser &> /dev/null; then
    echo "安装 Chromium..."
    apt-get install -y chromium-browser chromium-chromedriver || true
fi

# 创建虚拟环境
echo "创建 Python 虚拟环境..."
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate

# 安装依赖
echo "安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建 .env 文件
if [ ! -f .env ]; then
    echo "创建 .env 配置文件..."
    cp .env.example .env
    echo "请编辑 .env 文件填入实际 API Key: vim $PROJECT_DIR/.env"
fi

# 创建必要目录
mkdir -p data/raw data/processed data/graph data/predictions logs weights

# 构建图结构（如果还没有）
if [ ! -f data/graph/graph.json ]; then
    echo "构建图结构..."
    python scripts/build_graph.py || echo "图结构构建跳过（需要数据）"
fi

# 配置 systemd 服务
echo "配置 systemd 服务..."
VENV_PATH="$PROJECT_DIR/venv"

# 替换路径占位符
sed "s|/path/to/taihu-guard|$PROJECT_DIR|g; s|/path/to/venv|$VENV_PATH|g" \
    deploy/systemd/taihu-api.service > /etc/systemd/system/taihu-api.service

# 重载 systemd
systemctl daemon-reload
systemctl enable taihu-api
systemctl start taihu-api

echo "API 服务状态:"
systemctl status taihu-api --no-pager || true

# 配置 cron
echo "配置定时任务..."
CRON_CMD="0 0,4,8,12,16,20 * * * cd $PROJECT_DIR && $VENV_PATH/bin/python scraper/scheduler.py >> logs/cron.log 2>&1"
(crontab -l 2>/dev/null | grep -v "taihu-guard"; echo "$CRON_CMD") | crontab -
echo "Cron 任务已配置"

# 配置 Nginx（如果已安装）
if command -v nginx &> /dev/null; then
    echo "配置 Nginx..."
    sed "s|/path/to/taihu-guard|$PROJECT_DIR|g" \
        deploy/nginx.conf > /etc/nginx/sites-available/taihu-guard

    if [ ! -L /etc/nginx/sites-enabled/taihu-guard ]; then
        ln -s /etc/nginx/sites-available/taihu-guard /etc/nginx/sites-enabled/
    fi

    nginx -t && systemctl reload nginx
    echo "Nginx 已配置"
else
    echo "Nginx 未安装，跳过。API 直接通过端口 8087 访问。"
fi

echo ""
echo "============================================"
echo "  部署完成!"
echo ""
echo "  API 地址: http://localhost:8087"
echo "  Dashboard: http://localhost:8087/ (或 Nginx 代理)"
echo "  健康检查: http://localhost:8087/api/health"
echo ""
echo "  重要: 请编辑 .env 填入 API Key"
echo "  文件: $PROJECT_DIR/.env"
echo "============================================"
