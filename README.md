# TaihuGuard — 太湖流域水质智能预测与预警系统

AI-Powered Water Quality Prediction & Early Warning System for Taihu Lake Basin

---

## 项目简介 | Overview

基于深度学习的太湖流域水质预测与蓝藻水华预警系统。融合多源数据（地面水质监测站、卫星遥感、气象数据），使用时空图注意力网络（ST-GAT）进行水质预测，提供未来 3-7 天的水质变化趋势和蓝藻爆发概率预警。

A deep learning-based water quality prediction and cyanobacterial bloom early warning system for the Taihu Lake Basin. It integrates multi-source data (ground monitoring stations, satellite remote sensing, meteorological data) and uses a Spatial-Temporal Graph Attention Network (ST-GAT) for water quality forecasting.

## 技术栈 | Tech Stack

- **模型**: PyTorch + PyG (ST-GAT) → ONNX (CPU 推理)
- **数据采集**: Selenium 爬虫 + 和风天气 API + Open-Meteo + GEE
- **后端**: FastAPI (端口 8087)
- **前端**: HTML/CSS/JS + Mapbox GL JS + ECharts 5
- **部署**: Nginx + systemd + cron

## 三阶段工作流 | Three-Phase Workflow

### Phase 1: 本地开发
```bash
# 克隆项目
git clone <repo-url>
cd taihu-guard

# 创建环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际 API key

# 收集历史数据
python scripts/download_history.py

# 构建图结构
python scripts/build_graph.py
```

### Phase 2: RunPod GPU 训练
```bash
# 在 RunPod 上
git clone <repo-url>
cd taihu-guard
pip install -r requirements.txt

# 训练模型
python model/train.py --epochs 200 --batch_size 16 --lr 1e-3

# 评估
python model/evaluate.py --checkpoint weights/stgat_best.pt

# 导出 ONNX
python model/export_onnx.py --checkpoint weights/stgat_best.pt

# 推送模型
git add weights/ && git commit -m "Add trained model" && git push
```

### Phase 3: 服务器部署
```bash
# 在 Linux 服务器上
git clone <repo-url>
cd taihu-guard

# 一键部署
chmod +x deploy/setup.sh
sudo ./deploy/setup.sh

# 或手动部署
pip install -r requirements.txt
cp .env.example .env && vim .env

# 启动 API 服务
sudo systemctl start taihu-api
sudo systemctl enable taihu-api

# 配置 cron
crontab deploy/crontab.txt

# 配置 Nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/taihu-guard
sudo ln -s /etc/nginx/sites-available/taihu-guard /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## API 端点 | Endpoints

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/latest` | GET | 所有站点最新水质 + 预测 |
| `/api/station/{id}` | GET | 单站点历史 + 预测详情 |
| `/api/alerts` | GET | 当前预警列表 |
| `/api/model/metrics` | GET | 模型评估指标 |
| `/api/health` | GET | 健康检查 |

## 项目结构 | Structure

```
taihu-guard/
├── data/           # 数据文件
├── scraper/        # 数据采集脚本
├── model/          # ST-GAT 模型代码
├── dashboard/      # 前端 Dashboard
├── api/            # FastAPI 后端
├── deploy/         # 部署配置
├── weights/        # 模型权重
└── scripts/        # 工具脚本
```

## 许可证 | License

MIT
