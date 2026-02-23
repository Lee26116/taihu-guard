"""
TaihuGuard FastAPI 后端服务
端口: 8087
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.data_service import DataService
from api.predict import PredictionService

load_dotenv()

# 配置
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8087"))
DATA_DIR = os.getenv("DATA_DIR", "data")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "weights")

# 初始化
app = FastAPI(
    title="TaihuGuard API",
    description="太湖流域水质智能预测与预警系统 API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务实例
data_service = DataService(data_dir=DATA_DIR)
prediction_service = PredictionService(
    model_path=f"{WEIGHTS_DIR}/stgat_best.onnx",
    data_dir=DATA_DIR
)

# 站点元数据缓存
_stations_meta = None


def _load_stations_meta():
    """加载 stations.json 元数据 (缓存)"""
    global _stations_meta
    if _stations_meta is None:
        stations_file = Path(DATA_DIR) / "stations.json"
        if stations_file.exists():
            with open(stations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _stations_meta = {s["name"]: s for s in data.get("stations", [])}
        else:
            _stations_meta = {}
    return _stations_meta


def _get_water_quality_level(params):
    """根据 GB3838-2002 单因子评价法判定水质等级"""
    codmn = params.get("codmn") or 0
    nh3n = params.get("nh3n") or 0
    tp = params.get("tp") or 0
    do_val = params.get("do") or 99

    thresholds = [
        (2, 0.15, 0.02, 7.5, 1, "I类", "#22c55e"),
        (4, 0.5, 0.1, 6.0, 2, "II类", "#84cc16"),
        (6, 1.0, 0.2, 5.0, 3, "III类", "#eab308"),
        (10, 1.5, 0.3, 3.0, 4, "IV类", "#f97316"),
        (15, 2.0, 0.4, 2.0, 5, "V类", "#ef4444"),
    ]

    for max_codmn, max_nh3n, max_tp, min_do, lvl, name, color in thresholds:
        if codmn <= max_codmn and nh3n <= max_nh3n and tp <= max_tp and do_val >= min_do:
            return {"level": lvl, "name": name, "color": color}

    return {"level": 6, "name": "劣V类", "color": "#991b1b"}


def _get_bloom_warning(params):
    """根据 Chl-a 和藻密度计算蓝藻预警"""
    chla = params.get("chla")
    algae = params.get("algae_density")

    if chla is None and algae is None:
        return {"level": -1, "label": "无数据", "color": "#6b7280"}

    chla = chla or 0
    algae = algae or 0

    if chla >= 64 or algae >= 20000:
        level = 3
    elif chla >= 26 or algae >= 5000:
        level = 2
    elif chla >= 10 or algae >= 1000:
        level = 1
    else:
        level = 0

    labels = {0: "无风险", 1: "轻度", 2: "中度", 3: "重度"}
    colors = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}
    return {"level": level, "label": labels[level], "color": colors[level]}


def _format_raw_data(raw_wq):
    """将爬虫原始记录转为前端需要的站点格式"""
    stations_meta = _load_stations_meta()
    records = raw_wq.get("records", [])

    stations = []
    alerts = []

    for record in records:
        name = record.get("station_name", "")
        meta = stations_meta.get(name, {})

        # 提取水质参数
        param_keys = [
            "water_temp", "ph", "do", "conductivity", "turbidity",
            "codmn", "nh3n", "tp", "tn", "chla", "algae_density"
        ]
        current = {}
        for k in param_keys:
            v = record.get(k)
            if v is not None:
                current[k] = float(v)

        wq_level = _get_water_quality_level(current)
        bloom = _get_bloom_warning(current)

        station = {
            "id": meta.get("id", name),
            "name": name,
            "lat": meta.get("lat", 0),
            "lon": meta.get("lon", 0),
            "basin": meta.get("basin", ""),
            "type": meta.get("type", ""),
            "current": current,
            "water_quality_level": wq_level,
            "bloom_warning": bloom,
            "predictions": [],
        }
        stations.append(station)

        if bloom["level"] >= 2:
            alerts.append({
                "station_id": station["id"],
                "station_name": name,
                "basin": station["basin"],
                "level": bloom["level"],
                "label": bloom["label"],
                "color": bloom["color"],
                "lat": station["lat"],
                "lon": station["lon"],
            })

    alerts.sort(key=lambda x: -x["level"])
    return stations, alerts


@app.get("/api/config")
async def get_config():
    """前端配置（Mapbox Token 等）"""
    return {
        "mapbox_token": os.getenv("MAPBOX_TOKEN", ""),
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "TaihuGuard API",
        "time": datetime.now().isoformat(),
        "model_loaded": prediction_service.is_loaded(),
        "data_available": data_service.has_data()
    }


@app.get("/api/latest")
async def get_latest():
    """获取所有站点最新水质数据 + 预测结果"""
    # 尝试读取最新预测
    prediction = data_service.get_latest_prediction()

    if prediction:
        return {
            "update_time": prediction.get("prediction_time", ""),
            "stations": prediction.get("stations", []),
            "alerts": prediction.get("alerts", []),
            "has_prediction": True
        }

    # 尝试读取最新实测数据
    latest_data = data_service.get_latest_water_quality()
    if latest_data:
        stations, alerts = _format_raw_data(latest_data)
        return {
            "update_time": latest_data.get("scrape_time", ""),
            "stations": stations,
            "alerts": alerts,
            "has_prediction": False
        }

    # 无数据
    return {"stations": [], "alerts": [], "has_prediction": False, "message": "暂无数据"}


@app.get("/api/station/{station_id}")
async def get_station_detail(station_id: str):
    """获取单站点详情: 7天历史 + 预测"""
    # 从预测结果中找
    prediction = data_service.get_latest_prediction()
    station_pred = None
    if prediction:
        for s in prediction.get("stations", []):
            if s["id"] == station_id:
                station_pred = s
                break

    # 历史数据
    history = data_service.get_station_history(station_id, days=7)

    if not station_pred and not history:
        raise HTTPException(status_code=404, detail=f"站点 {station_id} 未找到")

    return {
        "station_id": station_id,
        "current": station_pred.get("current", {}) if station_pred else {},
        "water_quality_level": station_pred.get("water_quality_level", {}) if station_pred else {},
        "bloom_warning": station_pred.get("bloom_warning", {}) if station_pred else {},
        "predictions": station_pred.get("predictions", []) if station_pred else [],
        "history": history
    }


@app.get("/api/alerts")
async def get_alerts():
    """获取当前预警列表 (中度+重度)"""
    prediction = data_service.get_latest_prediction()
    if prediction and "alerts" in prediction:
        return {
            "update_time": prediction.get("prediction_time", ""),
            "alerts": prediction["alerts"]
        }

    # 尝试从实测数据生成预警
    latest_data = data_service.get_latest_water_quality()
    if latest_data:
        _, alerts = _format_raw_data(latest_data)
        return {
            "update_time": latest_data.get("scrape_time", ""),
            "alerts": alerts
        }

    return {"update_time": "", "alerts": []}


@app.get("/api/model/metrics")
async def get_model_metrics():
    """获取模型评估指标"""
    metrics_file = Path(WEIGHTS_DIR) / "evaluation_report.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "status": "pending_retrain",
        "note": "模型尚未在真实数据上完成训练"
    }


@app.get("/api/stations")
async def get_stations():
    """获取所有站点元数据"""
    stations_file = Path(DATA_DIR) / "stations.json"
    if stations_file.exists():
        with open(stations_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="站点数据未找到")


# 挂载静态文件 (Dashboard)
dashboard_dir = Path("dashboard")
if dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


def start():
    """启动服务"""
    import uvicorn
    logger.info(f"TaihuGuard API 启动: {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    start()
