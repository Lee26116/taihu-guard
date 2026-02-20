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

    # 如果没有预测数据，返回最新的实测数据
    if not prediction:
        latest_data = data_service.get_latest_water_quality()
        if not latest_data:
            # 返回 demo 数据
            return _get_demo_data()
        return {
            "update_time": latest_data.get("scrape_time", ""),
            "stations": latest_data.get("records", []),
            "has_prediction": False
        }

    return {
        "update_time": prediction.get("prediction_time", ""),
        "stations": prediction.get("stations", []),
        "alerts": prediction.get("alerts", []),
        "has_prediction": True
    }


@app.get("/api/station/{station_id}")
async def get_station_detail(station_id: str):
    """获取单站点详情: 7天历史 + 7天预测"""
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
        # 尝试 demo 数据
        demo = _get_demo_station(station_id)
        if demo:
            return demo
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

    # Demo 预警
    return _get_demo_alerts()


@app.get("/api/model/metrics")
async def get_model_metrics():
    """获取模型评估指标"""
    metrics_file = Path(WEIGHTS_DIR) / "evaluation_report.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 返回示例指标
    return {
        "water_quality": {
            "chla": {"mae": 3.21, "rmse": 5.12, "r2": 0.89},
            "do": {"mae": 0.42, "rmse": 0.68, "r2": 0.93},
            "tp": {"mae": 0.012, "rmse": 0.018, "r2": 0.87},
            "tn": {"mae": 0.31, "rmse": 0.52, "r2": 0.85},
            "nh3n": {"mae": 0.08, "rmse": 0.12, "r2": 0.88},
            "codmn": {"mae": 0.45, "rmse": 0.72, "r2": 0.86}
        },
        "bloom_warning": {
            "accuracy": 0.87,
            "f1_macro": 0.85,
            "auc_roc": 0.94
        },
        "feature_importance": [
            {"feature": "水温", "importance": 0.18},
            {"feature": "风速", "importance": 0.15},
            {"feature": "总磷", "importance": 0.13},
            {"feature": "太阳辐射", "importance": 0.12},
            {"feature": "气温", "importance": 0.10},
            {"feature": "叶绿素a", "importance": 0.09},
            {"feature": "溶解氧", "importance": 0.08},
            {"feature": "降水", "importance": 0.07},
            {"feature": "湿度", "importance": 0.05},
            {"feature": "氨氮", "importance": 0.03}
        ],
        "note": "示例数据，模型训练后将更新"
    }


@app.get("/api/stations")
async def get_stations():
    """获取所有站点元数据"""
    stations_file = Path(DATA_DIR) / "stations.json"
    if stations_file.exists():
        with open(stations_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="站点数据未找到")


def _get_demo_data():
    """生成 Demo 数据"""
    import numpy as np
    rng = np.random.RandomState(42)

    stations_file = Path(DATA_DIR) / "stations.json"
    if not stations_file.exists():
        return {"stations": [], "has_prediction": False, "demo": True}

    with open(stations_file, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    now = datetime.now()
    month = now.month
    season = np.sin((month - 3) * np.pi / 6)

    results = []
    total = len(stations_data["stations"])
    for idx, station in enumerate(stations_data["stations"]):
        # 为 demo 效果，让部分站点模拟高风险数据
        demo_boost = 0
        if idx < total * 0.1:       # ~10% 重度
            demo_boost = 6
        elif idx < total * 0.2:     # ~10% 中度
            demo_boost = 3
        elif idx < total * 0.35:    # ~15% 轻度
            demo_boost = 1.5

        effective_season = max(season, 0) + demo_boost * 0.4

        current = {
            "water_temp": round(15 + 10 * season + rng.normal(0, 2), 1),
            "ph": round(7.5 + rng.normal(0, 0.3), 2),
            "do": round(max(8.0 - 2 * effective_season + rng.normal(0, 0.5), 2), 2),
            "conductivity": round(400 + rng.normal(0, 30), 1),
            "turbidity": round(10 + 5 * effective_season + rng.normal(0, 2), 1),
            "codmn": round(4.0 + 2 * effective_season + rng.normal(0, 0.5), 2),
            "nh3n": round(max(0.3 + 0.2 * effective_season + rng.normal(0, 0.1), 0.01), 3),
            "tp": round(max(0.06 + 0.05 * effective_season + rng.normal(0, 0.01), 0.001), 4),
            "tn": round(max(1.5 + 0.8 * effective_season + rng.normal(0, 0.3), 0.1), 2),
            "chla": round(max(5 + 25 * effective_season + rng.normal(0, 5), 0), 1),
            "algae_density": round(max(300 + 3000 * effective_season + rng.normal(0, 300), 0), 0),
        }

        # 水质等级判断
        codmn = current["codmn"]
        nh3n = current["nh3n"]
        tp = current["tp"]
        if codmn <= 2 and nh3n <= 0.15 and tp <= 0.02:
            level = {"level": 1, "name": "I类", "color": "#22c55e"}
        elif codmn <= 4 and nh3n <= 0.5 and tp <= 0.1:
            level = {"level": 2, "name": "II类", "color": "#84cc16"}
        elif codmn <= 6 and nh3n <= 1.0 and tp <= 0.2:
            level = {"level": 3, "name": "III类", "color": "#eab308"}
        elif codmn <= 10 and nh3n <= 1.5 and tp <= 0.3:
            level = {"level": 4, "name": "IV类", "color": "#f97316"}
        elif codmn <= 15 and nh3n <= 2.0 and tp <= 0.4:
            level = {"level": 5, "name": "V类", "color": "#ef4444"}
        else:
            level = {"level": 6, "name": "劣V类", "color": "#991b1b"}

        # 蓝藻预警
        bloom_level = 0
        if current["chla"] >= 64 or current["algae_density"] >= 20000:
            bloom_level = 3
        elif current["chla"] >= 26 or current["algae_density"] >= 5000:
            bloom_level = 2
        elif current["chla"] >= 10 or current["algae_density"] >= 1000:
            bloom_level = 1

        bloom_labels = {0: "无风险", 1: "轻度", 2: "中度", 3: "重度"}
        bloom_colors = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}

        # 生成14天预测 (V2)
        predictions = []
        for day in range(1, 15):
            day_values = {}
            day_uncertainty = {}
            for k, v in current.items():
                drift = rng.normal(0, abs(v) * 0.03 * day) if v != 0 else 0
                day_values[k] = round(max(v + drift, 0), 4)
                # 不确定性随时间增大
                day_uncertainty[k] = round(abs(v) * 0.05 * (1 + day * 0.15), 4)
            predictions.append({
                "date": (now + timedelta(days=day)).strftime("%Y-%m-%d"),
                "values": day_values,
                "uncertainty": day_uncertainty
            })

        results.append({
            "id": station["id"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "basin": station.get("basin", ""),
            "type": station.get("type", ""),
            "current": current,
            "water_quality_level": level,
            "bloom_warning": {
                "level": bloom_level,
                "label": bloom_labels[bloom_level],
                "color": bloom_colors[bloom_level]
            },
            "predictions": predictions
        })

    # 生成预警列表
    alerts = []
    for s in results:
        if s["bloom_warning"]["level"] >= 2:
            alerts.append({
                "station_id": s["id"],
                "station_name": s["name"],
                "basin": s["basin"],
                "level": s["bloom_warning"]["level"],
                "label": s["bloom_warning"]["label"],
                "color": s["bloom_warning"]["color"],
                "lat": s["lat"],
                "lon": s["lon"],
            })

    return {
        "update_time": now.isoformat(),
        "stations": results,
        "alerts": sorted(alerts, key=lambda x: -x["level"]),
        "has_prediction": True,
        "demo": True
    }


def _get_demo_station(station_id):
    """单站点 demo 数据"""
    demo = _get_demo_data()
    for s in demo.get("stations", []):
        if s["id"] == station_id:
            return {
                "station_id": station_id,
                "current": s["current"],
                "water_quality_level": s["water_quality_level"],
                "bloom_warning": s["bloom_warning"],
                "predictions": s["predictions"],
                "history": [],
                "demo": True
            }
    return None


def _get_demo_alerts():
    """Demo 预警"""
    demo = _get_demo_data()
    return {
        "update_time": demo["update_time"],
        "alerts": demo.get("alerts", []),
        "demo": True
    }


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
