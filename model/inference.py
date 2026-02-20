"""
ONNX CPU 推理引擎
在服务器上每4小时运行一次，生成预测结果 JSON
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

from model.graph_builder import (
    load_graph, WATER_QUALITY_PARAMS, WEATHER_PARAMS,
    REMOTE_SENSING_PARAMS, FEATURE_DIM
)


class TaihuInference:
    """太湖水质预测推理引擎"""

    def __init__(self, model_path="weights/stgat_best.onnx",
                 stations_path="data/stations.json",
                 data_dir="data",
                 graph_dir="data/graph",
                 history_steps=42,
                 predict_steps=14):

        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.history_steps = history_steps
        self.predict_steps = predict_steps
        self.session = None

        # 加载站点信息
        with open(stations_path, "r", encoding="utf-8") as f:
            stations_data = json.load(f)
        self.stations = stations_data["stations"]

        # 加载图结构
        self.graph = load_graph(graph_dir)

        # 加载归一化参数
        norm_file = self.data_dir / "processed" / "norm_params.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                self.norm_params = json.load(f)
        else:
            self.norm_params = {}

        # 初始化 ONNX Runtime
        self._init_session()

    def _init_session(self):
        """初始化 ONNX Runtime 推理 session"""
        if not self.model_path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            return

        try:
            import onnxruntime as ort

            # CPU 优化选项
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX 推理引擎初始化成功: {self.model_path}")
        except Exception as e:
            logger.error(f"ONNX Runtime 初始化失败: {e}")

    def _load_recent_data(self):
        """加载最近的水质和气象数据用于推理"""
        num_nodes = len(self.stations)

        # 尝试加载最近的原始数据
        raw_dir = self.data_dir / "raw"
        water_data = {}
        weather_data = {}

        # 加载最近的水质数据
        wq_files = sorted(raw_dir.glob("**/water_quality_*.json"), reverse=True)
        for wq_file in wq_files[:self.history_steps]:
            try:
                with open(wq_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for record in data.get("records", []):
                        station = record.get("station_name", "")
                        time_str = record.get("time", data.get("scrape_time", ""))
                        key = (station, time_str)
                        water_data[key] = record
            except Exception:
                continue

        # 加载最近的气象数据
        wx_files = sorted(raw_dir.glob("**/weather_*.json"), reverse=True)
        for wx_file in wx_files[:5]:
            try:
                with open(wx_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for station in data.get("stations", []):
                        weather_data[station["name"]] = station
            except Exception:
                continue

        return water_data, weather_data

    def _build_input_tensor(self, water_data, weather_data):
        """构建模型输入张量"""
        num_nodes = len(self.stations)
        now = datetime.now()

        # (1, num_nodes, history_steps, feature_dim)
        x = np.zeros((1, num_nodes, self.history_steps, FEATURE_DIM), dtype=np.float32)

        for n_idx, station in enumerate(self.stations):
            for t_idx in range(self.history_steps):
                # 从最近到最远的时间步
                time_offset = (self.history_steps - 1 - t_idx) * 4  # 小时
                t = now - timedelta(hours=time_offset)

                features = []

                # 水质参数 (11维) — 使用最近匹配的数据
                for param in WATER_QUALITY_PARAMS:
                    val = 0.0
                    # 尝试从 water_data 匹配
                    for key, record in water_data.items():
                        if station["name"] in key[0]:
                            raw_val = record.get(param)
                            if raw_val is not None:
                                val = float(raw_val)
                                # 归一化
                                if param in self.norm_params:
                                    p = self.norm_params[param]
                                    val = (val - p["mean"]) / p["std"]
                                break
                    features.append(val)

                # V2 气象参数 (15维)
                wx = weather_data.get("太湖中心") or {}
                realtime = wx.get("realtime") or {}
                wx_values = {
                    "temperature": realtime.get("temp", 0),
                    "humidity": realtime.get("humidity", 0),
                    "dewpoint": realtime.get("dewpoint", 0),
                    "precipitation": realtime.get("precip", 0),
                    "rain": realtime.get("rain", 0),
                    "wind_speed": realtime.get("wind_speed", 0),
                    "wind_direction": realtime.get("wind_direction", 0),
                    "wind_gusts": realtime.get("wind_gusts", 0),
                    "solar_radiation": 0,
                    "direct_radiation": 0,
                    "diffuse_radiation": 0,
                    "pressure": realtime.get("pressure", 0),
                    "cloud_cover": realtime.get("cloud", 0),
                    "evapotranspiration": 0,
                    "soil_temperature": 0,
                }
                for param in WEATHER_PARAMS:
                    features.append(float(wx_values.get(param, 0) or 0))

                # V2 时间编码 (6维: hour, day_of_year, month 各 sin/cos)
                day_of_year = t.timetuple().tm_yday
                features.extend([
                    np.sin(2 * np.pi * t.hour / 24),
                    np.cos(2 * np.pi * t.hour / 24),
                    np.sin(2 * np.pi * day_of_year / 365),
                    np.cos(2 * np.pi * day_of_year / 365),
                    np.sin(2 * np.pi * t.month / 12),
                    np.cos(2 * np.pi * t.month / 12),
                ])

                # 遥感特征 (3维) — 使用最近可用值或 0
                features.extend([0.0, 0.0, 0.0])

                x[0, n_idx, t_idx, :len(features)] = features[:FEATURE_DIM]

        return x

    def _denormalize(self, value, param_name):
        """反归一化"""
        if param_name in self.norm_params:
            p = self.norm_params[param_name]
            return value * p["std"] + p["mean"]
        return value

    def _get_bloom_label(self, class_idx):
        """蓝藻风险等级文本"""
        labels = {0: "无风险", 1: "轻度", 2: "中度", 3: "重度"}
        return labels.get(class_idx, "未知")

    def _get_bloom_color(self, class_idx):
        """蓝藻风险等级颜色"""
        colors = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}
        return colors.get(class_idx, "#666")

    def _get_water_quality_level(self, station_data):
        """
        根据水质参数判断水质等级 (I-V类+劣V)
        简化判断: 主要看 CODMn, NH3-N, TP
        """
        codmn = station_data.get("codmn", 0)
        nh3n = station_data.get("nh3n", 0)
        tp = station_data.get("tp", 0)

        # 地表水环境质量标准 (GB3838-2002)
        if codmn <= 2 and nh3n <= 0.15 and tp <= 0.02:
            return 1, "I类", "#22c55e"
        elif codmn <= 4 and nh3n <= 0.5 and tp <= 0.1:
            return 2, "II类", "#84cc16"
        elif codmn <= 6 and nh3n <= 1.0 and tp <= 0.2:
            return 3, "III类", "#eab308"
        elif codmn <= 10 and nh3n <= 1.5 and tp <= 0.3:
            return 4, "IV类", "#f97316"
        elif codmn <= 15 and nh3n <= 2.0 and tp <= 0.4:
            return 5, "V类", "#ef4444"
        else:
            return 6, "劣V类", "#991b1b"

    def predict(self):
        """执行一次完整推理"""
        if self.session is None:
            logger.error("推理引擎未初始化")
            return None

        now = datetime.now()
        logger.info(f"开始推理 - {now.strftime('%Y-%m-%d %H:%M')}")

        # 加载最新数据
        water_data, weather_data = self._load_recent_data()

        # 构建输入
        x = self._build_input_tensor(water_data, weather_data)

        # V2 ONNX 推理 (3 输出: 预测值, 不确定性, 蓝藻分类)
        try:
            outputs = self.session.run(None, {"node_features": x})
            wq_pred = outputs[0]      # (1, N, P, 11) 水质预测均值
            wq_log_var = outputs[1]   # (1, N, P, 11) 不确定性 (log variance)
            bloom_pred = outputs[2]    # (1, N, 4) 蓝藻分类 logits
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            return None

        # 解析结果
        results = {
            "prediction_time": now.isoformat(),
            "stations": []
        }

        for n_idx, station in enumerate(self.stations):
            station_result = {
                "id": station["id"],
                "name": station["name"],
                "lat": station["lat"],
                "lon": station["lon"],
                "basin": station.get("basin", ""),
                "type": station.get("type", ""),
            }

            # 当前值（从最近数据取）
            current = {}
            for key, record in water_data.items():
                if station["name"] in key[0]:
                    for param in WATER_QUALITY_PARAMS:
                        if record.get(param) is not None:
                            current[param] = float(record[param])
                    break
            station_result["current"] = current

            # 水质等级
            level_num, level_name, level_color = self._get_water_quality_level(current)
            station_result["water_quality_level"] = {
                "level": level_num,
                "name": level_name,
                "color": level_color
            }

            # V2 预测值（含不确定性）
            predictions = []
            for step in range(self.predict_steps):
                day_pred = {}
                day_uncertainty = {}
                for p_idx, param in enumerate(WATER_QUALITY_PARAMS):
                    val = float(wq_pred[0, n_idx, step, p_idx])
                    day_pred[param] = round(self._denormalize(val, param), 4)
                    # 不确定性 = exp(log_var) 的标准差
                    uncertainty = float(np.sqrt(np.exp(wq_log_var[0, n_idx, step, p_idx])))
                    day_uncertainty[param] = round(uncertainty, 4)
                predictions.append({
                    "date": (now + timedelta(days=step + 1)).strftime("%Y-%m-%d"),
                    "values": day_pred,
                    "uncertainty": day_uncertainty
                })
            station_result["predictions"] = predictions

            # 蓝藻预警
            bloom_probs = np.exp(bloom_pred[0, n_idx]) / np.exp(bloom_pred[0, n_idx]).sum()
            bloom_class = int(np.argmax(bloom_probs))
            station_result["bloom_warning"] = {
                "level": bloom_class,
                "label": self._get_bloom_label(bloom_class),
                "color": self._get_bloom_color(bloom_class),
                "probabilities": {
                    "无风险": round(float(bloom_probs[0]), 4),
                    "轻度": round(float(bloom_probs[1]), 4),
                    "中度": round(float(bloom_probs[2]), 4),
                    "重度": round(float(bloom_probs[3]), 4),
                }
            }

            results["stations"].append(station_result)

        # 生成预警列表（中度+重度）
        alerts = []
        for sr in results["stations"]:
            if sr["bloom_warning"]["level"] >= 2:
                alerts.append({
                    "station_id": sr["id"],
                    "station_name": sr["name"],
                    "basin": sr["basin"],
                    "level": sr["bloom_warning"]["level"],
                    "label": sr["bloom_warning"]["label"],
                    "color": sr["bloom_warning"]["color"],
                    "lat": sr["lat"],
                    "lon": sr["lon"],
                })
        results["alerts"] = sorted(alerts, key=lambda x: -x["level"])

        # 保存结果
        self._save_results(results, now)

        logger.info(f"推理完成: {len(results['stations'])} 个站点, "
                     f"{len(alerts)} 个预警")
        return results

    def _save_results(self, results, timestamp):
        """保存预测结果"""
        output_dir = self.data_dir / Path("predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存最新结果（覆盖）
        latest_path = output_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 按日期保存历史
        date_path = output_dir / f"prediction_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        with open(date_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"预测结果已保存: {latest_path}")


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="TaihuGuard 推理")
    parser.add_argument("--model", default="weights/stgat_best.onnx")
    parser.add_argument("--stations", default="data/stations.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graph-dir", default="data/graph")
    args = parser.parse_args()

    engine = TaihuInference(
        model_path=args.model,
        stations_path=args.stations,
        data_dir=args.data_dir,
        graph_dir=args.graph_dir
    )
    results = engine.predict()
    if results:
        print(f"推理完成: {len(results['stations'])} 个站点")


if __name__ == "__main__":
    main()
