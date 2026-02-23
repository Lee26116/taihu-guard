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

# 模型预测开关
ENABLE_MODEL_PREDICTIONS = True


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

        # 构建站点名称 -> index 映射
        self.station_name_map = {s["name"]: i for i, s in enumerate(self.stations)}

        # 加载图结构
        self.graph = load_graph(graph_dir)

        # 加载归一化参数
        norm_file = self.data_dir / "processed" / "norm_params.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                self.norm_params = json.load(f)
        else:
            self.norm_params = {}

        # 初始化 ONNX Runtime (仅当预测开关开启时)
        if ENABLE_MODEL_PREDICTIONS:
            self._init_session()

    def _init_session(self):
        """初始化 ONNX Runtime 推理 session"""
        if not self.model_path.exists():
            logger.warning(f"模型文件不存在: {self.model_path}")
            return

        try:
            import onnxruntime as ort

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

    # ==================== 数据加载 ====================

    def _load_raw_water_quality(self):
        """加载最新的水质原始数据，返回 {station_name: record}"""
        raw_dir = self.data_dir / "raw"
        wq_files = sorted(raw_dir.glob("**/water_quality_*.json"), reverse=True)

        station_data = {}
        for wq_file in wq_files[:3]:  # 最近3个文件
            try:
                with open(wq_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for record in data.get("records", []):
                    name = record.get("station_name", "")
                    if name and name not in station_data:
                        station_data[name] = record
            except Exception:
                continue

        logger.info(f"加载原始水质数据: {len(station_data)} 个断面")
        return station_data

    def _load_weather_data(self):
        """加载最新的气象数据"""
        raw_dir = self.data_dir / "raw"
        wx_files = sorted(raw_dir.glob("**/weather_*.json"), reverse=True)

        weather = {}
        for wx_file in wx_files[:2]:
            try:
                with open(wx_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for station in data.get("stations", []):
                    name = station.get("name", "")
                    if name and name not in weather:
                        weather[name] = station
            except Exception:
                continue

        return weather

    # ==================== 站点匹配 ====================

    def _match_stations(self, raw_data):
        """
        将 cnemc API 原始数据精确匹配到我们的站点。
        返回 {station_name: {param: value, ...}}
        """
        matched = {}

        for station in self.stations:
            sname = station["name"]
            if sname in raw_data:
                matched[sname] = self._extract_params(raw_data[sname])

        logger.info(f"站点匹配: {len(matched)}/{len(self.stations)} 个站点有实测数据")
        return matched

    def _extract_params(self, record):
        """从原始记录提取水质参数 (爬虫已做单位转换)"""
        params = {}
        for p in WATER_QUALITY_PARAMS:
            val = record.get(p)
            if val is not None:
                params[p] = round(float(val), 4)
        return params

    # ==================== 水质 & 蓝藻判定 ====================

    def _get_water_quality_level(self, station_data):
        """
        根据水质参数判断水质等级 (I-V类+劣V)
        采用单因子评价法 (GB3838-2002): 由最差指标决定整体等级
        评价指标: DO, CODMn, NH3-N, TP
        """
        codmn = station_data.get("codmn") or 0
        nh3n = station_data.get("nh3n") or 0
        tp = station_data.get("tp") or 0
        do = station_data.get("do") or 99  # DO 缺失时不影响评级

        # GB3838 各等级阈值: (CODMn≤, NH3-N≤, TP≤, DO≥)
        thresholds = [
            (2,  0.15, 0.02, 7.5, 1, "I类",  "#22c55e"),
            (4,  0.5,  0.1,  6.0, 2, "II类", "#84cc16"),
            (6,  1.0,  0.2,  5.0, 3, "III类", "#eab308"),
            (10, 1.5,  0.3,  3.0, 4, "IV类", "#f97316"),
            (15, 2.0,  0.4,  2.0, 5, "V类",  "#ef4444"),
        ]

        for max_codmn, max_nh3n, max_tp, min_do, lvl, name, color in thresholds:
            if codmn <= max_codmn and nh3n <= max_nh3n and tp <= max_tp and do >= min_do:
                return lvl, name, color

        return 6, "劣V类", "#991b1b"

    def _compute_bloom_warning(self, current):
        """根据 Chl-a 和藻密度计算蓝藻预警等级"""
        chla = current.get("chla")
        algae = current.get("algae_density")

        # 无 Chl-a 数据的站点返回无数据状态
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

        return {
            "level": level,
            "label": labels[level],
            "color": colors[level],
        }

    # ==================== ONNX 推理 (预测开关控制) ====================

    def _build_input_tensor(self, current_data, weather_data):
        """构建模型输入张量"""
        num_nodes = len(self.stations)
        now = datetime.now()

        x = np.zeros((1, num_nodes, self.history_steps, FEATURE_DIM), dtype=np.float32)

        for n_idx, station in enumerate(self.stations):
            station_current = current_data.get(station["name"], {})

            for t_idx in range(self.history_steps):
                time_offset = (self.history_steps - 1 - t_idx) * 4
                t = now - timedelta(hours=time_offset)

                features = []

                # 水质参数 (11维)
                for param in WATER_QUALITY_PARAMS:
                    val = station_current.get(param, 0.0)
                    if val and param in self.norm_params:
                        p = self.norm_params[param]
                        val = (val - p["mean"]) / max(p["std"], 1e-6)
                    features.append(float(val))

                # 气象参数 (15维)
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

                # 时间编码 (6维)
                day_of_year = t.timetuple().tm_yday
                features.extend([
                    np.sin(2 * np.pi * t.hour / 24),
                    np.cos(2 * np.pi * t.hour / 24),
                    np.sin(2 * np.pi * day_of_year / 365),
                    np.cos(2 * np.pi * day_of_year / 365),
                    np.sin(2 * np.pi * t.month / 12),
                    np.cos(2 * np.pi * t.month / 12),
                ])

                # 遥感特征 (3维)
                features.extend([0.0, 0.0, 0.0])

                x[0, n_idx, t_idx, :len(features)] = features[:FEATURE_DIM]

        return x

    def _denormalize(self, value, param_name):
        """反归一化"""
        if param_name in self.norm_params:
            p = self.norm_params[param_name]
            return value * p["std"] + p["mean"]
        return value

    def _run_onnx_predictions(self, current_data, weather_data):
        """运行 ONNX 模型推理，返回每个站点的预测结果"""
        if self.session is None:
            return None

        try:
            x = self._build_input_tensor(current_data, weather_data)
            outputs = self.session.run(None, {"node_features": x})
            wq_pred = outputs[0]      # (1, N, P, 11)
            wq_log_var = outputs[1]   # (1, N, P, 11)
            bloom_pred = outputs[2]   # (1, N, 4)

            # 验证输出是否合理
            temp_pred = float(wq_pred[0, 0, 0, 0])
            temp_denorm = self._denormalize(temp_pred, "water_temp")
            if temp_denorm < -5 or temp_denorm > 40:
                logger.warning(f"ONNX 水温预测异常 ({temp_denorm:.1f}°C)，跳过模型输出")
                return None

            return {
                "wq_pred": wq_pred,
                "wq_log_var": wq_log_var,
                "bloom_pred": bloom_pred
            }
        except Exception as e:
            logger.warning(f"ONNX 推理失败: {e}")
            return None

    def _fill_predictions(self, station_results, onnx_output, now):
        """将 ONNX 输出转为 dashboard 需要的 predictions 列表"""
        wq_pred = onnx_output["wq_pred"]      # (1, N, P, 11)
        wq_log_var = onnx_output["wq_log_var"] # (1, N, P, 11)
        bloom_pred = onnx_output["bloom_pred"] # (1, N, 4)

        for n_idx, sr in enumerate(station_results):
            predictions = []
            for p_idx in range(self.predict_steps):
                pred_date = now + timedelta(days=p_idx + 1)

                # 反归一化预测值
                values = {}
                uncertainty = {}
                for q_idx, param in enumerate(WATER_QUALITY_PARAMS):
                    raw_val = float(wq_pred[0, n_idx, p_idx, q_idx])
                    denorm_val = self._denormalize(raw_val, param)
                    values[param] = round(denorm_val, 4)

                    # 不确定性: exp(log_var/2) = std，再反归一化
                    log_var = float(wq_log_var[0, n_idx, p_idx, q_idx])
                    std_norm = np.exp(log_var / 2)
                    if param in self.norm_params:
                        std_real = std_norm * self.norm_params[param]["std"]
                    else:
                        std_real = std_norm
                    uncertainty[param] = round(float(std_real), 4)

                predictions.append({
                    "date": pred_date.strftime("%m-%d"),
                    "values": values,
                    "uncertainty": uncertainty,
                })

            # 蓝藻预测等级
            bloom_logits = bloom_pred[0, n_idx]
            bloom_class = int(np.argmax(bloom_logits))
            bloom_labels = {0: "无风险", 1: "轻度", 2: "中度", 3: "重度"}
            bloom_colors = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}
            sr["bloom_forecast"] = {
                "level": bloom_class,
                "label": bloom_labels[bloom_class],
                "color": bloom_colors[bloom_class],
            }

            sr["predictions"] = predictions

    # ==================== 主推理流程 ====================

    def predict(self):
        """执行一次完整推理"""
        now = datetime.now()
        logger.info(f"开始推理 - {now.strftime('%Y-%m-%d %H:%M')}")

        # 1. 加载原始数据
        raw_water = self._load_raw_water_quality()
        weather_data = self._load_weather_data()

        # 2. 匹配站点 (精确名称匹配，不插值)
        station_current = self._match_stations(raw_water)

        # 3. 构建每个站点的结果
        results = {
            "prediction_time": now.isoformat(),
            "stations": []
        }

        for n_idx, station in enumerate(self.stations):
            sname = station["name"]
            current = station_current.get(sname, {})

            station_result = {
                "id": station.get("id", sname),
                "name": sname,
                "lat": station["lat"],
                "lon": station["lon"],
                "basin": station.get("basin", ""),
                "type": station.get("type", ""),
                "current": current,
            }

            # 水质等级
            level_num, level_name, level_color = self._get_water_quality_level(current)
            station_result["water_quality_level"] = {
                "level": level_num,
                "name": level_name,
                "color": level_color
            }

            # 蓝藻预警
            station_result["bloom_warning"] = self._compute_bloom_warning(current)

            # 预测: 仅当模型开关开启且有 ONNX 输出时才有
            station_result["predictions"] = []

            results["stations"].append(station_result)

        # 4. 如果预测开关开启，尝试 ONNX 推理
        if ENABLE_MODEL_PREDICTIONS:
            onnx_output = self._run_onnx_predictions(station_current, weather_data)
            if onnx_output:
                self._fill_predictions(results["stations"], onnx_output, now)
                results["has_prediction"] = True
                logger.info("ONNX 模型推理成功，14 天预测已生成")
            else:
                results["has_prediction"] = False
                logger.info("ONNX 模型不可用，仅返回实测数据")
        else:
            results["has_prediction"] = False
            logger.info("模型预测已关闭 (ENABLE_MODEL_PREDICTIONS=False)，仅返回实测数据")

        # 5. 生成预警列表
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

        # 6. 保存结果
        self._save_results(results, now)

        logger.info(f"推理完成: {len(results['stations'])} 个站点, "
                     f"{len(alerts)} 个预警")
        return results

    def _save_results(self, results, timestamp):
        """保存预测结果"""
        output_dir = self.data_dir / Path("predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        latest_path = output_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

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
