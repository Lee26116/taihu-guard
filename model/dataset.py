"""
PyTorch Dataset / DataLoader
将多源数据组装为 ST-GAT 模型输入
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from model.graph_builder import (
    WATER_QUALITY_PARAMS, WEATHER_PARAMS, REMOTE_SENSING_PARAMS,
    FEATURE_DIM, load_graph
)


class TaihuDataset(Dataset):
    """
    太湖水质时空图数据集

    每个样本 = (历史时间窗口, 标签):
      - x: (num_nodes, history_steps, feature_dim)  # 节点特征
      - y_wq: (num_nodes, predict_steps, 11)          # 水质预测标签
      - y_bloom: (num_nodes,)                          # 蓝藻风险分类标签
    """

    def __init__(self, data_dir="data", graph_dir="data/graph",
                 history_steps=18, predict_steps=7,
                 start_date=None, end_date=None, mode="train"):
        """
        Args:
            data_dir: 数据根目录
            graph_dir: 图结构目录
            history_steps: 历史时间步数 (18 = 3天 × 每天6个4小时间隔)
            predict_steps: 预测步数 (7 = 未来7天，每天取一个值)
            start_date: 数据起始日期
            end_date: 数据结束日期
            mode: 'train' / 'val' / 'test'
        """
        self.history_steps = history_steps
        self.predict_steps = predict_steps
        self.mode = mode

        # 加载图结构
        self.graph = load_graph(graph_dir)
        self.num_nodes = self.graph["num_nodes"]
        self.node_ids = self.graph["node_ids"]

        # 加载数据
        self.data_dir = Path(data_dir)
        self._load_data(start_date, end_date)

        logger.info(f"Dataset [{mode}]: {len(self)} 个样本, "
                     f"{self.num_nodes} 节点, "
                     f"history={history_steps}, predict={predict_steps}")

    def _load_data(self, start_date, end_date):
        """加载并对齐所有数据源"""
        # 加载水质数据
        self.wq_data = self._load_water_quality(start_date, end_date)

        # 加载气象数据
        self.weather_data = self._load_weather(start_date, end_date)

        # 加载遥感数据（训练时可用，推理时用最近值填充）
        self.rs_data = self._load_remote_sensing()

        # 构建时间索引
        self._build_time_index()

        # 计算归一化参数
        if self.mode == "train":
            self._compute_normalization()
        else:
            self._load_normalization()

    def _load_water_quality(self, start_date, end_date):
        """加载水质数据"""
        processed_file = self.data_dir / "processed" / "water_quality.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file, parse_dates=["time"])
            if start_date:
                df = df[df["time"] >= start_date]
            if end_date:
                df = df[df["time"] <= end_date]
            return df

        # 如果没有预处理文件，从原始 JSON 构建
        raw_dir = self.data_dir / "raw"
        records = []
        for json_file in sorted(raw_dir.glob("**/water_quality_*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for r in data.get("records", []):
                    records.append(r)

        if not records:
            logger.warning("未找到水质数据，使用合成数据")
            return self._generate_synthetic_wq(start_date, end_date)

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df

    def _load_weather(self, start_date, end_date):
        """加载气象数据"""
        processed_file = self.data_dir / "processed" / "weather.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file, parse_dates=["time"])
            if start_date:
                df = df[df["time"] >= start_date]
            if end_date:
                df = df[df["time"] <= end_date]
            return df

        # 尝试从 Open-Meteo 历史数据加载
        history_file = self.data_dir / "raw" / "weather_history_all.csv"
        if history_file.exists():
            df = pd.read_csv(history_file, parse_dates=["time"])
            return df

        logger.warning("未找到气象数据，使用合成数据")
        return self._generate_synthetic_weather(start_date, end_date)

    def _load_remote_sensing(self):
        """加载遥感数据"""
        s2_file = self.data_dir / "raw" / "sentinel2_features.csv"
        lst_file = self.data_dir / "raw" / "modis_lst.csv"

        rs_data = {}
        if s2_file.exists():
            s2 = pd.read_csv(s2_file)
            for _, row in s2.iterrows():
                key = (row["station_id"], row["month"])
                rs_data[key] = {"ndci": row.get("ndci"), "fai": row.get("fai")}

        if lst_file.exists():
            lst = pd.read_csv(lst_file)
            for _, row in lst.iterrows():
                key = (row["station_id"], row.get("week_start", "")[:7])
                if key in rs_data:
                    rs_data[key]["lst"] = row.get("lst_celsius")
                else:
                    rs_data[key] = {"lst": row.get("lst_celsius")}

        return rs_data

    def _generate_synthetic_wq(self, start_date, end_date):
        """生成合成水质数据（开发测试用）"""
        logger.info("生成合成水质数据用于测试...")
        start = pd.Timestamp(start_date or "2021-06-01")
        end = pd.Timestamp(end_date or "2025-12-31")
        times = pd.date_range(start, end, freq="4h")

        records = []
        rng = np.random.RandomState(42)

        for t in times:
            for sid in self.node_ids:
                # 带季节性的合成数据
                month = t.month
                # 夏季水温高、叶绿素高（蓝藻高发）
                season_factor = np.sin((month - 3) * np.pi / 6)

                record = {
                    "station_id": sid,
                    "time": t,
                    "water_temp": 15 + 10 * season_factor + rng.normal(0, 2),
                    "ph": 7.5 + rng.normal(0, 0.5),
                    "do": 8.0 - 2 * season_factor + rng.normal(0, 1),
                    "conductivity": 400 + rng.normal(0, 50),
                    "turbidity": 10 + 5 * abs(season_factor) + rng.normal(0, 3),
                    "codmn": 4.0 + 2 * season_factor + rng.normal(0, 1),
                    "nh3n": 0.5 + 0.3 * season_factor + rng.normal(0, 0.15),
                    "tp": 0.08 + 0.04 * season_factor + rng.normal(0, 0.02),
                    "tn": 2.0 + 1.0 * season_factor + rng.normal(0, 0.5),
                    "chla": max(5 + 30 * max(season_factor, 0) + rng.normal(0, 5), 0),
                    "algae_density": max(500 + 3000 * max(season_factor, 0) + rng.normal(0, 300), 0),
                }
                records.append(record)

        return pd.DataFrame(records)

    def _generate_synthetic_weather(self, start_date, end_date):
        """生成合成气象数据（开发测试用）"""
        logger.info("生成合成气象数据用于测试...")
        start = pd.Timestamp(start_date or "2021-06-01")
        end = pd.Timestamp(end_date or "2025-12-31")
        times = pd.date_range(start, end, freq="1h")

        records = []
        rng = np.random.RandomState(123)

        for t in times:
            month = t.month
            season_factor = np.sin((month - 3) * np.pi / 6)

            record = {
                "time": t,
                "temperature": 15 + 15 * season_factor + rng.normal(0, 3),
                "humidity": 65 + 15 * season_factor + rng.normal(0, 10),
                "precipitation": max(rng.exponential(1.0) * (0.5 + 0.5 * season_factor), 0),
                "wind_speed": 3.0 + rng.exponential(2.0),
                "wind_direction": rng.uniform(0, 360),
                "solar_radiation": max(200 + 300 * season_factor + rng.normal(0, 50), 0),
                "pressure": 1013 + rng.normal(0, 5),
                "location_name": "太湖中心"
            }
            records.append(record)

        return pd.DataFrame(records)

    def _build_time_index(self):
        """构建统一时间索引"""
        if self.wq_data is None or len(self.wq_data) == 0:
            self.time_index = []
            return

        # 找到水质数据的时间范围，以4小时为间隔
        if "time" in self.wq_data.columns:
            min_time = self.wq_data["time"].min()
            max_time = self.wq_data["time"].max()
            self.time_index = pd.date_range(min_time, max_time, freq="4h")
        else:
            self.time_index = []

        # 需要足够长的时间窗口
        min_len = self.history_steps + self.predict_steps * 6  # predict_steps 天 × 6 步/天
        if len(self.time_index) < min_len:
            logger.warning(f"时间序列长度不足: {len(self.time_index)} < {min_len}")

    def _compute_normalization(self):
        """计算训练集的归一化参数"""
        self.norm_params = {}

        # 水质参数归一化
        for param in WATER_QUALITY_PARAMS:
            if param in self.wq_data.columns:
                vals = self.wq_data[param].dropna()
                self.norm_params[param] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if vals.std() > 0 else 1.0
                }

        # 保存归一化参数
        norm_file = self.data_dir / "processed" / "norm_params.json"
        norm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_file, "w") as f:
            json.dump(self.norm_params, f, indent=2)

    def _load_normalization(self):
        """加载训练集的归一化参数"""
        norm_file = self.data_dir / "processed" / "norm_params.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                self.norm_params = json.load(f)
        else:
            self._compute_normalization()

    def _normalize(self, value, param_name):
        """归一化单个值"""
        if param_name in self.norm_params:
            p = self.norm_params[param_name]
            return (value - p["mean"]) / p["std"]
        return value

    def _time_encoding(self, timestamp):
        """计算时间编码 [hour_sin, hour_cos, month_sin, month_cos]"""
        hour = timestamp.hour
        month = timestamp.month
        return [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ]

    def _get_features_at_time(self, time_step, node_idx):
        """获取指定时间步和节点的特征向量"""
        features = []
        station_id = self.node_ids[node_idx]

        # 1. 水质参数 (11维)
        wq_mask = (self.wq_data.get("station_id") == station_id) if "station_id" in self.wq_data.columns else pd.Series(False, index=self.wq_data.index)
        if "time" in self.wq_data.columns:
            time_mask = (self.wq_data["time"] >= time_step - timedelta(hours=2)) & \
                        (self.wq_data["time"] <= time_step + timedelta(hours=2))
            matched = self.wq_data[wq_mask & time_mask]
        else:
            matched = pd.DataFrame()

        for param in WATER_QUALITY_PARAMS:
            if len(matched) > 0 and param in matched.columns:
                val = matched[param].iloc[-1]
                if pd.notna(val):
                    features.append(self._normalize(val, param))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

        # 2. 气象参数 (7维)
        if self.weather_data is not None and "time" in self.weather_data.columns:
            wx_mask = (self.weather_data["time"] >= time_step - timedelta(hours=1)) & \
                      (self.weather_data["time"] <= time_step + timedelta(hours=1))
            wx_matched = self.weather_data[wx_mask]
        else:
            wx_matched = pd.DataFrame()

        for param in WEATHER_PARAMS:
            if len(wx_matched) > 0 and param in wx_matched.columns:
                val = wx_matched[param].iloc[-1]
                features.append(float(val) if pd.notna(val) else 0.0)
            else:
                features.append(0.0)

        # 3. 时间编码 (4维)
        features.extend(self._time_encoding(time_step))

        # 4. 遥感特征 (3维)
        month_key = (station_id, time_step.strftime("%Y-%m"))
        rs = self.rs_data.get(month_key, {})
        features.append(float(rs.get("ndci", 0) or 0))
        features.append(float(rs.get("fai", 0) or 0))
        features.append(float(rs.get("lst", 0) or 0))

        return features

    def _get_bloom_label(self, chla, algae_density):
        """
        蓝藻水华风险分类
        0: 无风险 (Chl-a < 10, 藻密度 < 1000)
        1: 轻度 (10 <= Chl-a < 26, 1000 <= 藻密度 < 5000)
        2: 中度 (26 <= Chl-a < 64, 5000 <= 藻密度 < 20000)
        3: 重度 (Chl-a >= 64, 藻密度 >= 20000)
        """
        if chla >= 64 or algae_density >= 20000:
            return 3
        elif chla >= 26 or algae_density >= 5000:
            return 2
        elif chla >= 10 or algae_density >= 1000:
            return 1
        else:
            return 0

    def __len__(self):
        min_len = self.history_steps + self.predict_steps * 6
        return max(0, len(self.time_index) - min_len)

    def __getitem__(self, idx):
        # 历史时间窗口
        history_times = self.time_index[idx:idx + self.history_steps]

        # 未来预测时间点（每天1个，共 predict_steps 天）
        predict_start = self.time_index[idx + self.history_steps]
        predict_times = [predict_start + timedelta(days=d) for d in range(self.predict_steps)]

        # 构建节点特征 (num_nodes, history_steps, feature_dim)
        x = np.zeros((self.num_nodes, self.history_steps, FEATURE_DIM), dtype=np.float32)
        for t_idx, t in enumerate(history_times):
            for n_idx in range(self.num_nodes):
                x[n_idx, t_idx, :] = self._get_features_at_time(t, n_idx)

        # 构建标签
        # 水质回归标签 (num_nodes, predict_steps, 11)
        y_wq = np.zeros((self.num_nodes, self.predict_steps, len(WATER_QUALITY_PARAMS)), dtype=np.float32)
        # 蓝藻分类标签 (num_nodes,) — 取预测窗口内最高风险
        y_bloom = np.zeros(self.num_nodes, dtype=np.int64)

        for n_idx in range(self.num_nodes):
            max_bloom = 0
            for p_idx, pt in enumerate(predict_times):
                feats = self._get_features_at_time(pt, n_idx)
                y_wq[n_idx, p_idx, :] = feats[:len(WATER_QUALITY_PARAMS)]

                # 蓝藻分类 (Chl-a = index 9, 藻密度 = index 10)
                chla = feats[9] if len(feats) > 9 else 0
                algae = feats[10] if len(feats) > 10 else 0
                bloom = self._get_bloom_label(chla, algae)
                max_bloom = max(max_bloom, bloom)

            y_bloom[n_idx] = max_bloom

        return {
            "x": torch.FloatTensor(x),
            "y_wq": torch.FloatTensor(y_wq),
            "y_bloom": torch.LongTensor(y_bloom),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    x = torch.stack([item["x"] for item in batch])           # (B, N, T, F)
    y_wq = torch.stack([item["y_wq"] for item in batch])     # (B, N, P, 11)
    y_bloom = torch.stack([item["y_bloom"] for item in batch])  # (B, N)
    return {"x": x, "y_wq": y_wq, "y_bloom": y_bloom}
