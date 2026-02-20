"""
V2 PyTorch Dataset / DataLoader
预计算所有特征到 numpy 数组，__getitem__ 仅做数组切片，极快
"""

import json
from datetime import timedelta
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

NUM_WQ = len(WATER_QUALITY_PARAMS)
NUM_WX = len(WEATHER_PARAMS)
NUM_TIME = 6  # hour sin/cos, day_of_year sin/cos, month sin/cos
NUM_RS = len(REMOTE_SENSING_PARAMS)


class TaihuDataset(Dataset):
    """
    V2 太湖水质时空图数据集 (预计算版)

    初始化时将所有数据预计算为:
      - feature_tensor: (T, N, F)  全时间步 × 全节点 × 特征
      - wq_tensor: (T, N, 11)     水质参数原始值 (用于标签)

    __getitem__ 仅做 numpy 切片，速度极快
    """

    def __init__(self, data_dir="data", graph_dir="data/graph",
                 history_steps=42, predict_steps=14,
                 start_date=None, end_date=None, mode="train"):
        self.history_steps = history_steps
        self.predict_steps = predict_steps
        self.mode = mode
        self.data_dir = Path(data_dir)

        # 加载图结构
        self.graph = load_graph(graph_dir)
        self.num_nodes = self.graph["num_nodes"]
        self.node_ids = self.graph["node_ids"]

        # 加载原始数据
        self.wq_df = self._load_water_quality(start_date, end_date)
        self.wx_df = self._load_weather(start_date, end_date)
        self.rs_data = self._load_remote_sensing()

        # 构建时间索引
        self._build_time_index()

        # 归一化
        if self.mode == "train":
            self._compute_normalization()
        else:
            self._load_normalization()

        # 核心: 预计算全部特征为 numpy 数组
        self._precompute_tensors()

        logger.info(f"Dataset [{mode}]: {len(self)} 样本, "
                     f"{self.num_nodes} 节点, "
                     f"feature_tensor={self.feature_tensor.shape}, "
                     f"history={history_steps}, predict={predict_steps}")

    # ----------------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------------

    def _load_water_quality(self, start_date, end_date):
        processed_file = self.data_dir / "processed" / "water_quality.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file, parse_dates=["time"])
            if start_date:
                df = df[df["time"] >= start_date]
            if end_date:
                df = df[df["time"] <= end_date]
            return df

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
        processed_file = self.data_dir / "processed" / "weather.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file, parse_dates=["time"])
            if start_date:
                df = df[df["time"] >= start_date]
            if end_date:
                df = df[df["time"] <= end_date]
            return df

        history_file = self.data_dir / "raw" / "weather_history_all.csv"
        if history_file.exists():
            df = pd.read_csv(history_file, parse_dates=["time"])
            if start_date:
                df = df[df["time"] >= start_date]
            if end_date:
                df = df[df["time"] <= end_date]
            return df

        logger.warning("未找到气象数据，使用合成数据")
        return self._generate_synthetic_weather(start_date, end_date)

    def _load_remote_sensing(self):
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

    # ----------------------------------------------------------------
    # 合成数据 (无真实数据时)
    # ----------------------------------------------------------------

    def _generate_synthetic_wq(self, start_date, end_date):
        logger.info("生成合成水质数据...")
        start = pd.Timestamp(start_date or "2016-01-01")
        end = pd.Timestamp(end_date or "2025-12-31")
        times = pd.date_range(start, end, freq="4h")
        rng = np.random.RandomState(42)

        n_times = len(times)
        n_nodes = len(self.node_ids)
        total = n_times * n_nodes

        # 向量化生成
        months = np.tile(times.month.values, n_nodes)
        season = np.sin((months - 3) * np.pi / 6)

        records = {
            "station_id": np.repeat(self.node_ids, n_times),
            "time": np.tile(times, n_nodes),
            "water_temp": 15 + 10 * season + rng.normal(0, 2, total),
            "ph": 7.5 + rng.normal(0, 0.5, total),
            "do": 8.0 - 2 * season + rng.normal(0, 1, total),
            "conductivity": 400 + rng.normal(0, 50, total),
            "turbidity": 10 + 5 * np.abs(season) + rng.normal(0, 3, total),
            "codmn": 4.0 + 2 * season + rng.normal(0, 1, total),
            "nh3n": 0.5 + 0.3 * season + rng.normal(0, 0.15, total),
            "tp": 0.08 + 0.04 * season + rng.normal(0, 0.02, total),
            "tn": 2.0 + 1.0 * season + rng.normal(0, 0.5, total),
            "chla": np.maximum(5 + 30 * np.maximum(season, 0) + rng.normal(0, 5, total), 0),
            "algae_density": np.maximum(500 + 3000 * np.maximum(season, 0) + rng.normal(0, 300, total), 0),
        }
        return pd.DataFrame(records)

    def _generate_synthetic_weather(self, start_date, end_date):
        logger.info("生成 V2 合成气象数据...")
        start = pd.Timestamp(start_date or "2016-01-01")
        end = pd.Timestamp(end_date or "2025-12-31")
        times = pd.date_range(start, end, freq="4h")  # 与水质对齐
        rng = np.random.RandomState(123)
        n = len(times)

        months = times.month.values
        season = np.sin((months - 3) * np.pi / 6)
        temp = 15 + 15 * season + rng.normal(0, 3, n)
        humidity = np.clip(65 + 15 * season + rng.normal(0, 10, n), 0, 100)
        solar = np.maximum(200 + 300 * season + rng.normal(0, 50, n), 0)

        records = {
            "time": times,
            "temperature": temp,
            "humidity": humidity,
            "dewpoint": temp - (100 - humidity) / 5.0 + rng.normal(0, 1, n),
            "precipitation": np.maximum(rng.exponential(1.0, n) * (0.5 + 0.5 * season), 0),
            "rain": np.maximum(rng.exponential(0.5, n) * (0.5 + 0.5 * season), 0),
            "wind_speed": 3.0 + rng.exponential(2.0, n),
            "wind_direction": rng.uniform(0, 360, n),
            "wind_gusts": 5.0 + rng.exponential(4.0, n),
            "solar_radiation": solar,
            "direct_radiation": solar * 0.6 + rng.normal(0, 20, n),
            "diffuse_radiation": solar * 0.4 + rng.normal(0, 15, n),
            "pressure": 1013 + rng.normal(0, 5, n),
            "cloud_cover": np.clip(50 + 20 * season + rng.normal(0, 20, n), 0, 100),
            "evapotranspiration": np.maximum(0.1 + 0.3 * np.maximum(season, 0) + rng.normal(0, 0.05, n), 0),
            "soil_temperature": temp - 2 + rng.normal(0, 1, n),
        }
        return pd.DataFrame(records)

    # ----------------------------------------------------------------
    # 时间索引 & 归一化
    # ----------------------------------------------------------------

    def _build_time_index(self):
        if self.wq_df is None or len(self.wq_df) == 0:
            self.time_index = pd.DatetimeIndex([])
            return

        if "time" in self.wq_df.columns:
            min_time = self.wq_df["time"].min()
            max_time = self.wq_df["time"].max()
            self.time_index = pd.date_range(min_time, max_time, freq="4h")
        else:
            self.time_index = pd.DatetimeIndex([])

        min_len = self.history_steps + self.predict_steps * 6
        if len(self.time_index) < min_len:
            logger.warning(f"时间序列长度不足: {len(self.time_index)} < {min_len}")

    def _compute_normalization(self):
        self.norm_params = {}
        for param in WATER_QUALITY_PARAMS:
            if param in self.wq_df.columns:
                vals = self.wq_df[param].dropna()
                self.norm_params[param] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if vals.std() > 0 else 1.0
                }

        norm_file = self.data_dir / "processed" / "norm_params.json"
        norm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_file, "w") as f:
            json.dump(self.norm_params, f, indent=2)

    def _load_normalization(self):
        norm_file = self.data_dir / "processed" / "norm_params.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                self.norm_params = json.load(f)
        else:
            self._compute_normalization()

    # ----------------------------------------------------------------
    # 核心: 预计算特征张量
    # ----------------------------------------------------------------

    def _precompute_tensors(self):
        """
        一次性将所有数据预计算为 numpy 数组:
          feature_tensor: (T, N, F) — 所有时间步的完整特征
          wq_raw_tensor:  (T, N, 11) — 水质原始值 (用于标签和蓝藻分类)
        """
        T = len(self.time_index)
        N = self.num_nodes
        F = FEATURE_DIM

        if T == 0:
            self.feature_tensor = np.zeros((0, N, F), dtype=np.float32)
            self.wq_raw_tensor = np.zeros((0, N, NUM_WQ), dtype=np.float32)
            return

        logger.info(f"预计算特征张量: T={T}, N={N}, F={F}...")

        feature_tensor = np.zeros((T, N, F), dtype=np.float32)
        wq_raw_tensor = np.zeros((T, N, NUM_WQ), dtype=np.float32)

        # 1. 构建水质查找表: {(station_id, time_idx) -> wq_values}
        wq_lookup = self._build_wq_lookup()

        # 2. 构建气象查找表: {time_idx -> wx_values}
        wx_lookup = self._build_wx_lookup()

        # 3. 填充特征张量
        for t_idx, t in enumerate(self.time_index):
            # 时间编码 (对所有节点相同)
            hour = t.hour
            doy = t.timetuple().tm_yday
            month = t.month
            time_enc = np.array([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
            ], dtype=np.float32)

            # 气象 (对所有节点相同，取最近网格点均值)
            wx_vals = wx_lookup.get(t_idx, np.zeros(NUM_WX, dtype=np.float32))

            month_str = t.strftime("%Y-%m")

            for n_idx in range(N):
                sid = self.node_ids[n_idx]

                # 水质 (11维, 归一化)
                wq_raw = wq_lookup.get((sid, t_idx), np.zeros(NUM_WQ, dtype=np.float32))
                wq_raw_tensor[t_idx, n_idx] = wq_raw

                wq_norm = np.zeros(NUM_WQ, dtype=np.float32)
                for i, param in enumerate(WATER_QUALITY_PARAMS):
                    if param in self.norm_params:
                        p = self.norm_params[param]
                        wq_norm[i] = (wq_raw[i] - p["mean"]) / p["std"]
                    else:
                        wq_norm[i] = wq_raw[i]

                # 遥感 (3维)
                rs = self.rs_data.get((sid, month_str), {})
                rs_vals = np.array([
                    float(rs.get("ndci", 0) or 0),
                    float(rs.get("fai", 0) or 0),
                    float(rs.get("lst", 0) or 0),
                ], dtype=np.float32)

                # 拼接: [wq(11) | wx(15) | time(6) | rs(3)] = 35
                feature_tensor[t_idx, n_idx] = np.concatenate([
                    wq_norm, wx_vals, time_enc, rs_vals
                ])

            if (t_idx + 1) % 5000 == 0:
                logger.info(f"  预计算进度: {t_idx + 1}/{T}")

        self.feature_tensor = feature_tensor
        self.wq_raw_tensor = wq_raw_tensor
        logger.info(f"预计算完成: feature={feature_tensor.shape}, "
                     f"内存={feature_tensor.nbytes / 1e6:.1f} MB")

    def _build_wq_lookup(self):
        """构建水质数据查找表，使用时间索引对齐"""
        lookup = {}
        if self.wq_df is None or len(self.wq_df) == 0:
            return lookup

        if "time" not in self.wq_df.columns or "station_id" not in self.wq_df.columns:
            return lookup

        # 将水质时间对齐到最近的 4h 时间步
        wq = self.wq_df.copy()
        wq["time_aligned"] = wq["time"].dt.round("4h")

        # 转为 time_index 查找
        time_to_idx = {t: i for i, t in enumerate(self.time_index)}

        for sid in self.node_ids:
            station_data = wq[wq["station_id"] == sid]
            for _, row in station_data.iterrows():
                t_aligned = row["time_aligned"]
                t_idx = time_to_idx.get(t_aligned)
                if t_idx is None:
                    continue
                vals = np.array([
                    float(row[p]) if pd.notna(row.get(p)) else 0.0
                    for p in WATER_QUALITY_PARAMS
                ], dtype=np.float32)
                lookup[(sid, t_idx)] = vals

        return lookup

    def _build_wx_lookup(self):
        """构建气象数据查找表"""
        lookup = {}
        if self.wx_df is None or len(self.wx_df) == 0:
            return lookup

        if "time" not in self.wx_df.columns:
            return lookup

        wx = self.wx_df.copy()
        wx["time_aligned"] = wx["time"].dt.round("4h")

        time_to_idx = {t: i for i, t in enumerate(self.time_index)}

        # 按时间聚合（多个网格点取均值）
        wx_params = [p for p in WEATHER_PARAMS if p in wx.columns]
        missing_params = [p for p in WEATHER_PARAMS if p not in wx.columns]

        if wx_params:
            grouped = wx.groupby("time_aligned")[wx_params].mean()
            for t, row in grouped.iterrows():
                t_idx = time_to_idx.get(t)
                if t_idx is None:
                    continue
                vals = np.zeros(NUM_WX, dtype=np.float32)
                for i, param in enumerate(WEATHER_PARAMS):
                    if param in row.index and pd.notna(row[param]):
                        vals[i] = float(row[param])
                lookup[t_idx] = vals

        return lookup

    # ----------------------------------------------------------------
    # 蓝藻分类
    # ----------------------------------------------------------------

    @staticmethod
    def _get_bloom_label(chla, algae_density):
        if chla >= 64 or algae_density >= 20000:
            return 3
        elif chla >= 26 or algae_density >= 5000:
            return 2
        elif chla >= 10 or algae_density >= 1000:
            return 1
        else:
            return 0

    # ----------------------------------------------------------------
    # Dataset 接口
    # ----------------------------------------------------------------

    def __len__(self):
        min_len = self.history_steps + self.predict_steps * 6
        return max(0, len(self.time_index) - min_len)

    def __getitem__(self, idx):
        # 历史窗口: [idx, idx + history_steps)
        x = self.feature_tensor[idx:idx + self.history_steps]  # (H, N, F)
        x = x.transpose(1, 0, 2)  # (N, H, F)

        # 未来预测时间点 (每天1个, 共 predict_steps 天)
        pred_start = idx + self.history_steps
        pred_indices = []
        for d in range(self.predict_steps):
            # 每天 = 6 个 4h 步
            target_idx = pred_start + d * 6
            if target_idx < len(self.time_index):
                pred_indices.append(target_idx)
            else:
                pred_indices.append(len(self.time_index) - 1)

        # 水质标签 (N, P, 11) — 归一化值
        y_wq = np.stack([
            self.feature_tensor[pi, :, :NUM_WQ]  # 取前 11 维 (归一化水质)
            for pi in pred_indices
        ], axis=1)  # (N, P, 11)

        # 蓝藻分类标签 (N,) — 取预测窗口最高风险
        y_bloom = np.zeros(self.num_nodes, dtype=np.int64)
        for n in range(self.num_nodes):
            max_bloom = 0
            for pi in pred_indices:
                raw = self.wq_raw_tensor[pi, n]
                chla = raw[9]  # chla index
                algae = raw[10]  # algae_density index
                bloom = self._get_bloom_label(chla, algae)
                max_bloom = max(max_bloom, bloom)
            y_bloom[n] = max_bloom

        return {
            "x": torch.from_numpy(x.copy()),
            "y_wq": torch.from_numpy(y_wq.copy()),
            "y_bloom": torch.from_numpy(y_bloom.copy()),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    x = torch.stack([item["x"] for item in batch])
    y_wq = torch.stack([item["y_wq"] for item in batch])
    y_bloom = torch.stack([item["y_bloom"] for item in batch])
    return {"x": x, "y_wq": y_wq, "y_bloom": y_bloom}
