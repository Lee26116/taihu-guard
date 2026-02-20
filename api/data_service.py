"""
数据查询服务
提供水质、气象、预测数据的查询接口
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


class DataService:
    """数据查询服务"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def has_data(self):
        """检查是否有可用数据"""
        prediction_file = self.data_dir / "predictions" / "latest.json"
        raw_dir = self.data_dir / "raw"
        return prediction_file.exists() or any(raw_dir.glob("**/*.json"))

    def get_latest_prediction(self):
        """获取最新预测结果"""
        prediction_file = self.data_dir / "predictions" / "latest.json"
        if not prediction_file.exists():
            return None

        try:
            with open(prediction_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查数据是否过期（超过 8 小时）
            pred_time = datetime.fromisoformat(data.get("prediction_time", "2000-01-01"))
            if datetime.now() - pred_time > timedelta(hours=8):
                logger.warning("预测数据已过期")

            return data
        except Exception as e:
            logger.error(f"读取预测数据失败: {e}")
            return None

    def get_latest_water_quality(self):
        """获取最新的实测水质数据"""
        raw_dir = self.data_dir / "raw"
        wq_files = sorted(raw_dir.glob("**/water_quality_*.json"), reverse=True)

        if not wq_files:
            return None

        try:
            with open(wq_files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取水质数据失败: {e}")
            return None

    def get_station_history(self, station_id, days=7):
        """获取指定站点的历史数据"""
        raw_dir = self.data_dir / "raw"
        cutoff = datetime.now() - timedelta(days=days)

        history = []
        wq_files = sorted(raw_dir.glob("**/water_quality_*.json"))

        for wq_file in wq_files:
            try:
                # 从文件名解析日期
                fname = wq_file.stem  # water_quality_YYYYMMDD_HHMM
                parts = fname.split("_")
                if len(parts) >= 3:
                    date_str = parts[2]
                    if len(date_str) == 8:
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        if file_date < cutoff:
                            continue

                with open(wq_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for record in data.get("records", []):
                    # 匹配站点 (按名称或 ID)
                    if (record.get("station_id") == station_id or
                        record.get("station_name", "").startswith(station_id)):
                        history.append({
                            "time": record.get("time", data.get("scrape_time", "")),
                            **{k: record.get(k) for k in [
                                "water_temp", "ph", "do", "conductivity",
                                "turbidity", "codmn", "nh3n", "tp", "tn",
                                "chla", "algae_density"
                            ]}
                        })
            except Exception:
                continue

        # 按时间排序
        history.sort(key=lambda x: x.get("time", ""))
        return history

    def get_latest_weather(self):
        """获取最新气象数据"""
        raw_dir = self.data_dir / "raw"
        wx_files = sorted(raw_dir.glob("**/weather_*.json"), reverse=True)

        if not wx_files:
            return None

        try:
            with open(wx_files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取气象数据失败: {e}")
            return None
