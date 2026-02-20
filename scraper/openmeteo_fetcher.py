"""
Open-Meteo Historical Weather API 数据获取
免费，无需 API Key
用于获取长期历史气象数据（训练用）
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 所需气象参数
HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "surface_pressure",
]

# 太湖流域气象观测点
WEATHER_LOCATIONS = [
    {"name": "太湖中心", "lat": 31.23, "lon": 120.14},
    {"name": "无锡", "lat": 31.49, "lon": 120.31},
    {"name": "苏州", "lat": 31.30, "lon": 120.62},
    {"name": "湖州", "lat": 30.87, "lon": 120.09},
    {"name": "宜兴", "lat": 31.36, "lon": 119.82},
]

MAX_RETRIES = 3


class OpenMeteoFetcher:
    """Open-Meteo 历史气象数据获取器"""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_history(self, lat, lon, start_date, end_date, location_name=""):
        """获取指定位置的历史气象数据"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_PARAMS),
            "timezone": "Asia/Shanghai",
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"获取 {location_name}({lat},{lon}) "
                    f"{start_date} ~ {end_date} 气象数据 (第{attempt}次)"
                )
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                if "hourly" not in data:
                    logger.warning(f"响应中无 hourly 数据: {data.get('reason', 'unknown')}")
                    return None

                # 转为 DataFrame
                hourly = data["hourly"]
                df = pd.DataFrame({
                    "time": pd.to_datetime(hourly["time"]),
                    "temperature": hourly.get("temperature_2m"),
                    "humidity": hourly.get("relative_humidity_2m"),
                    "precipitation": hourly.get("precipitation"),
                    "wind_speed": hourly.get("wind_speed_10m"),
                    "wind_direction": hourly.get("wind_direction_10m"),
                    "solar_radiation": hourly.get("shortwave_radiation"),
                    "pressure": hourly.get("surface_pressure"),
                })

                df["lat"] = lat
                df["lon"] = lon
                df["location_name"] = location_name

                logger.info(f"成功获取 {len(df)} 条记录")
                return df

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait_time = 30 * attempt
                    logger.warning(f"请求过于频繁，等待 {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP 错误: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(10)
            except Exception as e:
                logger.error(f"请求失败: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(10)

        return None

    def fetch_all_locations(self, start_date, end_date):
        """获取所有气象站点的历史数据"""
        all_dfs = []

        for loc in WEATHER_LOCATIONS:
            df = self.fetch_history(
                lat=loc["lat"],
                lon=loc["lon"],
                start_date=start_date,
                end_date=end_date,
                location_name=loc["name"]
            )
            if df is not None:
                all_dfs.append(df)
            # Open-Meteo 建议请求间隔
            time.sleep(2)

        if not all_dfs:
            logger.error("未获取到任何气象数据")
            return None

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"共获取 {len(combined)} 条气象记录，"
                     f"覆盖 {len(all_dfs)} 个站点")
        return combined

    def save_history_csv(self, df, filename="weather_history.csv"):
        """将历史数据保存为 CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"历史气象数据已保存: {filepath}")
        return filepath

    def fetch_and_save(self, start_date="2021-06-01", end_date="2025-12-31"):
        """一站式获取并保存所有历史气象数据"""
        df = self.fetch_all_locations(start_date, end_date)
        if df is not None:
            # 按年分段保存，避免单文件过大
            for year in df["time"].dt.year.unique():
                year_df = df[df["time"].dt.year == year]
                filename = f"weather_history_{year}.csv"
                self.save_history_csv(year_df, filename)

            # 同时保存一份完整文件
            self.save_history_csv(df, "weather_history_all.csv")
        return df


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Open-Meteo 历史气象数据获取")
    parser.add_argument("--start", default="2021-06-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data/raw", help="数据存储目录")
    args = parser.parse_args()

    fetcher = OpenMeteoFetcher(data_dir=args.data_dir)
    df = fetcher.fetch_and_save(start_date=args.start, end_date=args.end)
    if df is not None:
        print(f"完成，共 {len(df)} 条记录")


if __name__ == "__main__":
    main()
