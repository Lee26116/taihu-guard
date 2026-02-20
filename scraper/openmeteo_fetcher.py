"""
Open-Meteo Historical Weather API 数据获取 (V2 增强版)
免费，无需 API Key
用于获取长期历史气象数据（训练用）

V2 升级:
  - 10 年数据 (2016-2025)
  - 20 个网格点覆盖全流域
  - 15 个气象参数（含露点、土壤温度、蒸发量等）
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# V2 扩展气象参数 (15个)
HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "surface_pressure",
    "cloud_cover",
    "et0_fao_evapotranspiration",
    "soil_temperature_0_to_7cm",
]

# V2 参数名映射 (API 名 → 内部名)
PARAM_MAPPING = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "dewpoint_2m": "dewpoint",
    "precipitation": "precipitation",
    "rain": "rain",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
    "shortwave_radiation": "solar_radiation",
    "direct_radiation": "direct_radiation",
    "diffuse_radiation": "diffuse_radiation",
    "surface_pressure": "pressure",
    "cloud_cover": "cloud_cover",
    "et0_fao_evapotranspiration": "evapotranspiration",
    "soil_temperature_0_to_7cm": "soil_temperature",
}

# V2: 20 个网格点覆盖整个太湖流域 (约 0.15° 间隔)
# 范围: lat 30.85~31.55, lon 119.80~120.65
WEATHER_LOCATIONS = [
    # 第一行 (北)
    {"name": "G01_宜兴北", "lat": 31.55, "lon": 119.80},
    {"name": "G02_常州南", "lat": 31.55, "lon": 120.05},
    {"name": "G03_无锡西", "lat": 31.55, "lon": 120.30},
    {"name": "G04_无锡东", "lat": 31.55, "lon": 120.55},
    # 第二行
    {"name": "G05_宜兴", "lat": 31.35, "lon": 119.80},
    {"name": "G06_竺山湾", "lat": 31.35, "lon": 120.05},
    {"name": "G07_梅梁湾", "lat": 31.35, "lon": 120.30},
    {"name": "G08_贡湖", "lat": 31.35, "lon": 120.55},
    # 第三行 (湖心)
    {"name": "G09_西岸", "lat": 31.15, "lon": 119.95},
    {"name": "G10_湖心西", "lat": 31.15, "lon": 120.15},
    {"name": "G11_湖心东", "lat": 31.15, "lon": 120.35},
    {"name": "G12_东太湖", "lat": 31.15, "lon": 120.55},
    # 第四行
    {"name": "G13_南岸西", "lat": 30.95, "lon": 119.95},
    {"name": "G14_南岸中", "lat": 30.95, "lon": 120.15},
    {"name": "G15_南岸东", "lat": 30.95, "lon": 120.35},
    {"name": "G16_吴江", "lat": 30.95, "lon": 120.55},
    # 补充点 (重要区域加密)
    {"name": "G17_太湖中心", "lat": 31.23, "lon": 120.14},
    {"name": "G18_苏州", "lat": 31.30, "lon": 120.62},
    {"name": "G19_湖州", "lat": 30.87, "lon": 120.09},
    {"name": "G20_太浦河口", "lat": 30.96, "lon": 120.58},
]

MAX_RETRIES = 3


class OpenMeteoFetcher:
    """Open-Meteo 历史气象数据获取器 (V2 增强版)"""

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
                    f"{start_date} ~ {end_date} (第{attempt}次)"
                )
                resp = requests.get(BASE_URL, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                if "hourly" not in data:
                    logger.warning(f"响应中无 hourly 数据: {data.get('reason', 'unknown')}")
                    return None

                # 转为 DataFrame
                hourly = data["hourly"]
                row = {"time": pd.to_datetime(hourly["time"])}

                for api_name, internal_name in PARAM_MAPPING.items():
                    row[internal_name] = hourly.get(api_name)

                df = pd.DataFrame(row)
                df["lat"] = lat
                df["lon"] = lon
                df["location_name"] = location_name

                logger.info(f"成功获取 {len(df)} 条记录, {len(PARAM_MAPPING)} 个参数")
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
        """获取所有气象网格点的历史数据"""
        all_dfs = []
        total = len(WEATHER_LOCATIONS)

        for i, loc in enumerate(WEATHER_LOCATIONS):
            logger.info(f"[{i+1}/{total}] {loc['name']}")
            df = self.fetch_history(
                lat=loc["lat"],
                lon=loc["lon"],
                start_date=start_date,
                end_date=end_date,
                location_name=loc["name"]
            )
            if df is not None:
                all_dfs.append(df)
            # Open-Meteo 限流
            time.sleep(3)

        if not all_dfs:
            logger.error("未获取到任何气象数据")
            return None

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"共获取 {len(combined):,} 条气象记录，"
                     f"覆盖 {len(all_dfs)} 个网格点, "
                     f"{len(PARAM_MAPPING)} 个参数")
        return combined

    def save_history_csv(self, df, filename="weather_history.csv"):
        """将历史数据保存为 CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        size_mb = filepath.stat().st_size / 1e6
        logger.info(f"已保存: {filepath} ({size_mb:.1f} MB)")
        return filepath

    def fetch_and_save(self, start_date="2016-01-01", end_date="2025-12-31"):
        """一站式获取并保存所有历史气象数据"""
        # Open-Meteo 对单次请求时间跨度有限制，按 2 年分段
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_dfs = []
        current_start = start
        chunk_years = 2

        while current_start < end:
            chunk_end = min(
                current_start.replace(year=current_start.year + chunk_years) ,
                end
            )
            cs = current_start.strftime("%Y-%m-%d")
            ce = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"\n{'='*50}")
            logger.info(f"下载时段: {cs} ~ {ce}")
            logger.info(f"{'='*50}")

            df = self.fetch_all_locations(cs, ce)
            if df is not None:
                all_dfs.append(df)
                # 分段保存
                seg_name = f"weather_{current_start.year}_{chunk_end.year}.csv"
                self.save_history_csv(df, seg_name)

            current_start = chunk_end

        if not all_dfs:
            logger.error("全部下载失败")
            return None

        combined = pd.concat(all_dfs, ignore_index=True)

        # 按年保存
        for year in sorted(combined["time"].dt.year.unique()):
            year_df = combined[combined["time"].dt.year == year]
            self.save_history_csv(year_df, f"weather_history_{year}.csv")

        # 保存完整文件
        self.save_history_csv(combined, "weather_history_all.csv")

        logger.info(f"\n{'='*50}")
        logger.info(f"全部完成: {len(combined):,} 条记录")
        logger.info(f"时间范围: {combined['time'].min()} ~ {combined['time'].max()}")
        logger.info(f"网格点数: {combined['location_name'].nunique()}")
        logger.info(f"参数数量: {len(PARAM_MAPPING)}")
        logger.info(f"{'='*50}")

        return combined


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Open-Meteo 历史气象数据获取 (V2)")
    parser.add_argument("--start", default="2016-01-01", help="开始日期")
    parser.add_argument("--end", default="2025-12-31", help="结束日期")
    parser.add_argument("--data-dir", default="data/raw", help="数据存储目录")
    args = parser.parse_args()

    fetcher = OpenMeteoFetcher(data_dir=args.data_dir)
    df = fetcher.fetch_and_save(start_date=args.start, end_date=args.end)
    if df is not None:
        print(f"完成，共 {len(df):,} 条记录")


if __name__ == "__main__":
    main()
