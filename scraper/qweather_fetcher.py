"""
和风天气 API 数据获取
免费版: 每日1000次调用限制
端点: devapi.qweather.com/v7/
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("QWEATHER_API_KEY", "")
QWEATHER_HOST = os.getenv("QWEATHER_HOST", "kc44ua927w.re.qweatherapi.com")
BASE_URL = f"https://{QWEATHER_HOST}/v7"

# 太湖流域气象站点
WEATHER_STATIONS = [
    {"name": "太湖中心", "lat": 31.2258, "lon": 120.1375},
    {"name": "无锡", "lat": 31.49, "lon": 120.31},
    {"name": "苏州", "lat": 31.30, "lon": 120.62},
    {"name": "湖州", "lat": 30.87, "lon": 120.09},
    {"name": "宜兴", "lat": 31.36, "lon": 119.82},
]

# 请求间隔（秒），避免频率限制
REQUEST_INTERVAL = 1.0
MAX_RETRIES = 3


class QWeatherFetcher:
    """和风天气数据获取器"""

    def __init__(self, api_key=None, data_dir="data/raw"):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            logger.warning("未配置和风天气 API Key，请检查 .env 文件")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._request_count = 0
        self._last_request_time = 0

    def _throttle(self):
        """请求限流"""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def _make_request(self, endpoint, params):
        """发送 API 请求（带重试）"""
        if not self.api_key:
            logger.error("未配置 API Key")
            return None

        params["key"] = self.api_key
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._throttle()
                resp = requests.get(url, params=params, timeout=15)
                data = resp.json()

                if data.get("code") == "200":
                    return data
                elif data.get("code") == "429":
                    logger.warning("API 调用次数超限，等待后重试")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"API 返回错误 code={data.get('code')}: {endpoint}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (第{attempt}次): {endpoint}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"请求失败 (第{attempt}次): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)

        return None

    def fetch_realtime(self, lat, lon):
        """获取实时天气"""
        location = f"{lon},{lat}"
        data = self._make_request("/weather/now", {"location": location})
        if data and "now" in data:
            return data["now"]
        return None

    def fetch_hourly_forecast(self, lat, lon):
        """获取24小时逐小时预报"""
        location = f"{lon},{lat}"
        data = self._make_request("/weather/24h", {"location": location})
        if data and "hourly" in data:
            return data["hourly"]
        return None

    def fetch_daily_forecast(self, lat, lon):
        """获取7天预报"""
        location = f"{lon},{lat}"
        data = self._make_request("/weather/7d", {"location": location})
        if data and "daily" in data:
            return data["daily"]
        return None

    def fetch_all_stations(self):
        """获取所有气象站点的实时和预报数据"""
        now = datetime.now()
        results = {
            "fetch_time": now.isoformat(),
            "stations": []
        }

        for station in WEATHER_STATIONS:
            logger.info(f"获取 {station['name']} 气象数据...")
            station_data = {
                "name": station["name"],
                "lat": station["lat"],
                "lon": station["lon"],
                "realtime": None,
                "hourly_forecast": None,
                "daily_forecast": None
            }

            # 实时天气
            realtime = self.fetch_realtime(station["lat"], station["lon"])
            if realtime:
                station_data["realtime"] = {
                    "temp": float(realtime.get("temp", 0)),
                    "humidity": float(realtime.get("humidity", 0)),
                    "precip": float(realtime.get("precip", 0)),
                    "wind_speed": float(realtime.get("windSpeed", 0)),
                    "wind_dir_360": float(realtime.get("wind360", 0)),
                    "pressure": float(realtime.get("pressure", 0)),
                    "vis": float(realtime.get("vis", 0)),
                    "cloud": float(realtime.get("cloud", 0)) if realtime.get("cloud") else None,
                    "obs_time": realtime.get("obsTime", "")
                }

            # 逐小时预报
            hourly = self.fetch_hourly_forecast(station["lat"], station["lon"])
            if hourly:
                station_data["hourly_forecast"] = [
                    {
                        "time": h.get("fxTime", ""),
                        "temp": float(h.get("temp", 0)),
                        "humidity": float(h.get("humidity", 0)),
                        "precip": float(h.get("precip", 0)),
                        "wind_speed": float(h.get("windSpeed", 0)),
                        "pressure": float(h.get("pressure", 0)),
                        "cloud": float(h.get("cloud", 0)) if h.get("cloud") else None
                    }
                    for h in hourly
                ]

            # 7天预报
            daily = self.fetch_daily_forecast(station["lat"], station["lon"])
            if daily:
                station_data["daily_forecast"] = [
                    {
                        "date": d.get("fxDate", ""),
                        "temp_max": float(d.get("tempMax", 0)),
                        "temp_min": float(d.get("tempMin", 0)),
                        "humidity": float(d.get("humidity", 0)),
                        "precip": float(d.get("precip", 0)),
                        "wind_speed": float(d.get("windSpeedDay", 0)),
                        "pressure": float(d.get("pressure", 0)),
                        "cloud": float(d.get("cloud", 0)) if d.get("cloud") else None,
                        "uv_index": float(d.get("uvIndex", 0))
                    }
                    for d in daily
                ]

            results["stations"].append(station_data)

        # 保存到文件
        self._save_data(results, now)
        logger.info(f"获取完成，共 {len(results['stations'])} 个站点，"
                     f"本次消耗 {self._request_count} 次 API 调用")
        return results

    def _save_data(self, data, timestamp):
        """保存气象数据"""
        date_dir = self.data_dir / timestamp.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = f"weather_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        filepath = date_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"气象数据已保存: {filepath}")
        return filepath


def main():
    """命令行入口"""
    fetcher = QWeatherFetcher()
    results = fetcher.fetch_all_stations()
    print(f"获取完成: {len(results.get('stations', []))} 个站点")


if __name__ == "__main__":
    main()
