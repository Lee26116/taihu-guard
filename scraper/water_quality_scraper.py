"""
水质监测数据获取
从生态环境部地表水水质自动监测实时数据发布系统获取太湖流域水质数据
直接调用后端 API，无需 Selenium
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

# API 端点
API_URL = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"

# 太湖流域 RiverID
TAIHU_RIVER_ID = "1200000000"

MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 水质参数顺序 (与 API 返回的 tbody 列顺序对应)
# 列: [省份, 流域, 断面名称, 监测时间, 水质类别, 水温, pH, DO, 电导率, 浊度, CODMn, NH3N, TP, TN, Chla, 藻密度]
PARAM_KEYS = ["water_temp", "ph", "do", "conductivity", "turbidity",
              "codmn", "nh3n", "tp", "tn", "chla", "algae_density"]


def _extract_value(html_str):
    """从 API 返回的 HTML span 中提取原始数值"""
    if not html_str or html_str == "--" or html_str == "&nbsp;":
        return None
    # 尝试从 title='原始值：xxx' 提取
    m = re.search(r'原始值：([\d.]+)', str(html_str))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # 尝试直接解析为数字
    try:
        return float(str(html_str).strip())
    except ValueError:
        return None


class WaterQualityScraper:
    """生态环境部水质数据获取 (HTTP API)"""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_data(self):
        """通过 POST API 获取太湖流域水质数据"""
        params = {
            "action": "getRealDatas",
            "AreaID": "",
            "RiverID": TAIHU_RIVER_ID,
            "MNName": "",
            "PageIndex": 1,
            "PageSize": 500,  # 足够获取所有太湖站点
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(API_URL, data=params, timeout=30, verify=False)
                resp.raise_for_status()
                data = resp.json()

                if data.get("result") and data["result"] != 0:
                    return data
                else:
                    logger.warning(f"API 返回无数据 (attempt {attempt})")

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (第{attempt}次)")
            except Exception as e:
                logger.error(f"请求失败 (第{attempt}次): {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        return None

    def _parse_records(self, data):
        """将 API 返回的 tbody 解析为结构化记录"""
        records = []
        tbody = data.get("tbody", [])

        for row in tbody:
            if len(row) < 6:
                continue

            record = {
                "province": row[0],
                "basin": row[1],
                "station_name": row[2],
                "time": row[3],
                "water_quality_level": int(row[4]) if row[4] and str(row[4]).isdigit() else None,
            }

            # 解析水质参数 (从第6列开始，索引5)
            for i, key in enumerate(PARAM_KEYS):
                idx = 5 + i
                if idx < len(row):
                    record[key] = _extract_value(row[idx])
                else:
                    record[key] = None

            if record["station_name"]:
                records.append(record)

        return records

    def scrape(self):
        """执行一次完整的水质数据获取"""
        now = datetime.now()
        logger.info(f"开始获取水质数据 - {now.strftime('%Y-%m-%d %H:%M')}")

        data = self._fetch_data()
        if not data:
            logger.error("未能获取到 API 数据")
            return []

        records = self._parse_records(data)
        if not records:
            logger.error("解析数据为空")
            return []

        # 添加获取时间戳
        for r in records:
            r["scrape_time"] = now.isoformat()

        # 保存数据
        self._save_data(records, now)
        logger.info(f"成功获取 {len(records)} 条水质记录")
        return records

    def _save_data(self, records, timestamp):
        """保存数据到 JSON 文件"""
        date_dir = self.data_dir / timestamp.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = f"water_quality_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        filepath = date_dir / filename

        if filepath.exists():
            logger.info(f"数据文件已存在，跳过: {filepath}")
            return filepath

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "scrape_time": timestamp.isoformat(),
                "basin": "太湖流域",
                "station_count": len(records),
                "records": records
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存: {filepath}")
        return filepath

    def get_latest_data(self):
        """获取最新一次的数据"""
        json_files = sorted(self.data_dir.glob("**/water_quality_*.json"), reverse=True)
        if not json_files:
            return None
        with open(json_files[0], "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="太湖水质数据获取")
    parser.add_argument("--data-dir", default="data/raw", help="数据存储目录")
    args = parser.parse_args()

    scraper = WaterQualityScraper(data_dir=args.data_dir)
    records = scraper.scrape()
    if records:
        print(f"成功获取 {len(records)} 条记录")
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()
