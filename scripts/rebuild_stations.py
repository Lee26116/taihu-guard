"""
一次性脚本: 从 CNEMC API 获取太湖流域全部断面，重建 data/stations.json
使用高德地理编码 API 获取缺失坐标
"""

import json
import re
import time
from pathlib import Path

import requests
from loguru import logger

# ---------- 配置 ----------
API_URL = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
TAIHU_RIVER_ID = "1200000000"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

# 保留这些省份的站点 (排除上海等远距离站)
KEEP_PROVINCES = {"江苏省", "浙江省", "安徽省"}

# 太湖流域大致经纬度范围 (用于校验地理编码结果)
TAIHU_BBOX = {
    "lat_min": 29.5,
    "lat_max": 33.0,
    "lon_min": 118.5,
    "lon_max": 122.0,
}

# 太湖中心坐标 (地理编码失败时的回退)
TAIHU_CENTER = (31.2258, 120.1375)

# 高德 API 限流间隔 (秒)
GEOCODE_DELAY = 0.25

# CNEMC PARAM_KEYS (与 scraper 一致)
PARAM_KEYS = [
    "water_temp", "ph", "do", "conductivity", "turbidity",
    "codmn", "nh3n", "tp", "tn", "chla", "algae_density",
]


def fetch_cnemc_stations():
    """调用 CNEMC API，返回太湖流域全部断面记录"""
    params = {
        "action": "getRealDatas",
        "AreaID": "",
        "RiverID": TAIHU_RIVER_ID,
        "MNName": "",
        "PageIndex": 1,
        "PageSize": 500,
    }
    resp = requests.post(API_URL, data=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    tbody = data.get("tbody", [])
    logger.info(f"CNEMC API 返回 {len(tbody)} 条记录")

    stations = []
    for row in tbody:
        if len(row) < 6:
            continue
        province = row[0]
        basin = row[1]
        name = row[2]

        # 检查 chla 是否有值
        chla_idx = 5 + PARAM_KEYS.index("chla")
        has_chla = (
            len(row) > chla_idx
            and row[chla_idx]
            and str(row[chla_idx]) not in ("--", "&nbsp;", "", "None")
        )

        stations.append({
            "name": name,
            "province": province,
            "basin": basin,
            "has_chla": has_chla,
        })

    return stations


def classify_station(name):
    """根据断面名称推断站点类型"""
    # 已知的太湖湖体断面名
    known_lake = {
        "兰山嘴", "拖山", "五里湖心", "胥湖心", "乌龟山南", "落蓬湾",
    }
    if name in known_lake:
        return "湖体"

    # 出湖相关
    outflow_keywords = ["望亭", "太浦", "瓜泾"]
    if any(k in name for k in outflow_keywords):
        return "出湖"

    # 入湖相关 — 含"港"但不含"桥"
    if "港" in name and "桥" not in name:
        return "入湖"

    # 其余为河道
    return "河道"


def _in_taihu_bbox(lat, lon):
    """检查坐标是否在太湖流域范围内"""
    return (TAIHU_BBOX["lat_min"] <= lat <= TAIHU_BBOX["lat_max"]
            and TAIHU_BBOX["lon_min"] <= lon <= TAIHU_BBOX["lon_max"])


def load_existing_stations(path):
    """加载现有 stations.json，建立 name -> coord 映射"""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["name"]: (s["lat"], s["lon"]) for s in data.get("stations", [])}


def _geocode_one(address, city, api_key):
    """单次高德地理编码调用，返回 (lat, lon) 或 None"""
    params = {
        "key": api_key,
        "address": address,
        "city": city,
    }
    try:
        resp = requests.get(AMAP_GEOCODE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            loc = data["geocodes"][0].get("location", "")
            if loc:
                lon, lat = loc.split(",")
                return float(lat), float(lon)
    except Exception as e:
        logger.warning(f"  地理编码请求异常: {e}")
    return None


def geocode_with_fallback(name, province, api_key):
    """
    多策略地理编码:
    1. 直接用站点名 + 省份
    2. 加 "太湖流域" 前缀
    3. 加省份名前缀
    只接受落在太湖流域 bbox 内的结果
    """
    city = province.replace("省", "").replace("市", "")

    strategies = [
        (name, city),
        (f"太湖流域{name}", city),
        (f"{city}{name}", ""),
    ]

    for addr, c in strategies:
        result = _geocode_one(addr, c, api_key)
        time.sleep(GEOCODE_DELAY)
        if result and _in_taihu_bbox(result[0], result[1]):
            return result

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重建 stations.json")
    parser.add_argument("--amap-key", required=True, help="高德地理编码 API Key")
    parser.add_argument("--existing", default="data/stations.json",
                        help="现有 stations.json 路径 (用于复用坐标)")
    parser.add_argument("--output", default="data/stations.json",
                        help="输出路径")
    args = parser.parse_args()

    # 1. 获取 CNEMC 站点列表
    logger.info("Step 1: 获取 CNEMC 站点列表...")
    cnemc_stations = fetch_cnemc_stations()

    # 2. 筛选省份
    filtered = [s for s in cnemc_stations if s["province"] in KEEP_PROVINCES]
    logger.info(f"Step 2: 筛选后保留 {len(filtered)} 个站点 "
                f"(排除上海等, 原始 {len(cnemc_stations)})")

    # 3. 加载现有坐标
    existing_coords = load_existing_stations(args.existing)
    logger.info(f"Step 3: 现有 stations.json 中有 {len(existing_coords)} 个站点坐标")

    # 4. 构建新的站点列表
    logger.info("Step 4: 构建新站点列表 (地理编码缺失坐标)...")
    new_stations = []
    geocoded_count = 0
    fallback_count = 0
    reused_count = 0

    for s in filtered:
        name = s["name"]
        province = s["province"]

        # 坐标来源
        if name in existing_coords:
            lat, lon = existing_coords[name]
            coord_source = "existing"
            reused_count += 1
        else:
            result = geocode_with_fallback(name, province, args.amap_key)
            if result:
                lat, lon = result
                coord_source = "amap_geocode"
                geocoded_count += 1
                logger.info(f"  编码成功: {name} -> ({lat:.4f}, {lon:.4f})")
            else:
                lat, lon = TAIHU_CENTER
                coord_source = "fallback"
                fallback_count += 1
                logger.warning(f"  编码失败: {name} -> 使用太湖中心坐标")

        station_type = classify_station(name)

        station = {
            "id": name,
            "name": name,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "province": province,
            "basin": s["basin"],
            "type": station_type,
            "has_chla": s["has_chla"],
            "coord_source": coord_source,
        }
        new_stations.append(station)

    # 5. 按省份 + 类型排序
    type_order = {"湖体": 0, "入湖": 1, "出湖": 2, "河道": 3}
    new_stations.sort(key=lambda x: (x["province"], type_order.get(x["type"], 9), x["name"]))

    # 6. 输出
    output = {
        "metadata": {
            "description": "太湖流域国控地表水水质自动监测站点",
            "source": "生态环境部地表水水质自动监测实时数据发布系统 + 高德地理编码",
            "coordinate_system": "WGS84",
            "last_updated": time.strftime("%Y-%m-%d"),
            "total_stations": len(new_stations),
        },
        "stations": new_stations,
        "weather_stations": [
            {"name": "太湖中心", "lat": 31.2258, "lon": 120.1375},
            {"name": "无锡", "lat": 31.49, "lon": 120.31},
            {"name": "苏州", "lat": 31.30, "lon": 120.62},
            {"name": "湖州", "lat": 30.87, "lon": 120.09},
            {"name": "宜兴", "lat": 31.36, "lon": 119.82},
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"输出: {output_path}")
    logger.info(f"总站点数: {len(new_stations)}")
    logger.info(f"  复用坐标: {reused_count}")
    logger.info(f"  高德编码: {geocoded_count}")
    logger.info(f"  回退坐标: {fallback_count}")
    logger.info(f"  有 Chl-a: {sum(1 for s in new_stations if s['has_chla'])}")
    logger.info(f"  类型分布:")
    for t in ["湖体", "入湖", "出湖", "河道"]:
        count = sum(1 for s in new_stations if s["type"] == t)
        logger.info(f"    {t}: {count}")


if __name__ == "__main__":
    main()
