"""
Google Earth Engine 卫星遥感数据提取
仅在训练阶段使用，提取历史遥感特征
数据集: Sentinel-2, MODIS
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# GEE 导入延迟加载（可能未安装）
ee = None

# 太湖 ROI 范围
TAIHU_ROI = [119.88, 30.90, 120.60, 31.55]

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "water-protection-488007")


def _init_gee():
    """初始化 GEE"""
    global ee
    try:
        import ee as earth_engine
        ee = earth_engine
        ee.Initialize(project=GEE_PROJECT_ID)
        logger.info("Google Earth Engine 初始化成功")
        return True
    except Exception as e:
        logger.error(f"GEE 初始化失败: {e}")
        logger.info("请确保已安装 earthengine-api 并完成认证: earthengine authenticate")
        return False


class GEEExtractor:
    """GEE 卫星遥感特征提取器"""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.initialized = False

    def init(self):
        """延迟初始化 GEE"""
        if not self.initialized:
            self.initialized = _init_gee()
        return self.initialized

    def _get_taihu_roi(self):
        """获取太湖 ROI 几何对象"""
        return ee.Geometry.Rectangle(TAIHU_ROI)

    def _compute_ndci(self, image):
        """计算归一化差异叶绿素指数 NDCI = (B5-B4)/(B5+B4)"""
        ndci = image.normalizedDifference(["B5", "B4"]).rename("NDCI")
        return image.addBands(ndci)

    def _compute_fai(self, image):
        """
        计算浮游藻类指数 FAI
        FAI = R_NIR - R_RED - (R_SWIR - R_RED) * (λ_NIR - λ_RED) / (λ_SWIR - λ_RED)
        简化版: FAI ≈ B8 - B4 - (B11 - B4) * (832.8 - 664.6) / (1613.7 - 664.6)
        """
        fai = image.expression(
            "NIR - RED - (SWIR - RED) * 0.1771",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
                "SWIR": image.select("B11"),
            }
        ).rename("FAI")
        return image.addBands(fai)

    def extract_sentinel2_features(self, start_date, end_date, stations):
        """
        从 Sentinel-2 提取 NDCI 和 FAI
        按月聚合，提取每个站点周围 500m 范围的均值
        """
        if not self.init():
            return None

        roi = self._get_taihu_roi()

        # Sentinel-2 SR 影像集
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )

        logger.info(f"Sentinel-2 影像数量: {s2.size().getInfo()}")

        results = []

        # 按月聚合
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current < end:
            month_start = current.strftime("%Y-%m-%d")
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            month_end = next_month.strftime("%Y-%m-%d")

            try:
                monthly = s2.filterDate(month_start, month_end)
                count = monthly.size().getInfo()
                if count == 0:
                    current = next_month
                    continue

                # 对月度影像取中值合成
                composite = monthly.median()
                composite = self._compute_ndci(composite)
                composite = self._compute_fai(composite)

                # 为每个站点提取特征
                for station in stations:
                    point = ee.Geometry.Point([station["lon"], station["lat"]])
                    buffer = point.buffer(500)  # 500m 缓冲区

                    values = composite.select(["NDCI", "FAI"]).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=buffer,
                        scale=10,
                        maxPixels=1e8
                    ).getInfo()

                    results.append({
                        "station_id": station["id"],
                        "month": month_start[:7],
                        "ndci": values.get("NDCI"),
                        "fai": values.get("FAI"),
                        "image_count": count
                    })

                logger.info(f"完成 {month_start[:7]}: {count} 张影像")

            except Exception as e:
                logger.warning(f"处理 {month_start[:7]} 出错: {e}")

            current = next_month

        return pd.DataFrame(results) if results else None

    def extract_modis_lst(self, start_date, end_date, stations):
        """
        从 MODIS 提取地表温度 (LST)
        按周聚合，作为水温辅助特征
        """
        if not self.init():
            return None

        roi = self._get_taihu_roi()

        modis_lst = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .select("LST_Day_1km")
        )

        logger.info(f"MODIS LST 影像数量: {modis_lst.size().getInfo()}")

        results = []

        # 按周聚合
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        from datetime import timedelta
        current = start
        while current < end:
            week_end = min(current + timedelta(days=7), end)
            ws = current.strftime("%Y-%m-%d")
            we = week_end.strftime("%Y-%m-%d")

            try:
                weekly = modis_lst.filterDate(ws, we)
                count = weekly.size().getInfo()
                if count == 0:
                    current = week_end
                    continue

                # 周均值合成
                composite = weekly.mean()
                # MODIS LST 需要乘以缩放因子 0.02 转为开尔文再减 273.15 转摄氏度
                lst_celsius = composite.multiply(0.02).subtract(273.15)

                for station in stations:
                    point = ee.Geometry.Point([station["lon"], station["lat"]])
                    buffer = point.buffer(1000)  # 1km 缓冲区（MODIS 分辨率 1km）

                    values = lst_celsius.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=buffer,
                        scale=1000,
                        maxPixels=1e8
                    ).getInfo()

                    results.append({
                        "station_id": station["id"],
                        "week_start": ws,
                        "lst_celsius": values.get("LST_Day_1km"),
                        "image_count": count
                    })

            except Exception as e:
                logger.warning(f"处理 {ws} ~ {we} 出错: {e}")

            current = week_end

        return pd.DataFrame(results) if results else None

    def extract_and_save(self, start_date="2021-06-01", end_date="2025-12-31"):
        """一站式提取并保存所有遥感特征"""
        # 加载站点信息
        stations_file = Path("data/stations.json")
        if not stations_file.exists():
            logger.error("站点文件不存在: data/stations.json")
            return

        with open(stations_file, "r", encoding="utf-8") as f:
            stations_data = json.load(f)
        stations = stations_data["stations"]

        # 提取 Sentinel-2 NDCI/FAI
        logger.info("开始提取 Sentinel-2 NDCI/FAI...")
        s2_df = self.extract_sentinel2_features(start_date, end_date, stations)
        if s2_df is not None:
            filepath = self.data_dir / "sentinel2_features.csv"
            s2_df.to_csv(filepath, index=False)
            logger.info(f"Sentinel-2 特征已保存: {filepath}")

        # 提取 MODIS LST
        logger.info("开始提取 MODIS LST...")
        lst_df = self.extract_modis_lst(start_date, end_date, stations)
        if lst_df is not None:
            filepath = self.data_dir / "modis_lst.csv"
            lst_df.to_csv(filepath, index=False)
            logger.info(f"MODIS LST 已保存: {filepath}")


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="GEE 卫星遥感特征提取")
    parser.add_argument("--start", default="2021-06-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    extractor = GEEExtractor(data_dir=args.data_dir)
    extractor.extract_and_save(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
