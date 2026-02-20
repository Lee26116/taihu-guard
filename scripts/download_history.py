"""
一次性脚本: 下载历史数据
包含: Open-Meteo 历史气象数据 + GEE 遥感特征（可选）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def download_weather_history(start_date, end_date, data_dir):
    """下载历史气象数据 (Open-Meteo)"""
    logger.info("=" * 50)
    logger.info("下载历史气象数据 (Open-Meteo)")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info("=" * 50)

    from scraper.openmeteo_fetcher import OpenMeteoFetcher
    fetcher = OpenMeteoFetcher(data_dir=data_dir)
    df = fetcher.fetch_and_save(start_date=start_date, end_date=end_date)

    if df is not None:
        logger.info(f"气象数据下载完成: {len(df)} 条记录")
    else:
        logger.error("气象数据下载失败")


def download_gee_features(start_date, end_date, data_dir):
    """下载 GEE 卫星遥感特征（可选）"""
    logger.info("=" * 50)
    logger.info("下载 GEE 卫星遥感特征")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info("=" * 50)

    try:
        from scraper.gee_extractor import GEEExtractor
        extractor = GEEExtractor(data_dir=data_dir)
        extractor.extract_and_save(start_date=start_date, end_date=end_date)
        logger.info("GEE 特征提取完成")
    except ImportError:
        logger.warning("earthengine-api 未安装，跳过 GEE 特征提取")
        logger.info("安装: pip install earthengine-api")
    except Exception as e:
        logger.error(f"GEE 特征提取失败: {e}")
        logger.info("GEE 特征为可选项，不影响模型基本训练")


def main():
    parser = argparse.ArgumentParser(description="下载历史数据")
    parser.add_argument("--start", default="2021-06-01", help="开始日期")
    parser.add_argument("--end", default="2025-12-31", help="结束日期")
    parser.add_argument("--data-dir", default="data/raw", help="数据存储目录")
    parser.add_argument("--skip-gee", action="store_true", help="跳过 GEE 遥感数据")
    args = parser.parse_args()

    # 下载气象数据
    download_weather_history(args.start, args.end, args.data_dir)

    # 下载 GEE 数据（可选）
    if not args.skip_gee:
        download_gee_features(args.start, args.end, args.data_dir)
    else:
        logger.info("跳过 GEE 遥感数据下载")

    logger.info("\n历史数据下载完成!")
    logger.info("下一步: python scripts/build_graph.py")


if __name__ == "__main__":
    main()
