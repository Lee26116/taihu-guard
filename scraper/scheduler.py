"""
数据采集调度器
cron 每4小时调用一次: 爬取水质数据 + 获取气象数据 + 运行推理
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# 配置日志
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8"
)

PROJECT_ROOT = Path(__file__).parent.parent


def run_water_quality_scraper():
    """运行水质数据爬虫"""
    logger.info("=" * 60)
    logger.info("Step 1: 爬取水质监测数据")
    logger.info("=" * 60)

    try:
        from scraper.water_quality_scraper import WaterQualityScraper
        scraper = WaterQualityScraper(data_dir="data/raw")
        records = scraper.scrape()

        if records:
            logger.info(f"水质数据爬取成功: {len(records)} 条记录")
            return True
        else:
            logger.warning("水质数据爬取未获取到数据")
            return False

    except Exception as e:
        logger.error(f"水质数据爬取失败: {e}")
        return False


def run_weather_fetcher():
    """获取气象数据"""
    logger.info("=" * 60)
    logger.info("Step 2: 获取气象数据")
    logger.info("=" * 60)

    try:
        from scraper.qweather_fetcher import QWeatherFetcher
        fetcher = QWeatherFetcher(data_dir="data/raw")
        results = fetcher.fetch_all_stations()

        if results and results.get("stations"):
            logger.info(f"气象数据获取成功: {len(results['stations'])} 个站点")
            return True
        else:
            logger.warning("气象数据获取失败")
            return False

    except Exception as e:
        logger.error(f"气象数据获取失败: {e}")
        return False


def run_inference():
    """运行模型推理"""
    logger.info("=" * 60)
    logger.info("Step 3: 运行模型推理")
    logger.info("=" * 60)

    model_path = PROJECT_ROOT / "weights" / "stgat_best.onnx"
    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}，跳过推理")
        return False

    try:
        from model.inference import TaihuInference
        engine = TaihuInference(
            model_path=str(model_path),
            stations_path="data/stations.json",
            data_dir="data"
        )
        predictions = engine.predict()

        if predictions:
            logger.info(f"推理完成: {len(predictions)} 个站点")
            return True
        else:
            logger.warning("推理未生成结果")
            return False

    except Exception as e:
        logger.error(f"推理失败: {e}")
        return False


def run_pipeline():
    """运行完整数据管道"""
    start_time = time.time()
    now = datetime.now()
    logger.info(f"\n{'#' * 60}")
    logger.info(f"TaihuGuard 定时任务开始 - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#' * 60}\n")

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    results = {}

    # Step 1: 爬取水质数据
    results["water_quality"] = run_water_quality_scraper()

    # Step 2: 获取气象数据
    results["weather"] = run_weather_fetcher()

    # Step 3: 运行推理
    results["inference"] = run_inference()

    # 汇总
    elapsed = time.time() - start_time
    logger.info(f"\n{'#' * 60}")
    logger.info(f"任务完成 - 耗时 {elapsed:.1f}s")
    logger.info(f"  水质爬取: {'成功' if results['water_quality'] else '失败'}")
    logger.info(f"  气象获取: {'成功' if results['weather'] else '失败'}")
    logger.info(f"  模型推理: {'成功' if results['inference'] else '失败/跳过'}")
    logger.info(f"{'#' * 60}\n")

    return results


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="TaihuGuard 数据采集调度器")
    parser.add_argument("--step", choices=["all", "scrape", "weather", "inference"],
                        default="all", help="运行指定步骤")
    args = parser.parse_args()

    if args.step == "all":
        run_pipeline()
    elif args.step == "scrape":
        run_water_quality_scraper()
    elif args.step == "weather":
        run_weather_fetcher()
    elif args.step == "inference":
        run_inference()


if __name__ == "__main__":
    main()
