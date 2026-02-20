"""
水质监测数据爬虫
从生态环境部地表水水质自动监测实时数据发布系统爬取太湖流域水质数据
URL: http://106.37.208.243:8068/GJZ/Business/Publish/Main.html
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 水质参数列名映射
PARAM_COLUMNS = [
    "station_name", "time", "water_temp", "ph", "do", "conductivity",
    "turbidity", "codmn", "nh3n", "tp", "tn", "chla", "algae_density"
]

# 水质参数英文到中文映射
PARAM_CN = {
    "water_temp": "水温",
    "ph": "pH",
    "do": "溶解氧",
    "conductivity": "电导率",
    "turbidity": "浊度",
    "codmn": "高锰酸盐指数",
    "nh3n": "氨氮",
    "tp": "总磷",
    "tn": "总氮",
    "chla": "叶绿素a",
    "algae_density": "藻密度"
}

BASE_URL = "http://106.37.208.243:8068/GJZ/Business/Publish/Main.html"
MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒


class WaterQualityScraper:
    """生态环境部水质数据爬虫"""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.driver = None

    def _init_driver(self):
        """初始化 Selenium Chrome 驱动"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=zh-CN")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(60)
            logger.info("Chrome 驱动初始化成功")
        except Exception as e:
            logger.error(f"Chrome 驱动初始化失败: {e}")
            raise

    def _close_driver(self):
        """关闭浏览器驱动"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _select_taihu_basin(self):
        """选择太湖流域筛选条件"""
        try:
            wait = WebDriverWait(self.driver, 20)

            # 等待页面加载，查找流域选择下拉框
            basin_select = wait.until(
                EC.presence_of_element_located((By.ID, "lybm"))
            )
            select = Select(basin_select)
            # 选择太湖流域
            for option in select.options:
                if "太湖" in option.text:
                    select.select_by_visible_text(option.text)
                    logger.info(f"已选择流域: {option.text}")
                    break

            # 等待数据加载
            time.sleep(3)

            # 点击查询按钮（如果有）
            try:
                query_btn = self.driver.find_element(By.CSS_SELECTOR, ".btn-query, #btnQuery, .search-btn")
                query_btn.click()
                time.sleep(3)
            except Exception:
                pass  # 可能自动刷新无需点击

        except Exception as e:
            logger.warning(f"选择太湖流域时出错: {e}，尝试直接解析页面")

    def _parse_table_data(self):
        """解析页面表格数据"""
        records = []
        try:
            wait = WebDriverWait(self.driver, 20)

            # 尝试多种表格选择器
            table_selectors = [
                "table.data-table tbody tr",
                "#dataTable tbody tr",
                ".table-container table tbody tr",
                "table tbody tr",
            ]

            rows = []
            for selector in table_selectors:
                try:
                    rows = wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    if rows:
                        logger.info(f"找到 {len(rows)} 行数据 (selector: {selector})")
                        break
                except Exception:
                    continue

            if not rows:
                logger.warning("未找到数据表格")
                return records

            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 5:
                        continue

                    cell_texts = [c.text.strip() for c in cells]

                    # 解析数值，处理 '--' 和空值
                    record = {"station_name": cell_texts[0] if cell_texts else ""}

                    # 时间
                    record["time"] = cell_texts[1] if len(cell_texts) > 1 else ""

                    # 水质参数（从第3列开始）
                    param_keys = [
                        "water_temp", "ph", "do", "conductivity", "turbidity",
                        "codmn", "nh3n", "tp", "tn", "chla", "algae_density"
                    ]
                    for i, key in enumerate(param_keys):
                        idx = i + 2
                        if idx < len(cell_texts):
                            val = cell_texts[idx]
                            if val in ("--", "-", "", "—", "N/A"):
                                record[key] = None
                            else:
                                try:
                                    record[key] = float(val)
                                except ValueError:
                                    record[key] = None
                        else:
                            record[key] = None

                    if record["station_name"]:
                        records.append(record)
                except Exception as e:
                    logger.debug(f"解析行数据出错: {e}")
                    continue

            logger.info(f"成功解析 {len(records)} 条水质记录")

        except Exception as e:
            logger.error(f"解析表格数据失败: {e}")

        return records

    def _handle_pagination(self):
        """处理分页，获取所有页面的数据"""
        all_records = []

        # 先获取当前页数据
        records = self._parse_table_data()
        all_records.extend(records)

        # 尝试翻页
        page = 1
        while True:
            try:
                next_btn = self.driver.find_element(
                    By.CSS_SELECTOR, ".next-page, .pagination .next, a[title='下一页']"
                )
                if "disabled" in next_btn.get_attribute("class") or "":
                    break
                next_btn.click()
                time.sleep(2)
                page += 1
                records = self._parse_table_data()
                if not records:
                    break
                all_records.extend(records)
                logger.info(f"翻到第 {page} 页，累计 {len(all_records)} 条")
            except Exception:
                break

        return all_records

    def scrape(self):
        """执行一次完整的水质数据爬取"""
        now = datetime.now()
        logger.info(f"开始爬取水质数据 - {now.strftime('%Y-%m-%d %H:%M')}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._init_driver()
                self.driver.get(BASE_URL)
                time.sleep(5)  # 等待页面初始加载

                # 选择太湖流域
                self._select_taihu_basin()

                # 解析数据（含翻页）
                records = self._handle_pagination()

                if not records:
                    logger.warning(f"第 {attempt} 次尝试未获取到数据")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error("多次重试后仍未获取到数据")
                        return []

                # 添加爬取时间戳
                for r in records:
                    r["scrape_time"] = now.isoformat()

                # 保存数据
                self._save_data(records, now)

                logger.info(f"成功爬取 {len(records)} 条水质记录")
                return records

            except Exception as e:
                logger.error(f"第 {attempt} 次爬取失败: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
            finally:
                self._close_driver()

        return []

    def _save_data(self, records, timestamp):
        """保存爬取数据到 JSON 文件"""
        date_dir = self.data_dir / timestamp.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = f"water_quality_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        filepath = date_dir / filename

        # 检查是否已存在相同时段数据（避免重复爬取）
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
        """获取最新一次的爬取数据"""
        # 找到最新的数据文件
        json_files = sorted(self.data_dir.glob("**/*.json"), reverse=True)
        if not json_files:
            return None

        with open(json_files[0], "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="太湖水质数据爬虫")
    parser.add_argument("--data-dir", default="data/raw", help="数据存储目录")
    args = parser.parse_args()

    scraper = WaterQualityScraper(data_dir=args.data_dir)
    records = scraper.scrape()
    if records:
        print(f"成功爬取 {len(records)} 条记录")
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()
