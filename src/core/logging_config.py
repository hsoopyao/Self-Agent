import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

def setup_logging():
    """配置日志：同时输出到控制台和文件"""
    logging.basicConfig(
        level=logging.ERROR,  # 只记录 ERROR 及以上级别
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()  # 控制台也输出，便于开发调试
        ]
    )
    # 减少第三方库的日志噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)