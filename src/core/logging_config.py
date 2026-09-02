import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 默认日志级别可从环境变量读取
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL = LOG_LEVELS.get(DEFAULT_LOG_LEVEL, logging.INFO)

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging():
    """配置全局日志：控制台 + 滚动文件（按大小轮转，保留5个备份）"""
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # 清除可能已经添加的 handler（避免重复）
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler（按大小轮转）
    log_file = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    # file_handler = RotatingFileHandler(
    #     log_file,
    #     maxBytes=10*1024*1024,  # 10MB
    #     backupCount=5,
    #     encoding="utf-8"
    # )
    # 按时间轮转
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",  # 每天午夜轮转
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )

    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 可选：按天轮转（如果希望按时间而非大小，可使用 TimedRotatingFileHandler）
    # 但两种方式选一种即可，这里使用大小轮转

    # 降低第三方库的日志噪音
    for lib in ["urllib3", "httpx", "httpx2", "httpcore", "openai", "tavily"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root_logger

# 为项目模块提供便捷的 logger 获取函数
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)