"""
logger.py — Thiết lập logging cho toàn bộ ứng dụng.

Ghi log vào file (với rotation) và console đồng thời.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from config import LOG_DIR


def setup_logger(name: str = "pdf_rag", level: int = logging.INFO) -> logging.Logger:
    """
    Tạo và cấu hình logger với:
    - Console handler (INFO+)
    - File handler với rotation (DEBUG+, max 5MB, giữ 3 file backup)
    """
    logger = logging.getLogger(name)

    # Tránh thêm handler trùng lặp nếu gọi nhiều lần
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format chuẩn
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console Handler ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (Rotating) ──
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "app.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Logger mặc định cho toàn project
logger = setup_logger()
