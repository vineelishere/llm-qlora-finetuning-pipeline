"""
Logger Utility
--------------
One consistent logging setup for the whole project. Instead of scattered
print() calls, every module gets timestamps, log levels and a per-run log file
on disk for debugging:

  - Timestamped console output
  - Log levels (INFO, WARNING, ERROR, DEBUG)
  - A log file per run under logs/
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: Logger name (pass __name__ from the calling module).
        log_dir: Directory where per-run log files are written.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Reuse an already-configured logger so handlers (and log lines) don't duplicate
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Log format: [TIME] [LEVEL] [MODULE] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - prints to the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - keeps a copy of the run on disk
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Quick check
    test_logger = get_logger("test_logger")
    test_logger.info("Logger is working correctly.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
