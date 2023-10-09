# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import warnings
from logging.handlers import QueueHandler
from multiprocessing import Queue
from pathlib import Path
from threading import Thread

logger = logging.getLogger("gwas")
logging_thread: LoggingThread | None = None


def _showwarning(message, category, filename, lineno, file=None, line=None):
    logger = logging.getLogger("py.warnings")
    logger.warning(
        warnings.formatwarning(message, category, filename, lineno, line),
        stack_info=True,
    )


def setup_logging(level: str | int, log_path: Path) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)8s] %(funcName)s: "
        "%(message)s (%(filename)s:%(lineno)s)"
    )

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_path / "log.txt", "a", errors="backslashreplace"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    warnings.showwarning = _showwarning

    global logging_thread
    logging_thread = LoggingThread()
    logging_thread.start()


class LoggingThread(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.logging_queue: Queue[logging.LogRecord] = Queue()

    def run(self) -> None:
        while True:
            record = self.logging_queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)


def worker_configurer(
    logging_queue: Queue[logging.LogRecord], log_level: int | str
) -> None:
    queue_handler = QueueHandler(logging_queue)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(log_level)

    logging.captureWarnings(True)
