# -*- coding: utf-8 -*-
import logging
import warnings
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path
from typing import TextIO

logger = logging.getLogger("gwas")
multiprocessing_context = get_context("forkserver")
multiprocessing_context.set_forkserver_preload(["gwas"])


logging_queue: Queue[logging.LogRecord] | None = None
queue_listener: QueueListener | None = None

handlers: list[logging.Handler] = []


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    logging.getLogger("py.warnings").warning(
        warnings.formatwarning(message, category, filename, lineno, line),
        stack_info=True,
    )


def capture_warnings() -> None:
    warnings.showwarning = _showwarning


def setup_logging(
    level: str | int, path: Path | None = None, stream: bool = True
) -> None:
    setup_live_logging(level, path, stream)
    capture_warnings()
    setup_logging_queue()

    logger.debug(f"Configured logging with handlers {handlers}")


def add_handler(handler: logging.Handler) -> None:
    global handlers
    handlers.append(handler)


def setup_live_logging(
    level: str | int = logging.DEBUG, path: Path | None = None, stream: bool = True
) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)8s] [%(processName)14s] %(funcName)s: "
        "%(message)s (%(filename)s:%(lineno)s)"
    )

    global handlers
    if stream is True:
        handlers.append(logging.StreamHandler())
    if path is not None:
        handlers.append(
            logging.FileHandler(path / "log.txt", "a", errors="backslashreplace")
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root.addHandler(handler)


def setup_logging_queue() -> None:
    global logging_queue, queue_listener, handlers
    if logging_queue is not None:
        return
    logging_queue = multiprocessing_context.Queue()
    queue_listener = QueueListener(logging_queue, *handlers, respect_handler_level=True)
    queue_listener.start()


def teardown_logging() -> None:
    global logging_queue, queue_listener
    if queue_listener is not None:
        queue_listener.stop()
        queue_listener = None
    if logging_queue is not None:
        logging_queue.close()
        logging_queue = None

    root = logging.getLogger()

    global handlers
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()


def worker_configurer(
    logging_queue: Queue[logging.LogRecord], log_level: int | str
) -> None:
    queue_handler = QueueHandler(logging_queue)
    queue_handler.setLevel(log_level)

    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(log_level)

    capture_warnings()

    logger.debug(f"Configured logging with handler {queue_handler}")
