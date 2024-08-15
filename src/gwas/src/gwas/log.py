import logging
import multiprocessing as mp
import warnings
from logging import Handler, LogRecord
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.queues import SimpleQueue
from typing import TextIO, override

from upath import UPath

logger = logging.getLogger("gwas")
multiprocessing_context = mp.get_context("forkserver")
multiprocessing_context.set_forkserver_preload(["gwas"])


logging_queue: SimpleQueue[LogRecord] | None = None
queue_listener: QueueListener | None = None

handlers: list[Handler] = []


class _QueueHandler(QueueHandler):
    queue: SimpleQueue[LogRecord]  # type: ignore

    def __init__(self, queue: SimpleQueue[LogRecord]) -> None:
        Handler.__init__(self)
        self.queue = queue

    @override
    def enqueue(self, record: LogRecord) -> None:
        self.queue.put(record)


class _QueueListener(QueueListener):
    queue: SimpleQueue[LogRecord]  # type: ignore

    def __init__(
        self,
        queue: SimpleQueue[LogRecord],
        *handlers: Handler,
        respect_handler_level: bool = False,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self._thread = None
        self.respect_handler_level = respect_handler_level

    @override
    def dequeue(self, block: bool) -> LogRecord:
        return self.queue.get()

    @override
    def enqueue_sentinel(self):
        self.queue.put(self._sentinel)  # type: ignore


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
    level: str | int, path: UPath | None = None, stream: bool = True
) -> None:
    setup_live_logging(level, path, stream)
    capture_warnings()
    setup_logging_queue()

    logger.debug(f"Configured logging with handlers {handlers}")


def add_handler(handler: Handler) -> None:
    global handlers
    handlers.append(handler)


def setup_live_logging(
    level: str | int = logging.DEBUG, path: UPath | None = None, stream: bool = True
) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)16s] [%(levelname)8s] [%(processName)20s] "
        "%(funcName)s: %(message)s (%(filename)s:%(lineno)s)"
    )

    global handlers
    if stream is True:
        handlers.append(logging.StreamHandler())
    if path is not None:
        handlers.append(
            logging.FileHandler(path / "log.txt", "a", errors="backslashreplace")
        )

    loggers: list[logging.Logger] = [root, mp.get_logger()]
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        for logger in loggers:
            logger.setLevel(level)
            logger.addHandler(handler)


def setup_logging_queue() -> None:
    global logging_queue, queue_listener, handlers
    if logging_queue is not None:
        return
    logging_queue = multiprocessing_context.SimpleQueue()
    queue_listener = _QueueListener(logging_queue, *handlers, respect_handler_level=True)
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
    logging_queue: SimpleQueue[LogRecord], log_level: int | str
) -> None:
    queue_handler = _QueueHandler(logging_queue)
    queue_handler.setLevel(log_level)

    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(log_level)

    capture_warnings()

    logger.debug(f"Configured logging with handler {queue_handler}")
