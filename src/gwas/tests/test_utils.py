import logging
from pathlib import Path

from gwas.log import (
    logger,
    multiprocessing_context,
    setup_logging,
    teardown_logging,
)
from gwas.utils import Process
from pytest import FixtureRequest, LogCaptureFixture


class LogProcess(Process):
    def func(self) -> None:
        logger.debug("Hello, world!")


def test_process(
    caplog: LogCaptureFixture, request: FixtureRequest, tmp_path: Path
) -> None:
    caplog.set_level(logging.DEBUG)

    setup_logging(level=logging.DEBUG, stream=False, path=tmp_path)
    request.addfinalizer(teardown_logging)

    from gwas.log import queue_listener

    assert queue_listener is not None
    queue_listener.handlers = (*logging.getLogger().handlers,)

    exception_queue = multiprocessing_context.Queue()
    process = LogProcess(name="LogProcess", exception_queue=exception_queue)
    process.start()
    process.join()

    assert "Hello, world!" in caplog.text
    assert exception_queue.empty()
    assert (tmp_path / "log.txt").is_file()