import logging

import numpy as np
import pandas as pd
from pytest import FixtureRequest, LogCaptureFixture
from upath import UPath

from gwas.log import (
    logger,
    multiprocessing_context,
    setup_logging,
    teardown_logging,
)
from gwas.utils.genetics import greater_or_close, make_variant_mask
from gwas.utils.multiprocessing import Process


class LogProcess(Process):
    def func(self) -> None:
        logger.debug("Hello, world!")


def test_process(
    caplog: LogCaptureFixture, request: FixtureRequest, tmp_path: UPath
) -> None:
    caplog.set_level(logging.DEBUG)

    setup_logging(level=logging.DEBUG, stream=False, path=tmp_path)
    request.addfinalizer(teardown_logging)

    from gwas.log import queue_listener

    assert queue_listener is not None
    queue_listener.handlers = (*logging.getLogger().handlers,)

    exception_queue = multiprocessing_context.SimpleQueue()
    process = LogProcess(
        name="LogProcess", num_threads=None, exception_queue=exception_queue
    )
    process.start()
    process.join()

    assert "Hello, world!" in caplog.text
    assert exception_queue.empty()
    assert (tmp_path / "log.txt").is_file()


def test_greater_or_close() -> None:
    x = pd.Series([np.nan, 0.0, 0.01 - 1e-32, 0.01, 0.1])
    assert np.all(greater_or_close(x, 0.01) == [False, False, True, True, True])


def test_make_variant_mask() -> None:
    allele_frequencies = pd.Series([np.nan, 0.0, 0.01 - 1e-32, 0.01, 0.1])
    r_squared = pd.Series([1.0, 1.0, 1.0, 1.0, np.nan])
    assert np.all(
        make_variant_mask(
            allele_frequencies,
            r_squared,
            minor_allele_frequency_cutoff=0.01,
            r_squared_cutoff=0.8,
        )
        == [False, False, True, True, True]
    )
