# -*- coding: utf-8 -*-
from pathlib import Path
from time import time

import blosc2
import numpy as np
import pytest

from gwas.compression.arr import ArrayProxy, compression_methods
from gwas.log import logger
from gwas.rmw import Scorefile

from .conftest import RmwScore


def compare_scorefile_arrays(array, test_array):
    for record, test_record in zip(array, test_array):
        for a, b in zip(record, test_record):
            if np.issubdtype(type(a), np.floating):
                if np.isnan(a):
                    assert np.isnan(b)
                    continue
            assert a == b


def test_scorefile(
    tmp_path: Path,
    rmw_scorefile_paths: list[Path],
    sample_size_label: str,
):
    if sample_size_label != "small":
        pytest.skip()

    scorefile = rmw_scorefile_paths[0]

    header, array = Scorefile.read(scorefile)

    test_scorefile = tmp_path / "test.score.txt"
    Scorefile.write(test_scorefile, header, array)

    test_header, test_array = Scorefile.read(test_scorefile)

    assert header == test_header
    compare_scorefile_arrays(array, test_array)


def test_compression(
    tmp_path: Path,
    rmw_score: RmwScore,
):
    scores = np.ascontiguousarray(
        np.dstack([rmw_score.array["U_STAT"], rmw_score.array["SQRT_V_STAT"]])
    ).astype(np.float64)

    blosc2.print_versions()

    for name, compression_method in compression_methods.items():
        file_path = tmp_path / f"score.{name}.h5"

        start = time()
        array_writer = ArrayProxy(
            file_path,
            scores.shape,
            np.float64,
        )
        array_writer[:, :] = scores
        end = time()

        file_size = file_path.stat().st_size
        compression_ratio = (scores.size * scores.dtype.itemsize) / file_size
        logger.info(
            f"Method {name} achieved a compression ratio of {compression_ratio:.2f} "
            f"in {end - start:.2f} seconds"
        )

        array = blosc2.open(urlpath=str(file_path))
        np.testing.assert_array_equal(scores, array[...])
