# -*- coding: utf-8 -*-
from pathlib import Path
from time import time

import numpy as np

from gwas.compression.arr.base import (
    FileArray,
    TextCompressionMethod,
    compression_methods,
)
from gwas.compression.pipe import CompressedTextReader
from gwas.log import logger

from .conftest import RmwScore


def test_compression(
    tmp_path: Path,
    rmw_score: RmwScore,
) -> None:
    scores = np.ascontiguousarray(
        np.dstack([rmw_score.array["U_STAT"], rmw_score.array["SQRT_V_STAT"]])
    ).astype(np.float64)

    variant_count, phenotype_count, _ = scores.shape
    scores = scores.reshape(variant_count, phenotype_count * 2)

    for name, compression_method in compression_methods.items():
        file_path = tmp_path / f"score.{name}{compression_method.suffix}"

        start = time()
        writer = FileArray.create(
            file_path, scores.shape, np.float64, compression_method
        )
        with writer:
            writer[:, :] = scores
        end = time()

        file_size = file_path.stat().st_size
        compression_ratio = (scores.size * scores.dtype.itemsize) / file_size
        logger.info(
            f"Method {name} achieved a compression ratio of {compression_ratio:.2f} "
            f"in {end - start:.2f} seconds"
        )

        if isinstance(compression_method, TextCompressionMethod):
            with CompressedTextReader(file_path) as file_handle:
                array = np.loadtxt(file_handle)
        else:
            raise NotImplementedError
        np.testing.assert_array_equal(scores, array[...])
