from time import time

import numpy as np
import pytest
from upath import UPath

from gwas.compression.arr.base import (
    FileArray,
    compression_methods,
)
from gwas.log import logger
from gwas.utils.threads import cpu_count

from .conftest import RmwScore


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_compression(
    tmp_path: UPath,
    rmw_score: RmwScore,
) -> None:
    scores = np.ascontiguousarray(
        np.dstack([rmw_score.array["U_STAT"], rmw_score.array["SQRT_V_STAT"]])
    ).astype(np.float64)

    variant_count, phenotype_count, _ = scores.shape
    scores = scores.reshape(variant_count, phenotype_count * 2)
    scores = np.asfortranarray(scores)

    for name, compression_method in compression_methods.items():
        file_path = tmp_path / f"score.{name}{compression_method.suffix}"

        start = time()
        writer = FileArray.create(
            file_path,
            scores.shape,
            np.float64,
            compression_method,
            num_threads=cpu_count(),
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

        reader = FileArray.from_file(file_path, np.float64, cpu_count())
        with reader:
            array = reader[:, :]
        np.testing.assert_array_equal(scores, array[...])
