# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import numpy as np
import pytest

from gwas.compression.pipe import (
    CompressedBytesReader,
    CompressedBytesWriter,
    CompressedTextReader,
    CompressedTextWriter,
)


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "lrz", "bz2"])
def test_compressed_text(tmp_path: Path, compression: str):
    x: str = "test" * 1000

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedTextWriter(test_path) as file_handle:
        file_handle.write(f"{x}\n")
    with CompressedTextReader(test_path) as file_handle:
        assert file_handle.read().strip() == x


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "lrz", "bz2"])
def test_compressed_bytes(tmp_path: Path, compression: str):
    x = np.random.rand(1000, 1000)

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedBytesWriter(test_path) as file_handle:
        pickle.dump(x, file_handle)
    with CompressedBytesReader(test_path) as file_handle:
        assert np.allclose(pickle.load(file_handle), x)


# def test_array_proxy(tmp_path: Path):
#     array_proxy = ArrayProxy(
#         file_path=tmp_path / "test.b2array",
#         shape=(1000, 1000, 10),
#         dtype=np.float64,
#     )

#     import pdb

#     pdb.set_trace()
