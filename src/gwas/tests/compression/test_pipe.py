import pickle
from multiprocessing import cpu_count

import numpy as np
import pytest
from gwas.compression.pipe import (
    CompressedBytesReader,
    CompressedBytesWriter,
    CompressedTextReader,
    CompressedTextWriter,
)
from upath import UPath


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "bz2", "lz4"])
def test_compressed_text(tmp_path: UPath, compression: str) -> None:
    x: str = "test" * 1000

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedTextWriter(test_path, num_threads=cpu_count()) as file_handle:
        file_handle.write(f"{x}\n")
    with CompressedTextReader(test_path) as file_handle:
        assert file_handle.read().strip() == x


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "bz2", "lz4"])
def test_compressed_bytes(tmp_path: UPath, compression: str) -> None:
    x = np.random.rand(1000, 1000)

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedBytesWriter(test_path, num_threads=cpu_count()) as file_handle:
        pickle.dump(x, file_handle)
    with CompressedBytesReader(test_path) as file_handle:
        assert np.allclose(pickle.load(file_handle), x)
