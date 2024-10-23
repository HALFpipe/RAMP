import pickle
from random import randbytes

import numpy as np
import pytest
from upath import UPath

from gwas.compression.pipe import (
    CompressedBytesReader,
    CompressedBytesWriter,
    CompressedTextReader,
    CompressedTextWriter,
)
from gwas.utils.threads import cpu_count


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "bz2", "lz4"])
def test_compressed_text(tmp_path: UPath, compression: str) -> None:
    x: str = "test" * 1000

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedTextWriter(test_path, num_threads=cpu_count()) as file_handle:
        file_handle.write(f"{x}\n")
    with CompressedTextReader(test_path) as file_handle:
        assert file_handle.read().strip() == x


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "bz2", "lz4"])
def test_partial(tmp_path: UPath, compression: str) -> None:
    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedTextWriter(test_path, num_threads=cpu_count()) as file_handle:
        file_handle.write("test")
        file_handle.write(randbytes(100000).hex())
    with CompressedTextReader(test_path) as file_handle:
        assert file_handle.read(4) == "test"


@pytest.mark.parametrize("compression", ["zst", "xz", "gz", "bz2", "lz4"])
def test_compressed_bytes(tmp_path: UPath, compression: str) -> None:
    x = np.random.rand(1000, 1000)

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedBytesWriter(test_path, num_threads=cpu_count()) as file_handle:
        pickle.dump(x, file_handle)
    with CompressedBytesReader(test_path) as file_handle:
        np.testing.assert_allclose(pickle.load(file_handle), x)
