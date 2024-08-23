import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from gwas.compression.arr.base import (
    Blosc2CompressionMethod,
    FileArray,
    ParquetCompressionMethod,
    TextCompressionMethod,
    compression_methods,
)
from gwas.compression.arr.text import header_prefix
from gwas.compression.pipe import CompressedTextReader
from gwas.utils.threads import cpu_count

try:
    import blosc2 as blosc2
except ImportError:
    pass


@pytest.mark.parametrize("compression_method_name", compression_methods.keys())
def test_file_array(compression_method_name: str, tmp_path: UPath) -> None:
    compression_method = compression_methods[compression_method_name]
    if isinstance(compression_method, Blosc2CompressionMethod):
        if "blosc2" not in sys.modules:
            pytest.skip("blosc2 not installed")

    shape = (10, 10)
    writer = FileArray.create(
        file_path=tmp_path / "test",
        shape=shape,
        dtype=np.float64,
        compression_method=compression_method,
        num_threads=cpu_count(),
    )

    column_names = [f"column_{i + 1:04d}" for i in range(writer.shape[1])]
    writer.set_axis_metadata(1, column_names)

    row_names = [f"row_{i + 1:04d}" for i in range(writer.shape[0])]
    row_metadata = pd.DataFrame(
        {"name": row_names, "data": np.random.rand(writer.shape[0])}
    )
    writer.set_axis_metadata(0, row_metadata)

    with writer:
        writer[0:3, 0 : shape[1]] = np.full(
            (3, shape[1]), 7, dtype=np.float64, order="F"
        )
        writer[3 : shape[0], 0 : shape[1]] = np.full(
            (shape[0] - 3, shape[1]), 7, dtype=np.float64, order="F"
        )

    file_path = writer.file_path

    if isinstance(compression_method, Blosc2CompressionMethod):
        array = blosc2.open(str(file_path))
        matrix = array[:, :]
        metadata_bytes = array.schunk.vlmeta.get_vlmeta("axis_metadata")
        data_frame, column_names = pickle.loads(metadata_bytes)
    else:
        if isinstance(compression_method, TextCompressionMethod):
            with CompressedTextReader(file_path) as file_handle:
                data_frame = pd.read_csv(file_handle, sep="\t", skiprows=1)
            data_frame = data_frame.rename(
                columns=lambda c: c.removeprefix(header_prefix)
            )
        elif isinstance(compression_method, ParquetCompressionMethod):
            data_frame = pd.read_parquet(file_path)
        else:
            raise NotImplementedError

        matrix = data_frame[column_names].to_numpy()
        data_frame = data_frame.drop(columns=column_names)

    pd.testing.assert_frame_equal(data_frame, row_metadata)
    assert np.allclose(matrix, 7)

    reader = FileArray.from_file(file_path, dtype=np.float64, num_threads=cpu_count())
    assert reader.shape == writer.shape
    assert reader.dtype == writer.dtype
    assert reader.compression_method.suffix == writer.compression_method.suffix
    assert reader.file_paths <= writer.file_paths

    indices = np.array([0, 1, 2], dtype=np.uint32)
    data = reader[indices, indices]
    assert np.allclose(data, 7)
