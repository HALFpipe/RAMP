# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

from gwas.compression.arr.base import FileArray, compression_methods
from gwas.compression.pipe import CompressedTextReader


def test_text_file_array(tmp_path: Path):
    array_proxy = FileArray.create(
        file_path=tmp_path / "test",
        shape=(10, 10),
        dtype=np.float64,
        compression_method=compression_methods["zstd_text"],
    )

    column_names = [f"column_{i + 1:04d}" for i in range(array_proxy.shape[1])]
    array_proxy.set_axis_metadata(1, pd.Series(column_names))

    row_names = [f"row_{i + 1:04d}" for i in range(array_proxy.shape[0])]
    row_data = np.random.rand(array_proxy.shape[0])
    array_proxy.set_axis_metadata(
        0, pd.DataFrame({"name": row_names, "data": row_data})
    )

    with array_proxy:
        array_proxy[0 : array_proxy.shape[0], 0 : array_proxy.shape[1]] = np.full(
            array_proxy.shape, 7
        )

    (file_path,) = array_proxy.file_paths
    with CompressedTextReader(file_path) as file_handle:
        data_frame = pd.read_csv(file_handle, sep="\t")
        assert (data_frame.columns == ["name", "data", *column_names]).all()
        assert (data_frame["name"] == row_names).all()
        assert np.allclose(data_frame["data"], row_data)
        assert np.allclose(data_frame[column_names], 7)


def test_text_file_array_chunks(tmp_path: Path):
    array_proxy = FileArray.create(
        file_path=tmp_path / "test",
        shape=(10, 10),
        dtype=np.float64,
        compression_method=compression_methods["zstd_text"],
    )

    column_names = [f"column_{i + 1:04d}" for i in range(array_proxy.shape[1])]
    array_proxy.set_axis_metadata(1, pd.Series(column_names))

    row_names = [f"row_{i + 1:04d}" for i in range(array_proxy.shape[0])]
    row_data = np.random.rand(array_proxy.shape[0])
    array_proxy.set_axis_metadata(
        0, pd.DataFrame({"name": row_names, "data": row_data})
    )

    with array_proxy:
        array_proxy[0 : array_proxy.shape[0], 0 : array_proxy.shape[1]] = np.full(
            array_proxy.shape, 7
        )

    (file_path,) = array_proxy.file_paths
    with CompressedTextReader(file_path) as file_handle:
        data_frame = pd.read_csv(file_handle, sep="\t")
        assert (data_frame.columns == ["name", "data", *column_names]).all()
        assert (data_frame["name"] == row_names).all()
        assert np.allclose(data_frame["data"], row_data)
        assert np.allclose(data_frame[column_names], 7)
