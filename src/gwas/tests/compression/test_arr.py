# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
from gwas.compression.arr.base import FileArray, compression_methods
from gwas.compression.arr.bin import Blosc2FileArray
from gwas.compression.pipe import CompressedTextReader
from pandas.testing import assert_frame_equal, assert_series_equal


def test_text_file_array(tmp_path: Path) -> None:
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
    array_proxy.set_axis_metadata(0, pd.DataFrame({"name": row_names, "data": row_data}))

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


def test_text_file_array_chunks(tmp_path: Path) -> None:
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
    array_proxy.set_axis_metadata(0, pd.DataFrame({"name": row_names, "data": row_data}))

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


def test_blosc2_file_array(tmp_path: Path) -> None:
    array_prefix = tmp_path / "test"
    array_proxy = FileArray.create(
        file_path=array_prefix,
        shape=(10, 10),
        dtype=np.float64,
        compression_method=compression_methods["blosc2_zstd_bitshuffle"],
    )

    assert isinstance(array_proxy, Blosc2FileArray)
    extra_metadata = {"extra": "metadata"}
    array_proxy.extra_metadata = extra_metadata

    column_names = [f"column_{i + 1:04d}" for i in range(array_proxy.shape[1])]
    column_metadata = pd.Series(column_names)
    array_proxy.set_axis_metadata(1, column_metadata)

    row_names = [f"row_{i + 1:04d}" for i in range(array_proxy.shape[0])]
    row_data = np.random.rand(array_proxy.shape[0])
    row_metadata = pd.DataFrame({"name": row_names, "data": row_data})
    array_proxy.set_axis_metadata(0, row_metadata)

    with array_proxy:
        array_proxy[0 : array_proxy.shape[0], 0 : array_proxy.shape[1]] = np.full(
            array_proxy.shape, 7, dtype=np.float64
        )

    (file_path,) = array_proxy.file_paths
    array = blosc2.open(file_path)
    assert np.allclose(array[:], 7)
    assert array.shape == array_proxy.shape

    axis_metadata_bytes = array.schunk.vlmeta.get_vlmeta("axis_metadata")
    _row_metadata, _column_metadata = pickle.loads(axis_metadata_bytes)
    assert_frame_equal(row_metadata, _row_metadata)
    assert_series_equal(column_metadata, _column_metadata)

    loaded_array_proxy: Blosc2FileArray = Blosc2FileArray.from_file(array_prefix)
    _row_metadata, _column_metadata = loaded_array_proxy.axis_metadata
    assert_frame_equal(row_metadata, _row_metadata)
    assert_series_equal(column_metadata, _column_metadata)

    assert loaded_array_proxy.extra_metadata == extra_metadata
