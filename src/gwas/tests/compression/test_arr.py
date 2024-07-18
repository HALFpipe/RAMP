# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
from gwas.compression.arr.base import FileArray, compression_methods
from gwas.compression.arr.text import header_prefix
from gwas.compression.pipe import CompressedTextReader


def test_text_file_array(tmp_path: Path) -> None:
    shape = (10, 10)
    writer = FileArray.create(
        file_path=tmp_path / "test",
        shape=shape,
        dtype=np.float64,
        compression_method=compression_methods["zstd_text"],
    )

    column_names = [f"column_{i + 1:04d}" for i in range(writer.shape[1])]
    writer.set_axis_metadata(1, pd.Series(column_names))

    row_names = [f"row_{i + 1:04d}" for i in range(writer.shape[0])]
    row_data = np.random.rand(writer.shape[0])
    writer.set_axis_metadata(0, pd.DataFrame({"name": row_names, "data": row_data}))

    with writer:
        writer[0:3, 0 : shape[1]] = np.full((3, shape[1]), 7, dtype=np.float64)
        writer[3 : shape[0], 0 : shape[1]] = np.full(
            (shape[0] - 3, shape[1]), 7, dtype=np.float64
        )

    file_path = writer.file_path
    with CompressedTextReader(file_path) as file_handle:
        data_frame = pd.read_csv(file_handle, sep="\t", skiprows=1)
        data_frame = data_frame.rename(columns=lambda c: c.removeprefix(header_prefix))
        assert (data_frame.columns == ["name", "data", *column_names]).all()
        assert (data_frame["name"] == row_names).all()
        assert np.allclose(data_frame["data"], row_data)
        assert np.allclose(data_frame[column_names], 7)

    reader = FileArray.from_file(file_path, dtype=np.float64)
    assert reader.shape == writer.shape
    assert reader.dtype == writer.dtype
    assert reader.compression_method == writer.compression_method
    assert reader.file_paths <= writer.file_paths
