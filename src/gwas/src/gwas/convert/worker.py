from itertools import batched

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..compression.arr.base import (
    FileArrayReader,
    FileArrayWriter,
)
from ..log import logger
from ..mem.arr import SharedArray
from ..vcf.base import VCFFile, base_allele_frequency_columns


def update_row_metadata_dtypes(
    name: str, row_metadata: pd.DataFrame | pd.Series
) -> pd.DataFrame:
    if name.endswith("score"):
        if isinstance(row_metadata, pd.Series):
            raise ValueError("Score file must have a data frame of row metadata")
        VCFFile.update_data_frame_types(row_metadata)
        row_metadata["is_imputed"] = row_metadata["is_imputed"].astype(bool)
        float_columns: set[str] = {"r_squared"}
        for column in row_metadata.columns:
            if any(column.endswith(s) for s in base_allele_frequency_columns):
                float_columns.add(column)
        for column in float_columns:
            row_metadata[column] = row_metadata[column].astype(np.float64)
        data_frame = row_metadata
    elif name.endswith("covariance"):
        if isinstance(row_metadata, pd.DataFrame):
            if len(row_metadata.columns) == 1:
                row_metadata = row_metadata.iloc[:, 0]
        if isinstance(row_metadata, pd.Series):
            data_frame = pd.DataFrame(dict(variable=row_metadata), dtype="string")
        else:
            raise ValueError(
                f"Expected a series of row metadata, received {row_metadata}"
            )
    else:
        raise ValueError(f"Unknown file type: {name}")
    return data_frame


def copy(
    reader: FileArrayReader,
    writer: FileArrayWriter,
    array: SharedArray[np.float64],
) -> None:
    row_count, column_count = reader.shape
    column_indices = np.arange(column_count, dtype=np.uint32)
    row_chunk_size = array.size // column_count

    logger.debug(f'Converting "{reader.file_path}" with shape {reader.shape}')

    row_chunks = [
        np.asarray(row_indices, dtype=np.uint32)
        for row_indices in batched(range(row_count), row_chunk_size)
    ]

    with reader, writer:
        for row_indices in tqdm(row_chunks, desc="converting", unit="chunks"):
            row_start, row_end = row_indices[0], row_indices[-1] + 1
            logger.debug(
                f"Reading chunk from {row_start} to {row_end} of {row_count} total rows"
            )

            shape = (column_count, len(row_indices))
            # Ensure C-contiguous for blosc2
            data = array.to_numpy(shape=shape).transpose()
            reader.read_indices(row_indices, column_indices, data)

            # Ensure F-contiguous for parquet
            logger.debug("Transposing chunk")
            array.transpose(shape)

            logger.debug(f"Writing chunk with shape {data.shape} at {row_start}")
            data = array.to_numpy(shape=shape[::-1])
            writer[row_start:row_end, :] = data
    logger.info(
        f'Completed conversion from "{reader.file_path}" to "{writer.file_path}"'
    )


def verify(
    reader1: FileArrayReader,
    reader2: FileArrayReader,
    array: SharedArray[np.float64],
) -> None:
    logger.debug(f'Verifying "{reader2.file_path}" with shape {reader2.shape}')

    row_count, column_count = reader1.shape
    column_indices = np.arange(column_count, dtype=np.uint32)
    row_chunk_size = array.size // column_count // 2

    row_chunks = [
        np.asarray(row_indices, dtype=np.uint32)
        for row_indices in batched(range(row_count), row_chunk_size)
    ]

    with reader1, reader2:
        for row_indices in tqdm(row_chunks, desc="verifying", unit="chunks"):
            row_start, row_end = row_indices[0], row_indices[-1]
            logger.debug(
                f"Reading chunk from {row_start} to {row_end} of {row_count} total rows"
            )

            data = array.to_numpy(shape=(column_count, len(row_indices) * 2)).transpose()
            data.fill(np.nan)
            data1 = data[: len(row_indices), :]
            data2 = data[len(row_indices) :, :]

            reader1.read_indices(row_indices, column_indices, data1)
            reader2.read_indices(row_indices, column_indices, data2)

            np.testing.assert_array_equal(
                data1,
                data2,
                strict=True,
                err_msg=f"Failed to verify chunk {row_start}:{row_end}",
            )
    logger.info(
        f'Verified conversion from "{reader1.file_path}" to "{reader2.file_path}"'
    )
