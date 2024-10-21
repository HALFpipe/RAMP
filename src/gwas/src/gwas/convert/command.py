from argparse import Namespace

import numpy as np
from pandas.api.types import is_object_dtype
from upath import UPath

from ..compression.arr.base import FileArray
from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils.multiprocessing import (
    get_global_lock,
)
from .base import compression_method, dtype
from .worker import copy, update_row_metadata_dtypes, verify


def convert(arguments: Namespace, sw: SharedWorkspace) -> None:
    num_threads = arguments.num_threads
    input_path = UPath(arguments.path)
    size = sw.unallocated_size // dtype().itemsize

    with get_global_lock():
        array = sw.alloc(SharedArray.get_name(sw), size, dtype=dtype)

    output_path: UPath | None = None
    try:
        logger.debug(f'Reading metadata for "{input_path}"')

        reader = FileArray.from_file(input_path, dtype, num_threads=num_threads)
        row_count, column_count = reader.shape

        name = input_path.name.removesuffix(reader.compression_method.suffix)
        output_path = input_path.parent / f"{name}{compression_method.suffix}"
        if output_path.is_file():
            logger.warning(
                f'Not converting "{input_path}" because '
                f'"{output_path}" already exists'
            )
        else:
            writer = FileArray.create(
                output_path,
                (row_count, column_count),
                np.float64,
                compression_method=compression_method,
                num_threads=num_threads,
            )

            row_metadata = reader.row_metadata
            if row_metadata is not None:
                row_metadata = update_row_metadata_dtypes(name, row_metadata)
                for d in row_metadata.dtypes:
                    if is_object_dtype(d):
                        raise ValueError("Refusing to convert object dtype")

                writer.set_axis_metadata(0, row_metadata)
            if reader.column_names is not None:
                writer.set_axis_metadata(1, reader.column_names)

            copy(reader, writer, array)

        converted_reader = FileArray.from_file(
            output_path, dtype, num_threads=num_threads
        )
        verify(reader, converted_reader, array)
    except Exception as exception:
        logger.error(f'Error converting "{input_path}": {exception}', exc_info=True)
        if output_path is not None:
            logger.info(f'Deleting "{output_path}" after error', exc_info=True)
            output_path.unlink(missing_ok=True)
