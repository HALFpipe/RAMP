import pickle
from argparse import Namespace
from pathlib import Path

import blosc2
import numpy as np
from tqdm.auto import tqdm

from ..compression.arr.base import FileArray, default_compression_method
from ..compression.pipe import CompressedBytesReader
from ..log import logger

suffix_to_convert = ".b2array"


def axis_metadata_path(path: Path) -> Path:
    return path.parent / f"{path.stem}.axis-metadata.pkl.zst"


def convert(arguments: Namespace) -> None:
    num_threads = arguments.num_threads

    path = Path(arguments.path)
    if path.is_file():
        paths = [path]
    else:
        paths = list(path.rglob(f"*{suffix_to_convert}"))

    for path in tqdm(paths, unit="files"):
        if path.is_file():
            convert_file(path, num_threads)


def convert_file(path: Path, num_threads: int) -> None:
    if path.name.endswith(suffix_to_convert):
        array = blosc2.open(
            urlpath=str(path),
            cparams=dict(nthreads=num_threads),
            dparams=dict(nthreads=num_threads),
        )
        try:
            vlmeta = array.schunk.vlmeta
            axis_metadata_bytes = vlmeta.get_vlmeta("axis_metadata")
            row_metadata, column_metadata = pickle.loads(axis_metadata_bytes)
        except KeyError:
            if axis_metadata_path(path).is_file():
                with CompressedBytesReader(axis_metadata_path(path)) as file_handle:
                    row_metadata, column_metadata = pickle.load(file_handle)
            else:
                row_metadata, column_metadata = None, None

        row_chunk_size, _ = array.chunks
        row_count, column_count = array.shape
        stat_file_array_path = path.with_suffix(".txt.zst")
        if stat_file_array_path.is_file():
            logger.warning(
                f"Skipping {path} because {stat_file_array_path} already exists"
            )
            return
        stat_file_array = FileArray.create(
            stat_file_array_path,
            (row_count, column_count),
            np.float64,
            compression_method=default_compression_method,
            num_threads=num_threads,
        )
        stat_file_array.set_axis_metadata(0, row_metadata)
        stat_file_array.set_axis_metadata(1, column_metadata)

        with stat_file_array:
            for row_start in tqdm(
                range(0, row_count, row_chunk_size), unit="chunks", leave=False
            ):
                row_end = min(row_start + row_chunk_size, row_count)
                row_chunk = array[row_start:row_end, :]
                stat_file_array[row_start:row_end, :] = row_chunk
