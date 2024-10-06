import pickle
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self, Type, override

import blosc2
import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from ...compression.pipe import CompressedBytesReader
from ...log import logger
from .base import Blosc2CompressionMethod, FileArrayReader, FileArrayWriter, ScalarType

base_cparams = dict(
    codec=blosc2.Codec.ZSTD,
    clevel=9,
    filters=[blosc2.Filter.SHUFFLE],
    use_dict=False,
)


def get_vlmeta(array: blosc2.NDArray, key: str) -> Any:
    vlmeta = array.schunk.vlmeta
    return pickle.loads(vlmeta.get_vlmeta(key))


def set_vlmeta(array: blosc2.NDArray, key: str, value: Any) -> None:
    value_bytes = pickle.dumps(value)
    logger.debug(f"Setting vlmeta key {key} with {len(value_bytes)} bytes")
    vlmeta = array.schunk.vlmeta
    vlmeta.set_vlmeta(key, value_bytes, **base_cparams)


@dataclass(kw_only=True)
class Blosc2FileArrayWriter(FileArrayWriter[ScalarType]):
    array: blosc2.NDArray | None = None

    def __enter__(self) -> Self:
        self.array = blosc2.full(
            shape=self.shape,
            fill_value=np.nan,
            urlpath=str(self.file_path),
            dtype=self.dtype,
            cparams=dict(**base_cparams, nthreads=self.num_threads),
            dparams=dict(nthreads=self.num_threads),
        )
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.dump_metadata()
        self.array = None

    def __setitem__(
        self, key: tuple[slice, slice], value: npt.NDArray[ScalarType]
    ) -> None:
        if self.array is None:
            raise RuntimeError("Blosc2 array is not initialized")
        self.array[key] = np.ascontiguousarray(value)

    def dump_metadata(self) -> None:
        axis_metadata = [self.row_metadata, self.column_names]
        set_vlmeta(self.array, "axis_metadata", axis_metadata)
        if self.extra_metadata is not None:
            for key, value in self.extra_metadata.items():
                set_vlmeta(self.array, key, value)


def load_metadata(
    file_path: UPath, array: blosc2.NDArray
) -> tuple[pd.DataFrame | None, list[str] | None, dict[str, Any]]:
    try:
        row_metadata, column_metadata = get_vlmeta(array, "axis_metadata")
    except KeyError:
        # Compatibility with how metadata was stored previously
        axis_metadata_path = file_path.parent / f"{file_path.stem}.axis-metadata.pkl.zst"
        if axis_metadata_path.is_file():
            with CompressedBytesReader(axis_metadata_path) as file_handle:
                row_metadata, column_metadata = pickle.load(file_handle)
        else:
            row_metadata, column_metadata = None, None

    extra_metadata: dict[str, Any] = dict()
    keys = set(array.schunk.vlmeta.get_names()) - {"axis_metadata"}
    if keys:
        for key in keys:
            extra_metadata[key] = get_vlmeta(array, key)

    if column_metadata is not None:
        column_metadata = list(column_metadata)

    return row_metadata, column_metadata, extra_metadata


@dataclass(kw_only=True)
class Blosc2FileArrayReader(FileArrayReader[ScalarType]):
    num_threads: int

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def read_indices(
        self,
        row_indices: npt.NDArray[np.uint32],
        column_indices: npt.NDArray[np.uint32],
        array: npt.NDArray[ScalarType],
    ) -> None:
        from ._blosc2_get_orthogonal_selection import get_orthogonal_selection

        get_orthogonal_selection(
            urlpath=str(self.file_path).encode(),
            row_indices=row_indices.astype(np.int64),
            column_indices=column_indices.astype(np.int64),
            array=array,
            num_threads=self.num_threads,
        )

    @override
    @classmethod
    def from_file(
        cls, file_path: UPath, dtype: Type[ScalarType], num_threads: int
    ) -> FileArrayReader:
        params = dict(nthreads=num_threads)
        b2array = blosc2.open(urlpath=str(file_path), cparams=params, dparams=params)
        shape = b2array.shape
        dtype = b2array.dtype

        row_metadata, column_names, extra_metadata = load_metadata(file_path, b2array)
        return cls(
            file_path=file_path,
            shape=shape,
            dtype=dtype,
            num_threads=num_threads,
            compression_method=Blosc2CompressionMethod(),
            row_metadata=row_metadata,
            column_names=column_names,
            extra_metadata=extra_metadata,
        )
