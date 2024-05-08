# -*- coding: utf-8 -*-

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, Type

import blosc2
from blosc2.schunk import vlmeta
from numpy import typing as npt

from ...log import logger
from ..pipe import CompressedBytesReader
from .base import Blosc2CompressionMethod, FileArray, ScalarType


@dataclass
class Blosc2FileArray(FileArray[ScalarType]):
    compression_method: Blosc2CompressionMethod
    extra_metadata: dict[str, Any] | None = None

    array: blosc2.NDArray | None = None

    @property
    def cparams(self) -> dict[str, Any]:
        return dict(
            **self.compression_method.cparams,
            nthreads=self.num_threads,
        )

    @property
    def vlmeta(self) -> vlmeta:
        if self.array is None:
            raise RuntimeError("Blosc2 array not open")
        return self.array.schunk.vlmeta

    def __enter__(self) -> Self:
        return self.require_blosc2_ndarray()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.array = None

    @property
    def axis_metadata_path(self) -> Path:
        file_path = self.file_path
        return file_path.parent / f"{file_path.stem}.axis-metadata.pkl.zst"

    def get_vlmeta(self, key: str) -> Any:
        return pickle.loads(self.vlmeta.get_vlmeta(key))

    def set_vlmeta(self, key: str, value: Any) -> None:
        cparams = self.cparams
        if cparams.get("use_dict") is True:
            # This option is apparently not supported for vlmeta
            del cparams["use_dict"]
        self.vlmeta.set_vlmeta(key, pickle.dumps(value), **cparams)

    def load_metadata(self) -> None:
        try:
            self.axis_metadata = self.get_vlmeta("axis_metadata")
        except KeyError:
            if self.axis_metadata_path.is_file():
                with CompressedBytesReader(self.axis_metadata_path) as file_handle:
                    self.axis_metadata = pickle.load(file_handle)
            else:
                self.axis_metadata = [None] * len(self.shape)

        keys = set(self.vlmeta.get_names()) - {"axis_metadata"}
        if keys:
            self.extra_metadata = {}
            for key in keys:
                self.extra_metadata[key] = self.get_vlmeta(key)

    def dump_metadata(self) -> None:
        self.set_vlmeta("axis_metadata", self.axis_metadata)
        if self.extra_metadata is not None:
            for key, value in self.extra_metadata.items():
                self.set_vlmeta(key, value)

    def __setitem__(
        self, key: slice | tuple[slice, ...], value: npt.NDArray[ScalarType]
    ) -> None:
        if isinstance(self.compression_method, Blosc2CompressionMethod):
            if self.array is not None:
                self.array[key] = value
            else:
                raise RuntimeError("Blosc2 array not initialized")
        else:
            raise NotImplementedError(
                f"Unknown compression method {self.compression_method}"
            )

    @classmethod
    def get_file_path_with_suffix(cls, file_path: Path) -> Path:
        file_path = file_path
        if not str(file_path).endswith(Blosc2CompressionMethod.suffix):
            file_path = (
                file_path.parent / f"{file_path.name}{Blosc2CompressionMethod.suffix}"
            )
        return file_path

    @classmethod
    def open(cls, file_path: Path, num_threads: int = 1) -> blosc2.NDArray:
        file_path = cls.get_file_path_with_suffix(file_path)
        return blosc2.open(
            urlpath=str(file_path),
            cparams=dict(nthreads=num_threads),
            dparams=dict(nthreads=num_threads),
        )

    def require_blosc2_ndarray(self) -> Self:
        if not isinstance(self.compression_method, Blosc2CompressionMethod):
            raise RuntimeError(
                "Tried to get Blosc2 array with non-Blosc2 compression method"
            )

        file_path = self.get_file_path_with_suffix(self.file_path)
        self.file_paths.add(file_path)

        try:
            self.array = self.open(file_path, self.num_threads)
        except FileNotFoundError:
            pass
        if self.array is None:
            self.array = blosc2.empty(
                shape=self.shape,
                urlpath=str(file_path),
                dtype=self.dtype,
                cparams=self.cparams,
                dparams=dict(
                    nthreads=self.num_threads,
                ),
            )
            logger.debug(f"Created Blosc2 array: {self.array}")
            self.dump_metadata()

        array = self.array

        logger.debug(f"Using Blosc2 array {array.info}")

        if array.shape != self.shape or array.dtype != self.dtype:
            raise ValueError(
                f'Existing array at "{self.file_path}" with shape {array.shape} and '
                f"dtype {array.dtype} does not match required shape {self.shape} or "
                f"dtype {self.dtype}. Please delete the file or use a different "
                "output directory"
            )

        return self

    @classmethod
    def from_file(cls, file_path: Path, num_threads: int = 1) -> Self:
        b2array = cls.open(file_path, num_threads)
        shape = b2array.shape
        dtype = b2array.dtype

        cparams = b2array.schunk.cparams
        codec = cparams["codec"]
        clevel = cparams["clevel"]
        filters = tuple(f for f in cparams["filters"] if f != blosc2.Filter.NOFILTER)
        use_dict = cparams["use_dict"]
        compression_method = Blosc2CompressionMethod(
            codec=codec, clevel=clevel, filters=filters, use_dict=use_dict
        )

        array = cls(file_path, shape, dtype, compression_method, num_threads=num_threads)

        # Load metadata
        with array:
            array.load_metadata()

        return array
