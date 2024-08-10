import pickle
from typing import Any, Callable, override

from numpy import typing as npt
from upath import UPath

from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from .pipe import CompressedBytesReader, CompressedBytesWriter

cache_suffix: str = ".pickle.zst"


def load_from_cache(cache_path: UPath, key: str, sw: SharedWorkspace) -> Any:
    file_path = cache_path / f"{key}{cache_suffix}"
    if not file_path.is_file():
        logger.debug(f'Cache entry "{file_path}" not found')
        return None
    with CompressedBytesReader(file_path) as file_handle:
        try:
            return SharedArrayUnpickler(sw, file_handle).load()
        except (pickle.UnpicklingError, EOFError) as error:
            logger.warning(f'Failed to load "{file_path}"', exc_info=error)
            return None


def save_to_cache(
    cache_path: UPath,
    key: str,
    value: Any,
    num_threads: int,
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    with CompressedBytesWriter(
        cache_path / f"{key}{cache_suffix}", num_threads
    ) as file_handle:
        SharedArrayPickler(file_handle).dump(value)


def rebuild_shared_array(
    constructor: type[SharedArray],
    array: npt.NDArray,
    metadata: dict[str, Any],
    sw: SharedWorkspace,
) -> SharedArray:
    return constructor.from_numpy(array, sw, **metadata)


def reduce_shared_array(
    array: SharedArray,
) -> tuple[
    Callable[
        [type[SharedArray], npt.NDArray, dict[str, Any], SharedWorkspace], SharedArray
    ],
    tuple[type[SharedArray], npt.NDArray, dict[str, Any], SharedWorkspace],
]:
    return rebuild_shared_array, (
        type(array),
        array.to_numpy(),
        array.to_metadata(),
        array.sw,
    )


shared_workspace_persistent_id: int = 0


class SharedArrayPickler(pickle.Pickler):
    @override
    def reducer_override(self, obj: Any):
        if isinstance(obj, SharedArray):
            return reduce_shared_array(obj)
        return NotImplemented

    @override
    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, SharedWorkspace):
            return shared_workspace_persistent_id
        return None


class SharedArrayUnpickler(pickle.Unpickler):
    sw: SharedWorkspace

    def __init__(self, sw: SharedWorkspace, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sw = sw

    @override
    def persistent_load(self, pid: int) -> SharedWorkspace:
        if pid == shared_workspace_persistent_id:
            return self.sw
        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")
