from typing import Callable, TypeVar

import numpy as np
from numpy import typing as npt

T = TypeVar("T")

def read_str(
    row_list: list[T],
    row_parser: Callable[..., T],
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    ring_buffer_size: int = ...,
) -> None: ...
